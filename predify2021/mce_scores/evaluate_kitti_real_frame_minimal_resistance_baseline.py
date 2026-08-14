import csv
import json
import os
import random
from pathlib import Path

import torch

from predify2021.mce_scores.evaluate_kitti_real_frame_pc_phase1 import (
    ALLOWED_DRIVES,
    BASELINE_FRAMES,
    DISTURBANCE_FRAMES,
    FORBIDDEN_TEST_DRIVE_IDS,
    RECOVERY_FRAMES,
    phase_for_frame,
)
from predify2021.mce_scores.evaluate_kitti_real_frame_recurrent_error import (
    DEVICE,
    build_model as build_real_frame_model,
    sha256_file,
)
from predify2021.mce_scores.evaluate_kitti_real_frame_recurrent_error_cross_corruption import (
    build_drive_datasets,
)
from predify2021.mce_scores.evaluate_kitti_real_frame_robustness_benchmark import (
    FrozenVGGAdapter,
    OriginalPredifyAdapter,
)
from predify2021.mce_scores.kitti_controlled_corruption import (
    ControlledCorruptionConfig,
)


MODEL_NAMES = (
    "frozen_vgg16",
    "original_predify",
    "temporal_only",
)
CORRUPTIONS = {
    "gaussian_blur": ControlledCorruptionConfig(
        corruption_type="gaussian_blur",
        blur_kernel_size=11,
        blur_sigma=3.0,
    ),
    "brightness_overexposure": ControlledCorruptionConfig(
        corruption_type="bias",
        bias_rgb=(0.15, 0.15, 0.15),
    ),
}


class TemporalOnlyAdapter:
    def __init__(self, weights_path, checkpoint):
        self.model = build_real_frame_model(
            weights_path,
            "convgru_error",
            checkpoint=checkpoint,
            recurrent_input="temporal_only",
        ).to(DEVICE).eval()
        self.representations = None

    def reset(self):
        self.model.reset()
        self.representations = None

    def step(self, frame):
        self.model.step_frame(frame)
        self.representations = tuple(
            state.representation for state in self.model.layer_states
        )

    def sync_calibration_to(self, other):
        with torch.no_grad():
            other.model.pc_error_c_sqrt.copy_(self.model.pc_error_c_sqrt)


def normalized_representation_distance(candidate, reference):
    numerator = torch.linalg.vector_norm((candidate - reference).float())
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    return float((numerator / denominator).item())


def build_adapter(model_name, weights_path, temporal_only_checkpoint):
    if model_name == "frozen_vgg16":
        return FrozenVGGAdapter()
    if model_name == "original_predify":
        return OriginalPredifyAdapter(weights_path)
    if model_name == "temporal_only":
        return TemporalOnlyAdapter(weights_path, temporal_only_checkpoint)
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(clean, corrupted, datasets, model_name, corruption_name):
    rows = []
    calibration_synced = model_name in {"frozen_vgg16", "temporal_only"}
    for drive, (clean_dataset, corrupted_dataset, raw_frames) in datasets.items():
        clean.reset()
        corrupted.reset()
        for frame_offset, raw_frame_index in enumerate(raw_frames):
            clean_frame = clean_dataset._load_frame(
                clean_dataset.frame_paths[raw_frame_index]
            ).unsqueeze(0).to(DEVICE)
            corrupted_frame = corrupted_dataset._load_frame(
                corrupted_dataset.frame_paths[raw_frame_index]
            ).unsqueeze(0).to(DEVICE)

            clean.step(clean_frame)
            if not calibration_synced:
                clean.sync_calibration_to(corrupted)
                calibration_synced = True
            corrupted.step(corrupted_frame)

            phase, phase_frame_index = phase_for_frame(frame_offset)
            for layer, (clean_representation, corrupted_representation) in enumerate(
                zip(clean.representations, corrupted.representations),
                1,
            ):
                rows.append(
                    {
                        "model": model_name,
                        "corruption": corruption_name,
                        "drive": drive,
                        "frame_offset": frame_offset,
                        "phase": phase,
                        "phase_frame_index": phase_frame_index,
                        "raw_frame_index": raw_frame_index,
                        "frame_name": clean_dataset.frame_paths[raw_frame_index].name,
                        "layer": layer,
                        "representation_normalized_l2": normalized_representation_distance(
                            corrupted_representation,
                            clean_representation,
                        ),
                    }
                )
            del clean_frame, corrupted_frame
    return rows


def mean_distance(rows):
    return sum(float(row["representation_normalized_l2"]) for row in rows) / len(rows)


def summarize_rows(rows, drives):
    disturbance = [row for row in rows if row["phase"] == "disturbance"]
    recovery_first = [
        row
        for row in rows
        if row["phase"] == "recovery" and int(row["phase_frame_index"]) < 10
    ]
    recovery_last = [
        row
        for row in rows
        if row["phase"] == "recovery"
        and int(row["phase_frame_index"]) >= RECOVERY_FRAMES - 10
    ]
    return {
        "disturbance_mean_representation_normalized_l2": mean_distance(disturbance),
        "recovery_first_10_mean_representation_normalized_l2": mean_distance(
            recovery_first
        ),
        "recovery_last_10_mean_representation_normalized_l2": mean_distance(
            recovery_last
        ),
        "per_drive": {
            drive: {
                "disturbance_mean_representation_normalized_l2": mean_distance(
                    [row for row in disturbance if row["drive"] == drive]
                ),
                "recovery_first_10_mean_representation_normalized_l2": mean_distance(
                    [row for row in recovery_first if row["drive"] == drive]
                ),
                "recovery_last_10_mean_representation_normalized_l2": mean_distance(
                    [row for row in recovery_last if row["drive"] == drive]
                ),
            }
            for drive in drives
        },
        "per_layer": {
            str(layer): {
                "disturbance_mean_representation_normalized_l2": mean_distance(
                    [row for row in disturbance if row["layer"] == layer]
                )
            }
            for layer in range(1, 6)
        },
    }


def improvement_percent(reference, candidate):
    return 100.0 * (reference - candidate) / reference


def build_comparison(corruption_summary):
    frozen = corruption_summary["frozen_vgg16"][
        "disturbance_mean_representation_normalized_l2"
    ]
    original = corruption_summary["original_predify"][
        "disturbance_mean_representation_normalized_l2"
    ]
    temporal = corruption_summary["temporal_only"][
        "disturbance_mean_representation_normalized_l2"
    ]
    return {
        "metric": "disturbance_mean_representation_normalized_l2",
        "frozen_vgg16": frozen,
        "original_predify": original,
        "temporal_only": temporal,
        "predify_vs_vgg_improvement_percent": improvement_percent(frozen, original),
        "temporal_vs_predify_improvement_percent": improvement_percent(
            original, temporal
        ),
        "temporal_vs_vgg_improvement_percent": improvement_percent(frozen, temporal),
    }


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path, summary):
    lines = [
        "# Minimal Anti-corruption Baseline",
        "",
        "Evaluator-only comparison on Val drives 0011/0039. All models use the "
        "same frames, corruption implementation, and paired clean/corrupted "
        "reference protocol. Frozen Test drives 0051/0056 were not read.",
        "",
        "| Model | Blur disturbance | Brightness disturbance | Recovery first10 | Recovery last10 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in MODEL_NAMES:
        blur = summary["corruptions"]["gaussian_blur"]["models"][model]
        brightness = summary["corruptions"]["brightness_overexposure"]["models"][
            model
        ]
        lines.append(
            f"| {model} | "
            f"{blur['disturbance_mean_representation_normalized_l2']:.9f} | "
            f"{brightness['disturbance_mean_representation_normalized_l2']:.9f} | "
            f"{summary['model_macro_recovery'][model]['recovery_first_10_mean_representation_normalized_l2']:.9f} | "
            f"{summary['model_macro_recovery'][model]['recovery_last_10_mean_representation_normalized_l2']:.9f} |"
        )
    lines.append("")
    for corruption in CORRUPTIONS:
        comparison = summary["corruptions"][corruption]["comparison"]
        lines.extend(
            [
                f"## {corruption}",
                "",
                f"Predify vs VGG improvement: {comparison['predify_vs_vgg_improvement_percent']:.6f}%.",
                f"Temporal-only vs Original Predify improvement: {comparison['temporal_vs_predify_improvement_percent']:.6f}%.",
                f"Temporal-only vs VGG improvement: {comparison['temporal_vs_vgg_improvement_percent']:.6f}%.",
                "",
                "| Drive | Frozen VGG16 | Original Predify | Temporal-only |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for drive in summary["protocol"]["val_drives"]:
            drive_id = drive.split("_drive_")[-1].split("_sync")[0]
            models = summary["corruptions"][corruption]["models"]
            lines.append(
                f"| {drive_id} | "
                f"{models['frozen_vgg16']['per_drive'][drive]['disturbance_mean_representation_normalized_l2']:.9f} | "
                f"{models['original_predify']['per_drive'][drive]['disturbance_mean_representation_normalized_l2']:.9f} | "
                f"{models['temporal_only']['per_drive'][drive]['disturbance_mean_representation_normalized_l2']:.9f} |"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def write_manifest(path, summary):
    lines = [
        "experiment=minimal_anti_corruption_baseline",
        f"evaluation_revision={summary['evaluation_revision']}",
        "models=frozen_vgg16,original_predify,temporal_only",
        "corruptions=gaussian_blur,brightness_overexposure",
        "protocol=40 clean / 80 corruption / 40 recovery",
        f"val_drives={','.join(summary['protocol']['val_drives'])}",
        "frozen_test_drives_not_read=2011_09_26_drive_0051_sync,2011_09_26_drive_0056_sync",
        f"temporal_only_checkpoint={summary['temporal_only_checkpoint']['path']}",
        f"temporal_only_checkpoint_sha256={summary['temporal_only_checkpoint']['sha256']}",
        "training=false",
        "tuning=false",
        "checkpoint_selection=false",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("Formal KITTI evaluator requires GPU 0.")
    revision = os.environ["PREDIFY_GIT_REVISION"]
    output_dir = Path(os.environ["PREDIFY_MINIMAL_BASELINE_OUTPUT_DIR"])
    weights_path = os.environ["PREDIFY_PCODER_WEIGHTS"]
    root = os.environ["PREDIFY_KITTI_ROOT"]
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
    seed = int(os.environ.get("PREDIFY_SEED", "0"))
    drives = tuple(
        drive.strip()
        for drive in os.environ["PREDIFY_MINIMAL_BASELINE_DRIVES"].split(",")
        if drive.strip()
    )
    if drives != ALLOWED_DRIVES:
        raise ValueError(f"Drives must be exactly {ALLOWED_DRIVES}.")
    if any(forbidden in drive for forbidden in FORBIDDEN_TEST_DRIVE_IDS for drive in drives):
        raise RuntimeError("Evaluator attempted to read Frozen Test.")
    temporal_only_checkpoint = Path(os.environ["PREDIFY_TEMPORAL_ONLY_CHECKPOINT"])

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    all_rows = []
    results = {corruption: {"models": {}} for corruption in CORRUPTIONS}
    for model_name in MODEL_NAMES:
        print(f"Building model {model_name}...", flush=True)
        clean = build_adapter(model_name, weights_path, temporal_only_checkpoint)
        corrupted = build_adapter(model_name, weights_path, temporal_only_checkpoint)
        for corruption_name, config in CORRUPTIONS.items():
            print(
                f"Evaluating model={model_name} corruption={corruption_name}",
                flush=True,
            )
            datasets = {
                drive: build_drive_datasets(
                    root, drive, camera, fixed_dt_s, tolerance, config
                )
                for drive in drives
            }
            rows = evaluate_model(
                clean, corrupted, datasets, model_name, corruption_name
            )
            all_rows.extend(rows)
            results[corruption_name]["models"][model_name] = summarize_rows(
                rows, drives
            )
        del clean, corrupted
        torch.cuda.empty_cache()

    for corruption_name in CORRUPTIONS:
        results[corruption_name]["comparison"] = build_comparison(
            results[corruption_name]["models"]
        )

    model_macro_recovery = {}
    for model_name in MODEL_NAMES:
        first = [
            results[corruption]["models"][model_name][
                "recovery_first_10_mean_representation_normalized_l2"
            ]
            for corruption in CORRUPTIONS
        ]
        last = [
            results[corruption]["models"][model_name][
                "recovery_last_10_mean_representation_normalized_l2"
            ]
            for corruption in CORRUPTIONS
        ]
        model_macro_recovery[model_name] = {
            "recovery_first_10_mean_representation_normalized_l2": sum(first)
            / len(first),
            "recovery_last_10_mean_representation_normalized_l2": sum(last)
            / len(last),
        }

    summary = {
        "experiment": "minimal_anti_corruption_baseline",
        "evaluation_revision": revision,
        "protocol": {
            "val_drives": drives,
            "frozen_test_drives_read": False,
            "baseline_frames": BASELINE_FRAMES,
            "disturbance_frames": DISTURBANCE_FRAMES,
            "recovery_frames": RECOVERY_FRAMES,
            "models": MODEL_NAMES,
            "corruptions": tuple(CORRUPTIONS),
            "seed": seed,
            "training": False,
            "tuning": False,
            "checkpoint_selection": False,
            "paired_reference": "each model uses its own clean trajectory",
            "original_predify_path": "legacy pvgg PVGG16SeparateHP PCoderN",
            "temporal_only_source_revision": "268e92698019344b9d091f50177819154ba9e87f",
        },
        "temporal_only_checkpoint": {
            "path": str(temporal_only_checkpoint),
            "sha256": sha256_file(temporal_only_checkpoint),
        },
        "corruptions": results,
        "model_macro_recovery": model_macro_recovery,
        "per_frame_row_count": len(all_rows),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    write_csv(output_dir / "per_frame.csv", all_rows)
    write_readme(output_dir / "README.md", summary)
    write_manifest(output_dir / "manifest.txt", summary)
    print(json.dumps(summary["corruptions"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
