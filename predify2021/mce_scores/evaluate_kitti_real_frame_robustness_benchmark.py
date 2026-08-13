import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from torchvision.models import VGG16_Weights, vgg16

from predify2021.mce_scores.evaluate_kitti_real_frame_pc_phase1 import (
    ALLOWED_DRIVES,
    BASELINE_FRAMES,
    DISTURBANCE_FRAMES,
    FORBIDDEN_TEST_DRIVE_IDS,
    RECOVERY_FRAMES,
    normalized_representation_distance,
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
from predify2021.mce_scores.evaluate_kitti_real_frame_recurrent_error_frozen_test import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_MODEL_REVISION,
    validate_checkpoint,
)
from predify2021.mce_scores.kitti_controlled_corruption import (
    ControlledCorruptionConfig,
)
from predify2021.model_factory import get_model


MODEL_NAMES = (
    "frozen_vgg16",
    "original_predify",
    "current_stateful",
    "learned_recurrent_error_zeroed",
    "learned_recurrent_error",
)
LEGACY_INTERNAL_TIMESTEPS = 10
SEVERITIES = {
    "gaussian_blur": (
        ControlledCorruptionConfig(corruption_type="gaussian_blur", blur_sigma=1.0),
        ControlledCorruptionConfig(corruption_type="gaussian_blur", blur_sigma=2.0),
        ControlledCorruptionConfig(corruption_type="gaussian_blur", blur_sigma=3.0),
    ),
    "gaussian_noise": (
        ControlledCorruptionConfig(corruption_type="iid_gaussian", noise_std=0.03),
        ControlledCorruptionConfig(corruption_type="iid_gaussian", noise_std=0.055),
        ControlledCorruptionConfig(corruption_type="iid_gaussian", noise_std=0.08),
    ),
    "brightness": (
        ControlledCorruptionConfig(corruption_type="bias", bias_rgb=(0.05,) * 3),
        ControlledCorruptionConfig(corruption_type="bias", bias_rgb=(0.10,) * 3),
        ControlledCorruptionConfig(corruption_type="bias", bias_rgb=(0.15,) * 3),
    ),
    "motion_blur": (
        ControlledCorruptionConfig(corruption_type="motion_blur", motion_blur_kernel_size=5),
        ControlledCorruptionConfig(corruption_type="motion_blur", motion_blur_kernel_size=9),
        ControlledCorruptionConfig(corruption_type="motion_blur", motion_blur_kernel_size=13),
    ),
    "contrast": (
        ControlledCorruptionConfig(corruption_type="contrast", contrast_factor=0.8),
        ControlledCorruptionConfig(corruption_type="contrast", contrast_factor=0.6),
        ControlledCorruptionConfig(corruption_type="contrast", contrast_factor=0.4),
    ),
    "fog": (
        ControlledCorruptionConfig(corruption_type="fog", fog_alpha=0.1),
        ControlledCorruptionConfig(corruption_type="fog", fog_alpha=0.2),
        ControlledCorruptionConfig(corruption_type="fog", fog_alpha=0.3),
    ),
    "jpeg_compression": (
        ControlledCorruptionConfig(corruption_type="jpeg_compression", jpeg_quality=80),
        ControlledCorruptionConfig(corruption_type="jpeg_compression", jpeg_quality=50),
        ControlledCorruptionConfig(corruption_type="jpeg_compression", jpeg_quality=20),
    ),
}


class FrozenVGGAdapter:
    def __init__(self):
        self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(DEVICE).eval()
        self.model.requires_grad_(False)
        self.representations = None

    def reset(self):
        self.representations = None

    def step(self, frame):
        outputs = []
        current = frame
        endpoints = {3, 8, 15, 22, 29}
        with torch.no_grad():
            for index, module in enumerate(self.model.features):
                current = module(current)
                if index in endpoints:
                    outputs.append(current.detach())
                if index == 29:
                    break
        self.representations = tuple(outputs)

    def sync_calibration_to(self, other):
        return None


class OriginalPredifyAdapter:
    def __init__(self, weights_path):
        self.model = get_model(
            "pvgg",
            pretrained=True,
            deep_graph=False,
            pcoder_weights=weights_path,
        ).to(DEVICE).eval()
        self.model.requires_grad_(False)
        self.representations = None

    def reset(self):
        self.model.reset()
        self.representations = None

    def step(self, frame):
        self.model.reset()
        self.model(frame)
        for _ in range(LEGACY_INTERNAL_TIMESTEPS):
            self.model(None)
        self.representations = tuple(
            getattr(self.model, f"pcoder{layer}").rep.detach()
            for layer in range(1, 6)
        )

    def sync_calibration_to(self, other):
        with torch.no_grad():
            for layer in range(1, 6):
                source = getattr(self.model, f"pcoder{layer}").C_sqrt
                getattr(other.model, f"pcoder{layer}").C_sqrt.copy_(source)


class RealFrameAdapter:
    def __init__(self, mode, weights_path, checkpoint=None, error_input="dynamic"):
        self.model = build_real_frame_model(
            weights_path,
            mode,
            checkpoint=checkpoint,
            recurrent_error_input=error_input,
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


def build_adapter(model_name, weights_path, checkpoint):
    if model_name == "frozen_vgg16":
        return FrozenVGGAdapter()
    if model_name == "original_predify":
        return OriginalPredifyAdapter(weights_path)
    if model_name == "current_stateful":
        return RealFrameAdapter("predify", weights_path)
    if model_name == "learned_recurrent_error_zeroed":
        return RealFrameAdapter(
            "convgru_error",
            weights_path,
            checkpoint=checkpoint,
            error_input="zeroed",
        )
    if model_name == "learned_recurrent_error":
        return RealFrameAdapter(
            "convgru_error",
            weights_path,
            checkpoint=checkpoint,
        )
    raise ValueError(f"Unknown benchmark model: {model_name}")


def severity_parameters(config):
    fields = {
        "gaussian_blur": {"blur_sigma": config.blur_sigma},
        "iid_gaussian": {"noise_std": config.noise_std},
        "bias": {"brightness_shift_rgb": config.bias_rgb},
        "motion_blur": {"motion_blur_kernel_size": config.motion_blur_kernel_size},
        "contrast": {"contrast_factor": config.contrast_factor},
        "fog": {"fog_alpha": config.fog_alpha},
        "jpeg_compression": {"jpeg_quality": config.jpeg_quality},
    }
    return fields[config.corruption_type]


def evaluate_trajectory(clean, corrupted, datasets, model_name, corruption, severity):
    rows = []
    calibration_synced = model_name in {
        "frozen_vgg16",
        "learned_recurrent_error_zeroed",
        "learned_recurrent_error",
    }
    for drive, (clean_dataset, corrupted_dataset, raw_frames) in datasets.items():
        clean.reset()
        corrupted.reset()
        for frame_offset, raw_index in enumerate(raw_frames):
            clean_frame = clean_dataset._load_frame(
                clean_dataset.frame_paths[raw_index]
            ).unsqueeze(0).to(DEVICE)
            corrupted_frame = corrupted_dataset._load_frame(
                corrupted_dataset.frame_paths[raw_index]
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
                        "corruption": corruption,
                        "severity": severity,
                        "drive": drive,
                        "frame_offset": frame_offset,
                        "phase": phase,
                        "phase_frame_index": phase_frame_index,
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


def summarize_rows(rows):
    disturbance = [row for row in rows if row["phase"] == "disturbance"]
    recovery_first = [
        row
        for row in rows
        if row["phase"] == "recovery" and row["phase_frame_index"] < 10
    ]
    recovery_last = [
        row
        for row in rows
        if row["phase"] == "recovery"
        and row["phase_frame_index"] >= RECOVERY_FRAMES - 10
    ]
    return {
        "disturbance_mean_normalized_l2": mean_distance(disturbance),
        "recovery_first_10_mean_normalized_l2": mean_distance(recovery_first),
        "recovery_last_10_mean_normalized_l2": mean_distance(recovery_last),
        "per_drive": {
            drive: {
                "disturbance_mean_normalized_l2": mean_distance(
                    [row for row in disturbance if row["drive"] == drive]
                ),
                "recovery_first_10_mean_normalized_l2": mean_distance(
                    [row for row in recovery_first if row["drive"] == drive]
                ),
                "recovery_last_10_mean_normalized_l2": mean_distance(
                    [row for row in recovery_last if row["drive"] == drive]
                ),
            }
            for drive in ALLOWED_DRIVES
        },
        "per_layer": {
            str(layer): {
                "disturbance_mean_normalized_l2": mean_distance(
                    [row for row in disturbance if row["layer"] == layer]
                )
            }
            for layer in range(1, 6)
        },
    }


def build_aggregate_summary(results):
    corruption_means = {model: {} for model in MODEL_NAMES}
    for model in MODEL_NAMES:
        for corruption in SEVERITIES:
            corruption_means[model][corruption] = sum(
                results[corruption][str(severity)][model][
                    "disturbance_mean_normalized_l2"
                ]
                for severity in range(1, 4)
            ) / 3.0
    return {
        model: {
            "corruption_mean_disturbance": corruption_means[model],
            "mean_normalized_representation_deviation": sum(
                corruption_means[model].values()
            )
            / len(SEVERITIES),
        }
        for model in MODEL_NAMES
    }


def write_readme(path, summary):
    lines = [
        "# KITTI Real-frame Robustness Benchmark",
        "",
        "Val drives 0011/0039 only. Every model is compared with its own paired "
        "clean trajectory under a 40-clean/80-corruption/40-recovery protocol. "
        "Values are normalized representation deviations, not mCE.",
        "",
        "Original Predify uses the legacy pvgg PCoder path and independent "
        "per-frame t=0..10 internal inference. It is not current-stateful.",
        "",
        "| Model | " + " | ".join(SEVERITIES) + " | mean_normalized_representation_deviation |",
        "| --- | " + " | ".join("---:" for _ in SEVERITIES) + " | ---: |",
    ]
    for model in MODEL_NAMES:
        aggregate = summary["model_aggregate"][model]
        lines.append(
            f"| {model} | "
            + " | ".join(
                f"{aggregate['corruption_mean_disturbance'][corruption]:.9f}"
                for corruption in SEVERITIES
            )
            + f" | {aggregate['mean_normalized_representation_deviation']:.9f} |"
        )
    lines.append("")
    for corruption in SEVERITIES:
        lines.extend([f"## {corruption}", ""])
        for severity in range(1, 4):
            lines.extend(
                [
                    f"### Severity {severity}",
                    "",
                    "| Model | Disturbance | Recovery first 10 | Recovery last 10 |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for model in MODEL_NAMES:
                item = summary["results"][corruption][str(severity)][model]
                lines.append(
                    f"| {model} | {item['disturbance_mean_normalized_l2']:.9f} | "
                    f"{item['recovery_first_10_mean_normalized_l2']:.9f} | "
                    f"{item['recovery_last_10_mean_normalized_l2']:.9f} |"
                )
            lines.append("")
    Path(path).write_text("\n".join(lines), encoding="ascii")


def run_smoke(root, drives, camera, fixed_dt_s, tolerance, weights_path, checkpoint):
    new_corruptions = ("motion_blur", "contrast", "fog", "jpeg_compression")
    for model_name in MODEL_NAMES:
        clean = build_adapter(model_name, weights_path, checkpoint)
        corrupted = build_adapter(model_name, weights_path, checkpoint)
        for corruption in new_corruptions:
            config = SEVERITIES[corruption][1]
            datasets = {
                drive: build_drive_datasets(
                    root, drive, camera, fixed_dt_s, tolerance, config
                )
                for drive in drives[:1]
            }
            drive = drives[0]
            clean_dataset, corrupted_dataset, raw_frames = datasets[drive]
            raw_index = raw_frames[BASELINE_FRAMES]
            clean.reset()
            corrupted.reset()
            clean_frame = clean_dataset._load_frame(
                clean_dataset.frame_paths[raw_index]
            ).unsqueeze(0).to(DEVICE)
            corrupted_frame = corrupted_dataset._load_frame(
                corrupted_dataset.frame_paths[raw_index]
            ).unsqueeze(0).to(DEVICE)
            clean.step(clean_frame)
            clean.sync_calibration_to(corrupted)
            corrupted.step(corrupted_frame)
            if len(clean.representations) != 5 or len(corrupted.representations) != 5:
                raise RuntimeError("Smoke did not produce five representations.")
            print(f"smoke_ok model={model_name} corruption={corruption}", flush=True)
        del clean, corrupted
        torch.cuda.empty_cache()


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("Formal robustness benchmark requires GPU 0.")
    revision = os.environ["PREDIFY_GIT_REVISION"]
    output_dir = Path(os.environ["PREDIFY_ROBUSTNESS_OUTPUT_DIR"])
    training_dir = Path(os.environ["PREDIFY_RECURRENT_ERROR_TRAIN_OUTPUT_DIR"])
    checkpoint = training_dir / "best_recurrent_transition.pt"
    weights_path = os.environ["PREDIFY_PCODER_WEIGHTS"]
    root = os.environ["PREDIFY_KITTI_ROOT"]
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
    seed = int(os.environ.get("PREDIFY_SEED", "0"))
    drives = tuple(
        drive.strip()
        for drive in os.environ["PREDIFY_ROBUSTNESS_DRIVES"].split(",")
        if drive.strip()
    )
    if drives != ALLOWED_DRIVES:
        raise ValueError(f"Benchmark drives must be exactly {ALLOWED_DRIVES}.")
    if any(forbidden in drive for forbidden in FORBIDDEN_TEST_DRIVE_IDS for drive in drives):
        raise RuntimeError("Robustness benchmark attempted to read Frozen Test.")
    checkpoint_payload = validate_checkpoint(checkpoint)
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Benchmark checkpoint hash changed.")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if os.environ.get("PREDIFY_ROBUSTNESS_SMOKE", "0") == "1":
        run_smoke(
            root, drives, camera, fixed_dt_s, tolerance, weights_path, checkpoint
        )
        return

    results = {
        corruption: {str(severity): {} for severity in range(1, 4)}
        for corruption in SEVERITIES
    }
    severity_configs = {
        corruption: {
            str(index): severity_parameters(config)
            for index, config in enumerate(configs, 1)
        }
        for corruption, configs in SEVERITIES.items()
    }
    for model_name in MODEL_NAMES:
        print(f"Building benchmark model {model_name}...", flush=True)
        clean = build_adapter(model_name, weights_path, checkpoint)
        corrupted = build_adapter(model_name, weights_path, checkpoint)
        for corruption, configs in SEVERITIES.items():
            for severity, config in enumerate(configs, 1):
                print(
                    f"Evaluating model={model_name} corruption={corruption} severity={severity}",
                    flush=True,
                )
                datasets = {
                    drive: build_drive_datasets(
                        root, drive, camera, fixed_dt_s, tolerance, config
                    )
                    for drive in drives
                }
                rows = evaluate_trajectory(
                    clean,
                    corrupted,
                    datasets,
                    model_name,
                    corruption,
                    severity,
                )
                results[corruption][str(severity)][model_name] = summarize_rows(rows)
        del clean, corrupted
        torch.cuda.empty_cache()

    summary = {
        "experiment": "kitti_real_frame_unified_corruption_benchmark",
        "evaluator_revision": revision,
        "model_revision": EXPECTED_MODEL_REVISION,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": EXPECTED_CHECKPOINT_SHA256,
            "epoch": checkpoint_payload["epoch"],
            "val_prediction_mse": checkpoint_payload["val_prediction_mse"],
        },
        "protocol": {
            "val_drives": drives,
            "frozen_test_drives_read": False,
            "baseline_frames": BASELINE_FRAMES,
            "disturbance_frames": DISTURBANCE_FRAMES,
            "recovery_frames": RECOVERY_FRAMES,
            "models": MODEL_NAMES,
            "corruptions": tuple(SEVERITIES),
            "severity_count": 3,
            "seed": seed,
            "training": False,
            "tuning": False,
            "checkpoint_selection": False,
            "paired_reference": "each model uses its own clean trajectory",
            "original_predify_path": "legacy pvgg PVGG16SeparateHP PCoderN",
            "original_predify_internal_timesteps": LEGACY_INTERNAL_TIMESTEPS,
            "original_predify_real_frame_state": False,
            "aggregate_metric_name": "mean_normalized_representation_deviation",
            "explicitly_not_mce": True,
        },
        "severity_configs": severity_configs,
        "results": results,
        "model_aggregate": build_aggregate_summary(results),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    write_readme(output_dir / "README.md", summary)
    print(json.dumps(summary["model_aggregate"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
