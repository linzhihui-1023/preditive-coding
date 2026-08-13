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
    select_protocol_raw_frames,
)
from predify2021.mce_scores.evaluate_kitti_real_frame_recurrent_error import (
    CONDITIONS,
    DEVICE,
    build_model,
    condition_metrics,
    evaluate_condition,
    sha256_file,
)
from predify2021.mce_scores.evaluate_kitti_real_frame_recurrent_error_frozen_test import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_MODEL_REVISION,
    validate_checkpoint,
)
from predify2021.mce_scores.kitti_controlled_corruption import (
    ControlledCorruptionConfig,
    ControlledCorruptionKITTIDataset,
    ExplicitSeveritySchedule,
)


CORRUPTIONS = {
    "gaussian_noise": ControlledCorruptionConfig(
        corruption_type="iid_gaussian",
        bias_rgb=(0.0, 0.0, 0.0),
        noise_std=0.08,
        seed=0,
    ),
    "brightness_shift": ControlledCorruptionConfig(
        corruption_type="bias",
        bias_rgb=(0.15, 0.15, 0.15),
        noise_std=0.0,
        seed=0,
    ),
}


def build_drive_datasets(root, drive, camera, fixed_dt_s, tolerance, config):
    schedule = ExplicitSeveritySchedule(
        baseline_frames=BASELINE_FRAMES,
        disturbance_severities=(1.0,) * DISTURBANCE_FRAMES,
        recovery_frames=RECOVERY_FRAMES,
    )
    probe = ControlledCorruptionKITTIDataset(
        root,
        drive,
        camera=camera,
        horizons=(1,),
        fixed_dt_s=fixed_dt_s,
        dt_tolerance_s=tolerance,
        schedule=schedule,
        corruption_config=config,
        corruption_enabled=False,
    )
    raw_frames = select_protocol_raw_frames(probe)
    schedule_start = raw_frames[0]
    clean = ControlledCorruptionKITTIDataset(
        root,
        drive,
        camera=camera,
        horizons=(1,),
        fixed_dt_s=fixed_dt_s,
        dt_tolerance_s=tolerance,
        schedule=schedule,
        corruption_config=config,
        schedule_start_raw_index=schedule_start,
        corruption_enabled=False,
    )
    corrupted = ControlledCorruptionKITTIDataset(
        root,
        drive,
        camera=camera,
        horizons=(1,),
        fixed_dt_s=fixed_dt_s,
        dt_tolerance_s=tolerance,
        schedule=schedule,
        corruption_config=config,
        schedule_start_raw_index=schedule_start,
        corruption_enabled=True,
    )
    return clean, corrupted, raw_frames


def summarize_comparison(metrics, drives):
    current = metrics["current_stateful"]
    learned = metrics["learned_recurrent_error"]
    zeroed = metrics["learned_recurrent_error_zeroed"]

    def improvement(reference, candidate):
        return 100.0 * (reference - candidate) / reference

    per_drive = {}
    for drive in drives:
        current_value = current["per_drive"][drive][
            "disturbance_mean_representation_normalized_l2"
        ]
        learned_value = learned["per_drive"][drive][
            "disturbance_mean_representation_normalized_l2"
        ]
        zeroed_value = zeroed["per_drive"][drive][
            "disturbance_mean_representation_normalized_l2"
        ]
        per_drive[drive] = {
            "current_stateful": current_value,
            "learned_recurrent_error": learned_value,
            "learned_recurrent_error_zeroed": zeroed_value,
            "learned_vs_current_improvement_percent": improvement(
                current_value, learned_value
            ),
            "learned_vs_zeroed_improvement_percent": improvement(
                zeroed_value, learned_value
            ),
        }

    current_value = current["disturbance_mean_representation_normalized_l2"]
    learned_value = learned["disturbance_mean_representation_normalized_l2"]
    zeroed_value = zeroed["disturbance_mean_representation_normalized_l2"]
    return {
        "metric": "disturbance_mean_representation_normalized_l2",
        "current_stateful": current_value,
        "learned_recurrent_error": learned_value,
        "learned_recurrent_error_zeroed": zeroed_value,
        "learned_vs_current_improvement_percent": improvement(
            current_value, learned_value
        ),
        "learned_vs_zeroed_improvement_percent": improvement(
            zeroed_value, learned_value
        ),
        "per_drive": per_drive,
    }


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path, summary):
    lines = [
        "# Recurrent-error Cross-corruption Validation",
        "",
        "Frozen c9fec52 epoch-1 checkpoint on Val drives 0011/0039. Each "
        "corruption uses 40 clean, 80 persistent disturbed, and 40 recovery "
        "frames. No training, tuning, or checkpoint selection occurred.",
        "",
    ]
    for corruption in CORRUPTIONS:
        result = summary["corruptions"][corruption]
        lines.extend(
            [
                f"## {corruption}",
                "",
                "| Condition | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for condition in CONDITIONS:
            item = result["conditions"][condition]
            lines.append(
                f"| {condition} | "
                f"{item['disturbance_mean_representation_normalized_l2']:.9f} | "
                f"{item['recovery_first_10_mean_representation_normalized_l2']:.9f} | "
                f"{item['recovery_last_10_mean_representation_normalized_l2']:.9f} |"
            )
        comparison = result["comparison"]
        lines.extend(
            [
                "",
                f"Learned vs current: {comparison['learned_vs_current_improvement_percent']:.6f}%.",
                f"Learned vs zeroed: {comparison['learned_vs_zeroed_improvement_percent']:.6f}%.",
                "",
                "| Drive | Learned vs current | Learned vs zeroed |",
                "| --- | ---: | ---: |",
            ]
        )
        for drive in summary["protocol"]["val_drives"]:
            item = comparison["per_drive"][drive]
            drive_id = drive.split("_drive_")[-1].split("_sync")[0]
            lines.append(
                f"| {drive_id} | "
                f"{item['learned_vs_current_improvement_percent']:.6f}% | "
                f"{item['learned_vs_zeroed_improvement_percent']:.6f}% |"
            )
        lines.extend(
            [
                "",
                "| Layer | Current | Learned | Zeroed |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for layer in range(1, 6):
            key = str(layer)
            lines.append(
                f"| {layer} | "
                f"{result['conditions']['current_stateful']['per_layer'][key]['disturbance_mean_representation_normalized_l2']:.9f} | "
                f"{result['conditions']['learned_recurrent_error']['per_layer'][key]['disturbance_mean_representation_normalized_l2']:.9f} | "
                f"{result['conditions']['learned_recurrent_error_zeroed']['per_layer'][key]['disturbance_mean_representation_normalized_l2']:.9f} |"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="ascii")


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("Formal cross-corruption validation requires GPU 0.")
    revision = os.environ["PREDIFY_GIT_REVISION"]
    output_dir = Path(os.environ["PREDIFY_CROSS_CORRUPTION_OUTPUT_DIR"])
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
        for drive in os.environ["PREDIFY_CROSS_CORRUPTION_DRIVES"].split(",")
        if drive.strip()
    )
    if drives != ALLOWED_DRIVES:
        raise ValueError(f"Cross-corruption drives must be exactly {ALLOWED_DRIVES}.")
    if any(forbidden in drive for forbidden in FORBIDDEN_TEST_DRIVE_IDS for drive in drives):
        raise RuntimeError("Cross-corruption validation attempted to read Frozen Test.")
    checkpoint_payload = validate_checkpoint(checkpoint)
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Cross-corruption checkpoint hash changed.")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    models = {
        "current_stateful": build_model(weights_path, "predify"),
        "learned_recurrent_error": build_model(
            weights_path, "convgru_error", checkpoint
        ),
        "learned_recurrent_error_zeroed": build_model(
            weights_path,
            "convgru_error",
            checkpoint,
            recurrent_error_input="zeroed",
        ),
    }

    all_rows = []
    corruption_results = {}
    selected_raw_frames = {}
    output_dir.mkdir(parents=True, exist_ok=False)
    for corruption, config in CORRUPTIONS.items():
        datasets = {
            drive: build_drive_datasets(
                root, drive, camera, fixed_dt_s, tolerance, config
            )
            for drive in drives
        }
        rows = []
        for condition in CONDITIONS:
            print(f"Evaluating {corruption}: {condition}...", flush=True)
            condition_rows = evaluate_condition(
                models[condition], condition, datasets, seed
            )
            for row in condition_rows:
                row["corruption"] = corruption
            rows.extend(condition_rows)
        metrics = {
            condition: condition_metrics(
                [row for row in rows if row["condition"] == condition]
            )
            for condition in CONDITIONS
        }
        result = {
            "corruption": corruption,
            "config": config.to_dict(),
            "conditions": metrics,
            "comparison": summarize_comparison(metrics, drives),
        }
        corruption_results[corruption] = result
        selected_raw_frames[corruption] = {
            drive: {
                "start": raw_frames[0],
                "stop_inclusive": raw_frames[-1],
                "count": len(raw_frames),
            }
            for drive, (_, _, raw_frames) in datasets.items()
        }
        corruption_dir = output_dir / corruption
        corruption_dir.mkdir()
        write_csv(corruption_dir / "per_frame.csv", rows)
        with (corruption_dir / "summary.json").open("w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        all_rows.extend(rows)

    macro_conditions = {}
    for condition in CONDITIONS:
        macro_conditions[condition] = {
            field: sum(
                corruption_results[corruption]["conditions"][condition][field]
                for corruption in CORRUPTIONS
            )
            / len(CORRUPTIONS)
            for field in (
                "disturbance_mean_representation_normalized_l2",
                "recovery_first_10_mean_representation_normalized_l2",
                "recovery_last_10_mean_representation_normalized_l2",
            )
        }
    macro_comparison = summarize_comparison(
        {
            condition: {
                **macro_conditions[condition],
                "per_drive": {
                    drive: {
                        field: sum(
                            corruption_results[corruption]["conditions"][condition][
                                "per_drive"
                            ][drive][field]
                            for corruption in CORRUPTIONS
                        )
                        / len(CORRUPTIONS)
                        for field in (
                            "disturbance_mean_representation_normalized_l2",
                            "recovery_first_10_mean_representation_normalized_l2",
                            "recovery_last_10_mean_representation_normalized_l2",
                        )
                    }
                    for drive in drives
                },
            }
            for condition in CONDITIONS
        },
        drives,
    )
    summary = {
        "experiment": "real_frame_recurrent_error_cross_corruption_validation",
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
            "seed": seed,
            "training": False,
            "tuning": False,
            "checkpoint_selection": False,
            "corruptions": tuple(CORRUPTIONS),
        },
        "corruptions": corruption_results,
        "overall_macro_mean": {
            "conditions": macro_conditions,
            "comparison": macro_comparison,
        },
        "selected_raw_frames": selected_raw_frames,
        "per_frame_row_count": len(all_rows),
    }
    write_csv(output_dir / "per_frame.csv", all_rows)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    write_readme(output_dir / "README.md", summary)
    print(
        json.dumps(
            {
                corruption: result["comparison"]
                for corruption, result in corruption_results.items()
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
