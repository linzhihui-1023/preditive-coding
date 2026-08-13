import copy
import csv
import hashlib
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
    normalized_representation_distance,
    phase_for_frame,
    tensor_rms,
)
from predify2021.mce_scores.kitti_controlled_corruption import (
    ControlledCorruptionConfig,
    ControlledCorruptionKITTIDataset,
    ExplicitSeveritySchedule,
)
from predify2021.model_factory import get_model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONDITIONS = (
    "current_stateful",
    "temporal_only",
    "instant_error",
    "error_memory",
)
CORE_CONDITIONS = (
    "current_stateful",
    "temporal_only",
    "instant_error",
    "error_memory",
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


def build_model(
    weights_path,
    transition_mode,
    checkpoint=None,
    recurrent_input="dynamic",
):
    model = get_model(
        "pvgg_tf",
        pretrained=True,
        pcoder_weights=weights_path,
        task="real_frame_pc",
        dynamic_error=True,
        error_state_mode="ema",
        error_sample_time=0.1035,
        error_time_constant=(0.5,) * 5,
        error_gain=(1.0,) * 5,
        real_frame_transition_mode=transition_mode,
        real_frame_recurrent_error_input=recurrent_input,
    ).eval()
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.recurrent_transition_modules.load_state_dict(
            payload["recurrent_transition_state_dict"]
        )
        if payload.get("recurrent_error_encoder_state_dict") is not None:
            model.recurrent_error_encoder_modules.load_state_dict(
                payload["recurrent_error_encoder_state_dict"]
            )
    model.requires_grad_(False)
    return model


def checkpoint_for(training_dir, condition):
    return training_dir / f"best_{condition}.pt"


def select_protocol_raw_frames(dataset):
    total_frames = BASELINE_FRAMES + DISTURBANCE_FRAMES + RECOVERY_FRAMES
    for sample_segment in dataset.valid_sample_segments:
        starts = tuple(
            int(dataset.valid_start_indices[index]) for index in sample_segment
        )
        raw_frames = (*starts, starts[-1] + 1)
        if len(raw_frames) >= total_frames:
            return tuple(raw_frames[:total_frames])
    raise ValueError(f"No contiguous KITTI segment contains {total_frames} frames.")


def build_corruption_drive_datasets(
    root,
    drive,
    camera,
    fixed_dt_s,
    tolerance,
    config,
):
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


def assert_state(model, expected_frame_index):
    if model.real_frame_update_count != expected_frame_index:
        raise RuntimeError("A real frame did not produce exactly one update.")
    if model.recurrence_outputs["updates_per_layer"] != (1, 1, 1, 1, 1):
        raise RuntimeError("A PCoder layer updated more than once.")
    memories = (
        model.representation_state_memory
        + model.prediction_state_memory
        + model.instant_error_state_memory
        + model.error_state_memory
    )
    if any(memory.requires_grad or memory.grad_fn is not None for memory in memories):
        raise RuntimeError("Cross-frame state was not detached.")
    for state in model.layer_states:
        previous = (
            torch.zeros_like(state.instant_error)
            if state.previous_dynamic_error is None
            else state.previous_dynamic_error
        )
        expected = 0.207 * state.instant_error + 0.793 * previous
        if not torch.allclose(state.dynamic_error, expected, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Dynamic-error recurrence changed.")


def evaluate_condition(base_model, condition, corruption_name, datasets, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    clean_model = copy.deepcopy(base_model).to(DEVICE).eval()
    corrupted_model = copy.deepcopy(base_model).to(DEVICE).eval()
    rows = []
    c_sqrt_initialized = condition != "current_stateful"

    for drive, (clean_dataset, corrupted_dataset, raw_frames) in datasets.items():
        clean_model.reset()
        corrupted_model.reset()
        for frame_offset, raw_frame_index in enumerate(raw_frames):
            clean_frame = clean_dataset._load_frame(
                clean_dataset.frame_paths[raw_frame_index]
            ).unsqueeze(0).to(DEVICE)
            corrupted_frame = corrupted_dataset._load_frame(
                corrupted_dataset.frame_paths[raw_frame_index]
            ).unsqueeze(0).to(DEVICE)

            clean_model.step_frame(clean_frame)
            if not c_sqrt_initialized:
                corrupted_model.pc_error_c_sqrt.copy_(clean_model.pc_error_c_sqrt)
                c_sqrt_initialized = True
            corrupted_model.step_frame(corrupted_frame)
            assert_state(clean_model, frame_offset + 1)
            assert_state(corrupted_model, frame_offset + 1)
            next_frame_prediction_mses = [None] * len(corrupted_model.layer_states)
            if frame_offset + 1 < len(raw_frames):
                next_raw_frame_index = raw_frames[frame_offset + 1]
                next_frame = corrupted_dataset._load_frame(
                    corrupted_dataset.frame_paths[next_raw_frame_index]
                ).unsqueeze(0).to(DEVICE)
                next_targets, _ = corrupted_model._real_frame_feature_targets(
                    next_frame
                )
                next_frame_prediction_mses = [
                    float(
                        torch.nn.functional.mse_loss(
                            state.prediction,
                            target.detach(),
                        ).item()
                    )
                    for state, target in zip(corrupted_model.layer_states, next_targets)
                ]
                del next_frame

            phase, phase_frame_index = phase_for_frame(frame_offset)
            for layer_mse, clean_state, corrupted_state in zip(
                next_frame_prediction_mses,
                clean_model.layer_states,
                corrupted_model.layer_states,
            ):
                rows.append(
                    {
                        "condition": condition,
                        "corruption": corruption_name,
                        "drive": drive,
                        "frame_offset": frame_offset,
                        "phase": phase,
                        "phase_frame_index": phase_frame_index,
                        "raw_frame_index": raw_frame_index,
                        "frame_name": clean_dataset.frame_paths[raw_frame_index].name,
                        "layer": clean_state.layer_index,
                        "instantaneous_prediction_error_rms": tensor_rms(
                            corrupted_state.instant_error
                        ),
                        "dynamic_error_rms": tensor_rms(
                            corrupted_state.dynamic_error
                        ),
                        "next_frame_prediction_mse": layer_mse,
                        "representation_normalized_l2": (
                            normalized_representation_distance(
                                corrupted_state.representation,
                                clean_state.representation,
                            )
                        ),
                    }
                )
            del clean_frame, corrupted_frame

    del clean_model, corrupted_model
    torch.cuda.empty_cache()
    return rows


def mean_distance(rows):
    return sum(float(row["representation_normalized_l2"]) for row in rows) / len(rows)


def mean_next_mse(rows):
    values = [
        float(row["next_frame_prediction_mse"])
        for row in rows
        if row["next_frame_prediction_mse"] is not None
    ]
    return sum(values) / len(values)


def condition_metrics(rows):
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

    def grouped(field):
        values = {}
        for value in sorted({row[field] for row in rows}, key=str):
            subset = [row for row in rows if row[field] == value]
            values[str(value)] = {
                "mean_next_frame_prediction_mse": mean_next_mse(subset),
                "disturbance_mean_representation_normalized_l2": mean_distance(
                    [row for row in subset if row["phase"] == "disturbance"]
                ),
                "recovery_first_10_mean_representation_normalized_l2": mean_distance(
                    [
                        row for row in subset
                        if row["phase"] == "recovery"
                        and row["phase_frame_index"] < 10
                    ]
                ),
                "recovery_last_10_mean_representation_normalized_l2": mean_distance(
                    [
                        row for row in subset
                        if row["phase"] == "recovery"
                        and row["phase_frame_index"] >= RECOVERY_FRAMES - 10
                    ]
                ),
            }
        return values

    return {
        "mean_next_frame_prediction_mse": mean_next_mse(rows),
        "disturbance_mean_next_frame_prediction_mse": mean_next_mse(disturbance),
        "disturbance_mean_representation_normalized_l2": mean_distance(disturbance),
        "recovery_first_10_mean_representation_normalized_l2": mean_distance(
            recovery_first
        ),
        "recovery_last_10_mean_representation_normalized_l2": mean_distance(
            recovery_last
        ),
        "per_drive": grouped("drive"),
        "per_layer": grouped("layer"),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_readme(path, summary):
    lines = [
        "# Error-memory Recurrent Validation",
        "",
        "VGG, original Predify, and feedback decoders are frozen. Only the "
        "observation/error recurrent transition and signed-error encoder train. "
        "Validation uses drives 0011/0039 and the unchanged 40 clean / 80 "
        "corruption / 40 recovery protocol. Frozen Test drives 0051/0056 were "
        "not read.",
    ]
    for corruption, result in summary["corruptions"].items():
        conditions = result["conditions"]
        lines.extend([
            "",
            f"## {corruption}",
            "",
            "| Condition | Next-frame MSE | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for condition in CONDITIONS:
            metric = conditions[condition]
            lines.append(
                f"| {condition} | {metric['mean_next_frame_prediction_mse']:.9f} | "
                f"{metric['disturbance_mean_representation_normalized_l2']:.9f} | "
                f"{metric['recovery_first_10_mean_representation_normalized_l2']:.9f} | "
                f"{metric['recovery_last_10_mean_representation_normalized_l2']:.9f} |"
            )
        lines.extend([
            "",
            f"Conclusion: {result['comparison']['conclusion']}.",
            "",
            "| Drive | temporal_only | instant_error | error_memory |",
            "| --- | ---: | ---: | ---: |",
        ])
        for drive in summary["protocol"]["val_drives"]:
            lines.append(
                f"| {drive.split('_drive_')[-1].split('_sync')[0]} | "
                f"{conditions['temporal_only']['per_drive'][drive]['disturbance_mean_representation_normalized_l2']:.9f} | "
                f"{conditions['instant_error']['per_drive'][drive]['disturbance_mean_representation_normalized_l2']:.9f} | "
                f"{conditions['error_memory']['per_drive'][drive]['disturbance_mean_representation_normalized_l2']:.9f} |"
            )
    lines.extend(["", f"Overall conclusion: {summary['overall_conclusion']}."])
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="ascii")


def compare_conditions(metrics):
    temporal = metrics["temporal_only"]["disturbance_mean_representation_normalized_l2"]
    instant = metrics["instant_error"]["disturbance_mean_representation_normalized_l2"]
    memory = metrics["error_memory"]["disturbance_mean_representation_normalized_l2"]
    per_drive = {}
    for drive in metrics["temporal_only"]["per_drive"]:
        drive_temporal = metrics["temporal_only"]["per_drive"][drive][
            "disturbance_mean_representation_normalized_l2"
        ]
        drive_instant = metrics["instant_error"]["per_drive"][drive][
            "disturbance_mean_representation_normalized_l2"
        ]
        drive_memory = metrics["error_memory"]["per_drive"][drive][
            "disturbance_mean_representation_normalized_l2"
        ]
        per_drive[drive] = {
            "temporal_only": drive_temporal,
            "instant_error": drive_instant,
            "error_memory": drive_memory,
            "memory_lt_instant_lt_temporal": (
                drive_memory < drive_instant < drive_temporal
            ),
            "memory_approximately_instant_both_lt_temporal": (
                abs(drive_memory - drive_instant) <= 0.01 * max(drive_instant, 1e-12)
                and drive_memory < drive_temporal
                and drive_instant < drive_temporal
            ),
        }
    if memory < instant < temporal and all(
        item["memory_lt_instant_lt_temporal"] for item in per_drive.values()
    ):
        conclusion = "supports_accumulated_prediction_error_memory"
    elif (
        abs(memory - instant) <= 0.01 * max(instant, 1e-12)
        and memory < temporal
        and instant < temporal
    ):
        conclusion = "error_useful_but_accumulated_memory_has_no_extra_value"
    elif memory >= temporal and instant >= temporal:
        conclusion = "benefit_mainly_from_temporal_recurrence"
    else:
        conclusion = "mixed_drive_or_metric_direction"
    return {
        "metric": "disturbance_mean_representation_normalized_l2",
        "temporal_only": temporal,
        "instant_error": instant,
        "error_memory": memory,
        "memory_vs_temporal_improvement_percent": 100.0 * (temporal - memory) / temporal,
        "memory_vs_instant_improvement_percent": 100.0 * (instant - memory) / instant,
        "per_drive": per_drive,
        "conclusion": conclusion,
    }


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("Formal recurrent-error validation requires GPU 0.")
    revision = os.environ["PREDIFY_GIT_REVISION"]
    output_dir = Path(os.environ["PREDIFY_RECURRENT_ERROR_EVAL_OUTPUT_DIR"])
    training_dir = Path(os.environ["PREDIFY_RECURRENT_ERROR_TRAIN_OUTPUT_DIR"])
    checkpoints = {
        condition: checkpoint_for(training_dir, condition)
        for condition in ("temporal_only", "instant_error", "error_memory")
    }
    weights_path = os.environ["PREDIFY_PCODER_WEIGHTS"]
    root = os.environ["PREDIFY_KITTI_ROOT"]
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
    seed = int(os.environ.get("PREDIFY_SEED", "0"))
    drives = tuple(
        drive.strip()
        for drive in os.environ["PREDIFY_RECURRENT_ERROR_VAL_DRIVES"].split(",")
        if drive.strip()
    )
    if drives != ALLOWED_DRIVES:
        raise ValueError(f"Validation drives must be exactly {ALLOWED_DRIVES}.")
    if any(forbidden in drive for forbidden in FORBIDDEN_TEST_DRIVE_IDS for drive in drives):
        raise RuntimeError("Frozen Test drive entered validation.")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    models = {
        "current_stateful": build_model(weights_path, "predify"),
        "temporal_only": build_model(
            weights_path,
            "convgru_error",
            checkpoints["temporal_only"],
            recurrent_input="temporal_only",
        ),
        "instant_error": build_model(
            weights_path,
            "convgru_error",
            checkpoints["instant_error"],
            recurrent_input="instant",
        ),
        "error_memory": build_model(
            weights_path,
            "convgru_error",
            checkpoints["error_memory"],
            recurrent_input="memory",
        ),
    }
    rows = []
    corruption_results = {}
    for corruption_name, config in CORRUPTIONS.items():
        datasets = {
            drive: build_corruption_drive_datasets(
                root,
                drive,
                camera,
                fixed_dt_s,
                tolerance,
                config,
            )
            for drive in drives
        }
        for condition in CONDITIONS:
            print(f"Evaluating {condition} on {corruption_name}...", flush=True)
            rows.extend(
                evaluate_condition(
                    models[condition],
                    condition,
                    corruption_name,
                    datasets,
                    seed,
                )
            )
        metrics = {
            condition: condition_metrics(
                [
                    row for row in rows
                    if row["condition"] == condition
                    and row["corruption"] == corruption_name
                ]
            )
            for condition in CONDITIONS
        }
        corruption_results[corruption_name] = {
            "conditions": metrics,
            "comparison": compare_conditions(metrics),
            "config": config.to_dict(),
        }
    if all(
        result["comparison"]["conclusion"]
        == "supports_accumulated_prediction_error_memory"
        for result in corruption_results.values()
    ):
        overall_conclusion = "supports_accumulated_prediction_error_memory"
    elif all(
        result["comparison"]["conclusion"]
        == "error_useful_but_accumulated_memory_has_no_extra_value"
        for result in corruption_results.values()
    ):
        overall_conclusion = "error_useful_but_accumulated_memory_has_no_extra_value"
    elif all(
        result["comparison"]["conclusion"]
        == "benefit_mainly_from_temporal_recurrence"
        for result in corruption_results.values()
    ):
        overall_conclusion = "benefit_mainly_from_temporal_recurrence"
    else:
        overall_conclusion = "mixed_corruption_results"
    with (training_dir / "training_summary.json").open() as handle:
        training_summary = json.load(handle)
    summary = {
        "experiment": "real_frame_error_memory_validation",
        "git_revision": revision,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(DEVICE),
        "corruptions": corruption_results,
        "overall_conclusion": overall_conclusion,
        "protocol": {
            "train_drives": training_summary["train_drives"],
            "val_drives": drives,
            "frozen_test_drives_read": False,
            "baseline_frames": BASELINE_FRAMES,
            "disturbance_frames": DISTURBANCE_FRAMES,
            "recovery_frames": RECOVERY_FRAMES,
            "dynamic_error": "epsilon_t=0.207*e_t+0.793*epsilon_(t-1)",
            "instant_error": "e_t=F_t-Fhat_t",
            "core_conditions": CORE_CONDITIONS,
            "state_transition": "h_t=T(h_(t-1),F_t,E(error_input),feedback)",
            "learned_top_down_feedback": True,
            "matched_transition_capacity": True,
            "dedicated_error_encoder": True,
            "temporal_predictor_trained": False,
            "cross_frame_state_detached": True,
            "future_predictor": False,
            "online_learning_during_validation": False,
        },
        "training": training_summary,
        "checkpoints": {
            condition: str(path) for condition, path in checkpoints.items()
        } | {
            f"{condition}_sha256": sha256_file(path)
            for condition, path in checkpoints.items()
        },
        "per_frame_row_count": len(rows),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "per_frame.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with (output_dir / "training_summary.json").open("w") as handle:
        json.dump(training_summary, handle, indent=2, sort_keys=True)
    write_readme(output_dir / "README.md", summary)
    print(json.dumps(summary["overall_conclusion"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
