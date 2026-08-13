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
    BLUR_KERNEL_SIZE,
    BLUR_SIGMA,
    DISTURBANCE_FRAMES,
    FORBIDDEN_TEST_DRIVE_IDS,
    RECOVERY_FRAMES,
    build_drive_datasets,
    normalized_representation_distance,
    phase_for_frame,
    tensor_rms,
)
from predify2021.model_factory import get_model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONDITIONS = (
    "current_stateful",
    "observation_driven_recurrent",
    "error_driven_recurrent",
    "error_driven_recurrent_zeroed",
)
CORE_CONDITIONS = (
    "current_stateful",
    "observation_driven_recurrent",
    "error_driven_recurrent",
)


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
    model.requires_grad_(False)
    return model


def checkpoint_for(training_dir, condition):
    return training_dir / f"best_{condition}.pt"


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


def evaluate_condition(base_model, condition, datasets, seed):
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
    baseline = summary["conditions"]["current_stateful"]
    observation = summary["conditions"]["observation_driven_recurrent"]
    learned = summary["conditions"]["error_driven_recurrent"]
    zeroed = summary["conditions"]["error_driven_recurrent_zeroed"]
    lines = [
        "# Prediction-error-driven Recurrent Validation",
        "",
        "Frozen backbone, feedback decoders, and existing Predify parameters. "
        "Only the new recurrent transition was trained on drives 0005/0013/0014/0036. "
        "Validation uses drives 0011/0039 and the unchanged 40 clean / 80 blur / "
        "40 recovery protocol. Frozen Test drives 0051/0056 were not read.",
        "",
        "| Condition | Next-frame MSE | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Current stateful | {baseline['mean_next_frame_prediction_mse']:.9f} | {baseline['disturbance_mean_representation_normalized_l2']:.9f} | {baseline['recovery_first_10_mean_representation_normalized_l2']:.9f} | {baseline['recovery_last_10_mean_representation_normalized_l2']:.9f} |",
        f"| Observation-driven recurrent | {observation['mean_next_frame_prediction_mse']:.9f} | {observation['disturbance_mean_representation_normalized_l2']:.9f} | {observation['recovery_first_10_mean_representation_normalized_l2']:.9f} | {observation['recovery_last_10_mean_representation_normalized_l2']:.9f} |",
        f"| Error-driven recurrent | {learned['mean_next_frame_prediction_mse']:.9f} | {learned['disturbance_mean_representation_normalized_l2']:.9f} | {learned['recovery_first_10_mean_representation_normalized_l2']:.9f} | {learned['recovery_last_10_mean_representation_normalized_l2']:.9f} |",
        "",
        f"Error-driven vs current disturbance improvement: {summary['comparison']['error_vs_current_improvement_percent']:.6f}%.",
        f"Error-driven vs observation disturbance improvement: {summary['comparison']['error_vs_observation_improvement_percent']:.6f}%.",
        f"Conclusion: {summary['comparison']['conclusion']}.",
        "",
        "Sanity check: error-zeroed disturbance normalized L2 = "
        f"{zeroed['disturbance_mean_representation_normalized_l2']:.9f}; it is "
        "not used as the core performance control.",
        "",
        "## Per Drive",
        "",
        "| Drive | Current | Observation | Error |",
        "| --- | ---: | ---: | ---: |",
    ]
    for drive in summary["protocol"]["val_drives"]:
        lines.append(
            f"| {drive.split('_drive_')[-1].split('_sync')[0]} | "
            f"{baseline['per_drive'][drive]['disturbance_mean_representation_normalized_l2']:.9f} | "
            f"{observation['per_drive'][drive]['disturbance_mean_representation_normalized_l2']:.9f} | "
            f"{learned['per_drive'][drive]['disturbance_mean_representation_normalized_l2']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Per Layer",
            "",
            "| Layer | Current | Observation | Error |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for layer in range(1, 6):
        key = str(layer)
        lines.append(
            f"| {layer} | "
            f"{baseline['per_layer'][key]['disturbance_mean_representation_normalized_l2']:.9f} | "
            f"{observation['per_layer'][key]['disturbance_mean_representation_normalized_l2']:.9f} | "
            f"{learned['per_layer'][key]['disturbance_mean_representation_normalized_l2']:.9f} |"
        )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="ascii")


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("Formal recurrent-error validation requires GPU 0.")
    revision = os.environ["PREDIFY_GIT_REVISION"]
    output_dir = Path(os.environ["PREDIFY_RECURRENT_ERROR_EVAL_OUTPUT_DIR"])
    training_dir = Path(os.environ["PREDIFY_RECURRENT_ERROR_TRAIN_OUTPUT_DIR"])
    error_checkpoint = checkpoint_for(training_dir, "error_driven_recurrent")
    observation_checkpoint = checkpoint_for(
        training_dir, "observation_driven_recurrent"
    )
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

    datasets = {
        drive: build_drive_datasets(root, drive, camera, fixed_dt_s, tolerance)
        for drive in drives
    }
    models = {
        "current_stateful": build_model(weights_path, "predify"),
        "observation_driven_recurrent": build_model(
            weights_path,
            "convgru_error",
            observation_checkpoint,
            recurrent_input="observation",
        ),
        "error_driven_recurrent": build_model(
            weights_path, "convgru_error", error_checkpoint
        ),
        "error_driven_recurrent_zeroed": build_model(
            weights_path,
            "convgru_error",
            error_checkpoint,
            recurrent_input="zeroed",
        ),
    }
    rows = []
    for condition in CONDITIONS:
        print(f"Evaluating {condition} on {DEVICE}...", flush=True)
        rows.extend(
            evaluate_condition(models[condition], condition, datasets, seed)
        )
        del models[condition]

    metrics = {
        condition: condition_metrics(
            [row for row in rows if row["condition"] == condition]
        )
        for condition in CONDITIONS
    }
    baseline_value = metrics["current_stateful"][
        "disturbance_mean_representation_normalized_l2"
    ]
    observation_value = metrics["observation_driven_recurrent"][
        "disturbance_mean_representation_normalized_l2"
    ]
    learned_value = metrics["error_driven_recurrent"][
        "disturbance_mean_representation_normalized_l2"
    ]
    zeroed_value = metrics["error_driven_recurrent_zeroed"][
        "disturbance_mean_representation_normalized_l2"
    ]
    error_vs_current = 100.0 * (
        baseline_value - learned_value
    ) / baseline_value
    error_vs_observation = 100.0 * (
        observation_value - learned_value
    ) / observation_value
    if error_vs_current > 0.0 and error_vs_observation > 5.0:
        conclusion = "prediction_error_has_independent_value"
    elif error_vs_current > 0.0 and abs(error_vs_observation) <= 5.0:
        conclusion = "benefit_mainly_from_recurrent_temporal_modeling"
    elif observation_value < learned_value:
        conclusion = "strict_error_driven_mechanism_not_supported"
    elif error_vs_current > 0.0:
        conclusion = "weak_error_advantage_over_observation"
    else:
        conclusion = "learned_recurrent_transition_does_not_beat_current_stateful"
    with (training_dir / "training_summary.json").open() as handle:
        training_summary = json.load(handle)
    summary = {
        "experiment": "real_frame_matched_recurrent_validation",
        "git_revision": revision,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(DEVICE),
        "conditions": metrics,
        "comparison": {
            "metric": "disturbance_mean_representation_normalized_l2",
            "current_stateful": baseline_value,
            "observation_driven_recurrent": observation_value,
            "error_driven_recurrent": learned_value,
            "error_driven_recurrent_zeroed_sanity": zeroed_value,
            "error_vs_current_improvement_percent": error_vs_current,
            "error_vs_observation_improvement_percent": error_vs_observation,
            "error_vs_observation_absolute_change": observation_value - learned_value,
            "conclusion": conclusion,
        },
        "protocol": {
            "train_drives": training_summary["train_drives"],
            "val_drives": drives,
            "frozen_test_drives_read": False,
            "baseline_frames": BASELINE_FRAMES,
            "disturbance_frames": DISTURBANCE_FRAMES,
            "recovery_frames": RECOVERY_FRAMES,
            "blur_kernel_size": BLUR_KERNEL_SIZE,
            "blur_sigma": BLUR_SIGMA,
            "dynamic_error": "epsilon_t=0.207*e_t+0.793*epsilon_(t-1)",
            "instant_error": "e_t=F_t-Fhat_t",
            "core_conditions": CORE_CONDITIONS,
            "sanity_conditions": ("error_driven_recurrent_zeroed",),
            "state_transition_error": "h_t=T(h_(t-1),epsilon_t,feedback)",
            "state_transition_observation": "h_t=T(h_(t-1),F_t,feedback)",
            "current_feedforward_transition_input": False,
            "learned_top_down_feedback": True,
            "matched_transition_capacity": True,
            "error_zeroed_sanity_check": "only recurrent transition error drive is zero",
            "cross_frame_state_detached": True,
            "future_predictor": False,
            "online_learning_during_validation": False,
        },
        "training": training_summary,
        "checkpoints": {
            "error_driven_recurrent": str(error_checkpoint),
            "error_driven_recurrent_sha256": sha256_file(error_checkpoint),
            "observation_driven_recurrent": str(observation_checkpoint),
            "observation_driven_recurrent_sha256": sha256_file(
                observation_checkpoint
            ),
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
    print(json.dumps(summary["comparison"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
