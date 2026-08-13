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
CONDITIONS = ("current_stateful", "learned_recurrent_error")


def build_model(weights_path, transition_mode, checkpoint=None):
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
    ).eval()
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.recurrent_transition_modules.load_state_dict(
            payload["recurrent_transition_state_dict"]
        )
    model.requires_grad_(False)
    return model


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

            phase, phase_frame_index = phase_for_frame(frame_offset)
            for clean_state, corrupted_state in zip(
                clean_model.layer_states, corrupted_model.layer_states
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
    learned = summary["conditions"]["learned_recurrent_error"]
    lines = [
        "# Learned Recurrent-error Validation",
        "",
        "Frozen backbone, feedback decoders, and existing Predify parameters. "
        "Only the new recurrent transition was trained on drives 0005/0013/0014/0036. "
        "Validation uses drives 0011/0039 and the unchanged 40 clean / 80 blur / "
        "40 recovery protocol. Frozen Test drives 0051/0056 were not read.",
        "",
        "| Condition | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |",
        "| --- | ---: | ---: | ---: |",
        f"| Current stateful | {baseline['disturbance_mean_representation_normalized_l2']:.9f} | {baseline['recovery_first_10_mean_representation_normalized_l2']:.9f} | {baseline['recovery_last_10_mean_representation_normalized_l2']:.9f} |",
        f"| Learned recurrent-error | {learned['disturbance_mean_representation_normalized_l2']:.9f} | {learned['recovery_first_10_mean_representation_normalized_l2']:.9f} | {learned['recovery_last_10_mean_representation_normalized_l2']:.9f} |",
        "",
        f"Disturbance relative improvement: {summary['comparison']['disturbance_relative_improvement_percent']:.6f}%.",
        f"Direction: {summary['comparison']['direction']}.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="ascii")


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("Formal recurrent-error validation requires GPU 0.")
    revision = os.environ["PREDIFY_GIT_REVISION"]
    output_dir = Path(os.environ["PREDIFY_RECURRENT_ERROR_EVAL_OUTPUT_DIR"])
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
        "learned_recurrent_error": build_model(
            weights_path, "convgru_error", checkpoint
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
    learned_value = metrics["learned_recurrent_error"][
        "disturbance_mean_representation_normalized_l2"
    ]
    relative_improvement = 100.0 * (baseline_value - learned_value) / baseline_value
    with (training_dir / "training_summary.json").open() as handle:
        training_summary = json.load(handle)
    summary = {
        "experiment": "real_frame_learned_recurrent_error_validation",
        "git_revision": revision,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(DEVICE),
        "conditions": metrics,
        "comparison": {
            "metric": "disturbance_mean_representation_normalized_l2",
            "current_stateful": baseline_value,
            "learned_recurrent_error": learned_value,
            "disturbance_relative_improvement_percent": relative_improvement,
            "direction": "improved" if relative_improvement > 0.0 else "worsened",
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
            "cross_frame_state_detached": True,
            "future_predictor": False,
            "online_learning_during_validation": False,
        },
        "training": training_summary,
        "checkpoint_sha256": sha256_file(checkpoint),
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
