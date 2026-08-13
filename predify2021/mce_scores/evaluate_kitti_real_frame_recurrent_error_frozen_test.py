import csv
import json
import os
import random
from pathlib import Path

import torch

from predify2021.mce_scores.evaluate_kitti_real_frame_pc_phase1 import (
    BASELINE_FRAMES,
    BLUR_KERNEL_SIZE,
    BLUR_SIGMA,
    DISTURBANCE_FRAMES,
    RECOVERY_FRAMES,
    build_drive_datasets,
)
from predify2021.mce_scores.evaluate_kitti_real_frame_recurrent_error import (
    CONDITIONS,
    DEVICE,
    build_model,
    condition_metrics,
    evaluate_condition,
    sha256_file,
)


FROZEN_TEST_DRIVES = (
    "2011_09_26/2011_09_26_drive_0051_sync",
    "2011_09_26/2011_09_26_drive_0056_sync",
)
EXPECTED_MODEL_REVISION = "c9fec526680bb274aca2b96397db4e20741f248d"
EXPECTED_CHECKPOINT_SHA256 = (
    "26c5333da95a6b754af6036f9d16a5dd394673400e6f9456c0ebc0b27e714cf4"
)


def claim_frozen_test(checkpoint, evaluator_revision):
    checkpoint = Path(checkpoint)
    receipt_path = checkpoint.with_name(
        f"{checkpoint.stem}_real_frame_frozen_test_receipt.json"
    )
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Frozen checkpoint SHA-256 does not match c9fec52 epoch 1.")
    payload = {
        "status": "claimed_before_test_read",
        "evaluator_revision": evaluator_revision,
        "model_revision": EXPECTED_MODEL_REVISION,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "frozen_test_drives": FROZEN_TEST_DRIVES,
        "policy": "single_frozen_test_access_without_retraining_or_tuning",
    }
    try:
        with receipt_path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    except FileExistsError as error:
        raise RuntimeError(
            f"Frozen Test was already claimed for this checkpoint: {receipt_path}"
        ) from error
    return receipt_path, payload


def validate_checkpoint(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["git_revision"] != EXPECTED_MODEL_REVISION:
        raise ValueError("Checkpoint is not the frozen c9fec52 model revision.")
    if payload["epoch"] != 1:
        raise ValueError("Checkpoint is not the frozen epoch-1 selection.")
    if payload["transition_mode"] != "convgru_error":
        raise ValueError("Checkpoint transition mode changed.")
    if payload.get("recurrent_error_input") != "dynamic":
        raise ValueError("Checkpoint recurrent error input changed.")
    return payload


def comparison_summary(metrics):
    current = metrics["current_stateful"]
    learned = metrics["learned_recurrent_error"]
    zeroed = metrics["learned_recurrent_error_zeroed"]

    def improvement(reference, candidate):
        return 100.0 * (reference - candidate) / reference

    per_drive = {}
    for drive in FROZEN_TEST_DRIVES:
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
            "learned_better_than_current": learned_value < current_value,
            "learned_better_than_zeroed": learned_value < zeroed_value,
        }

    learned_beats_current = all(
        item["learned_better_than_current"] for item in per_drive.values()
    )
    learned_beats_zeroed = all(
        item["learned_better_than_zeroed"] for item in per_drive.values()
    )
    if learned_beats_current and learned_beats_zeroed:
        decision = "PASS_MAIN_MECHANISM"
    elif learned_beats_current:
        decision = "PASS_RECURRENT_ONLY_DYNAMIC_ERROR_NOT_SUPPORTED"
    else:
        decision = "FAIL_UNSTABLE_VS_CURRENT"

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
        "both_drives_learned_better_than_current": learned_beats_current,
        "both_drives_learned_better_than_zeroed": learned_beats_zeroed,
        "decision": decision,
    }


def write_readme(path, summary):
    conditions = summary["conditions"]
    comparison = summary["comparison"]
    lines = [
        "# Learned Recurrent-error Frozen Test",
        "",
        "Frozen c9fec52 structure and epoch-1 checkpoint on Test drives 0051/0056. "
        "The protocol is unchanged: 40 clean, 80 persistent Gaussian-blur, and "
        "40 clean recovery frames. No training, tuning, or checkpoint update occurred.",
        "",
        "| Condition | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        item = conditions[condition]
        lines.append(
            f"| {condition} | "
            f"{item['disturbance_mean_representation_normalized_l2']:.9f} | "
            f"{item['recovery_first_10_mean_representation_normalized_l2']:.9f} | "
            f"{item['recovery_last_10_mean_representation_normalized_l2']:.9f} |"
        )
    lines.extend(
        [
            "",
            f"Learned vs current: {comparison['learned_vs_current_improvement_percent']:.6f}%.",
            f"Learned vs zeroed: {comparison['learned_vs_zeroed_improvement_percent']:.6f}%.",
            f"Decision: {comparison['decision']}.",
            "",
            "## Per Drive",
            "",
            "| Drive | Current | Learned | Zeroed | Learned vs current | Learned vs zeroed |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for drive in FROZEN_TEST_DRIVES:
        item = comparison["per_drive"][drive]
        drive_id = drive.split("_drive_")[-1].split("_sync")[0]
        lines.append(
            f"| {drive_id} | {item['current_stateful']:.9f} | "
            f"{item['learned_recurrent_error']:.9f} | "
            f"{item['learned_recurrent_error_zeroed']:.9f} | "
            f"{item['learned_vs_current_improvement_percent']:.6f}% | "
            f"{item['learned_vs_zeroed_improvement_percent']:.6f}% |"
        )
    lines.extend(
        [
            "",
            "## Per Layer",
            "",
            "| Layer | Current | Learned | Zeroed |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for layer in range(1, 6):
        key = str(layer)
        lines.append(
            f"| {layer} | "
            f"{conditions['current_stateful']['per_layer'][key]['disturbance_mean_representation_normalized_l2']:.9f} | "
            f"{conditions['learned_recurrent_error']['per_layer'][key]['disturbance_mean_representation_normalized_l2']:.9f} | "
            f"{conditions['learned_recurrent_error_zeroed']['per_layer'][key]['disturbance_mean_representation_normalized_l2']:.9f} |"
        )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="ascii")


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("Formal Frozen Test requires GPU 0.")
    evaluator_revision = os.environ["PREDIFY_GIT_REVISION"]
    output_dir = Path(os.environ["PREDIFY_RECURRENT_ERROR_TEST_OUTPUT_DIR"])
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
        for drive in os.environ["PREDIFY_RECURRENT_ERROR_TEST_DRIVES"].split(",")
        if drive.strip()
    )
    if drives != FROZEN_TEST_DRIVES:
        raise ValueError(f"Frozen Test drives must be exactly {FROZEN_TEST_DRIVES}.")

    checkpoint_payload = validate_checkpoint(checkpoint)
    receipt_path, receipt = claim_frozen_test(checkpoint, evaluator_revision)
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
        "learned_recurrent_error_zeroed": build_model(
            weights_path,
            "convgru_error",
            checkpoint,
            recurrent_error_input="zeroed",
        ),
    }
    rows = []
    for condition in CONDITIONS:
        print(f"Evaluating Frozen Test {condition}...", flush=True)
        rows.extend(evaluate_condition(models[condition], condition, datasets, seed))
        del models[condition]

    metrics = {
        condition: condition_metrics(
            [row for row in rows if row["condition"] == condition]
        )
        for condition in CONDITIONS
    }
    comparison = comparison_summary(metrics)
    summary = {
        "experiment": "real_frame_learned_recurrent_error_frozen_test",
        "evaluator_revision": evaluator_revision,
        "model_revision": EXPECTED_MODEL_REVISION,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(DEVICE),
        "conditions": metrics,
        "comparison": comparison,
        "protocol": {
            "frozen_test_drives": drives,
            "baseline_frames": BASELINE_FRAMES,
            "disturbance_frames": DISTURBANCE_FRAMES,
            "recovery_frames": RECOVERY_FRAMES,
            "blur_kernel_size": BLUR_KERNEL_SIZE,
            "blur_sigma": BLUR_SIGMA,
            "seed": seed,
            "training": False,
            "tuning": False,
            "checkpoint_update": False,
            "learned_and_zeroed_share_checkpoint": True,
            "zeroed_control": "only recurrent transition error drive is zero",
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": receipt["checkpoint_sha256"],
            "epoch": checkpoint_payload["epoch"],
            "val_prediction_mse": checkpoint_payload["val_prediction_mse"],
        },
        "selected_raw_frames": {
            drive: {
                "start": raw_frames[0],
                "stop_inclusive": raw_frames[-1],
                "count": len(raw_frames),
            }
            for drive, (_, _, raw_frames) in datasets.items()
        },
        "per_frame_row_count": len(rows),
        "frozen_test_receipt": str(receipt_path),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "per_frame.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    write_readme(output_dir / "README.md", summary)

    completed_receipt = {
        **receipt,
        "status": "completed",
        "output_summary_path": str(summary_path),
        "decision": comparison["decision"],
    }
    with receipt_path.open("w") as handle:
        json.dump(completed_receipt, handle, indent=2, sort_keys=True)
    with (output_dir / "frozen_test_receipt.json").open("w") as handle:
        json.dump(completed_receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(comparison, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
