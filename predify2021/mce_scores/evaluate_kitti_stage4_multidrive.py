import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Subset

from predify2021.mce_scores.evaluate_kitti_aligned_temporal_difference import (
    DEVICE,
    build_model,
    evaluate_gate,
    evaluate_split,
    summarize_rows,
    validate_checkpoint,
)
from predify2021.mce_scores.kitti_pairs import KITTIMultiHorizonFrameDataset


TRAIN_DRIVES = (
    "2011_09_26/2011_09_26_drive_0005_sync",
    "2011_09_26/2011_09_26_drive_0013_sync",
    "2011_09_26/2011_09_26_drive_0014_sync",
    "2011_09_26/2011_09_26_drive_0036_sync",
)
VAL_DRIVES = (
    "2011_09_26/2011_09_26_drive_0011_sync",
    "2011_09_26/2011_09_26_drive_0039_sync",
)
FROZEN_TEST_DRIVES = (
    "2011_09_26/2011_09_26_drive_0051_sync",
    "2011_09_26/2011_09_26_drive_0056_sync",
)
EXPECTED_STAGE = 4


def validate_multidrive_checkpoint(checkpoint, expected_revision):
    validation = validate_checkpoint(
        checkpoint,
        expected_revision=expected_revision,
        expected_stage=EXPECTED_STAGE,
    )
    config = checkpoint["config"]
    required = {
        "formal_split": True,
        "same_drive_split": False,
        "same_drive_three_way_split": False,
        "stream_mode": True,
        "reset_each_frame": False,
        "shuffle_train_pairs": False,
        "shuffle_val_pairs": False,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Expected {key}={expected!r}, got {config.get(key)!r}.")
    if tuple(config.get("train_drives", ())) != TRAIN_DRIVES:
        raise ValueError("Checkpoint does not use the frozen four-drive Train split.")
    if tuple(config.get("val_drives", ())) != VAL_DRIVES:
        raise ValueError("Checkpoint does not use the frozen two-drive Val split.")
    if config.get("test_drives") is not None:
        raise ValueError("Training checkpoint must not contain or consume Test drives.")
    if set(TRAIN_DRIVES) & set(VAL_DRIVES):
        raise RuntimeError("Internal Train/Val split contract overlaps.")
    if (set(TRAIN_DRIVES) | set(VAL_DRIVES)) & set(FROZEN_TEST_DRIVES):
        raise RuntimeError("Internal frozen Test split contract overlaps Train or Val.")
    return validation


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def claim_frozen_test_access(checkpoint_path, expected_revision):
    checkpoint_path = Path(checkpoint_path)
    receipt_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_frozen_test_receipt.json"
    )
    payload = {
        "status": "claimed_before_test_read",
        "git_revision": expected_revision,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "frozen_test_drives": FROZEN_TEST_DRIVES,
        "policy": "single_access_after_best_validation_checkpoint_selection",
    }
    try:
        with receipt_path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    except FileExistsError as error:
        raise RuntimeError(
            "Frozen Test access was already claimed for this checkpoint: "
            f"{receipt_path}"
        ) from error
    return receipt_path, payload


def finalize_frozen_test_receipt(receipt_path, payload, output_summary_path):
    completed = {
        **payload,
        "status": "completed",
        "output_summary_path": str(output_summary_path),
    }
    with Path(receipt_path).open("w") as handle:
        json.dump(completed, handle, indent=2, sort_keys=True)
    return completed


def build_drive_dataset(root, drive, camera, fixed_dt_s, tolerance):
    return KITTIMultiHorizonFrameDataset(
        root,
        drive,
        camera=camera,
        horizons=(1,),
        fixed_dt_s=fixed_dt_s,
        dt_tolerance_s=tolerance,
    )


def evaluate_drive(model, dataset, split, drive, fixed_dt_s):
    rows = []
    drive_sample_index = 0
    for segment_index, sample_indices in enumerate(dataset.valid_sample_segments):
        segment_dataset = Subset(dataset, list(sample_indices))
        segment_rows = evaluate_split(
            model,
            segment_dataset,
            split=split,
            drive=drive,
            fixed_dt_s=fixed_dt_s,
        )
        for segment_sample_index, row in enumerate(segment_rows):
            row["sample_index"] = drive_sample_index
            row["segment_index"] = segment_index
            row["segment_sample_index"] = segment_sample_index
            rows.append(row)
            drive_sample_index += 1
    return rows


def evaluate_drive_group(model, root, camera, fixed_dt_s, tolerance, split, drives):
    rows = []
    per_drive = {}
    for drive in drives:
        dataset = build_drive_dataset(
            root,
            drive,
            camera,
            fixed_dt_s,
            tolerance,
        )
        drive_rows = evaluate_drive(
            model,
            dataset,
            split=split,
            drive=drive,
            fixed_dt_s=fixed_dt_s,
        )
        rows.extend(drive_rows)
        drive_summary = summarize_rows(drive_rows)
        per_drive[drive] = {
            "time_filter_stats": dataset.time_filter_stats,
            "valid_segment_count": len(dataset.valid_sample_segments),
            "valid_segment_lengths": tuple(
                len(segment) for segment in dataset.valid_sample_segments
            ),
            "summary": drive_summary,
            "same_stage_copy_current_gate": evaluate_gate(
                drive_summary["metrics"]
            ),
        }
    return rows, per_drive


def main():
    output_dir = Path(os.environ["PREDIFY_MULTIDRIVE_EVAL_OUTPUT_DIR"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_revision = os.environ["PREDIFY_GIT_REVISION"]
    checkpoint_path = Path(os.environ["PREDIFY_MULTIDRIVE_CHECKPOINT"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validation = validate_multidrive_checkpoint(checkpoint, expected_revision)
    root = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
    model = build_model(checkpoint, validation)

    rows = []
    summaries = {}
    per_drive = {}
    for split, drives in (("train", TRAIN_DRIVES), ("val", VAL_DRIVES)):
        split_rows, split_per_drive = evaluate_drive_group(
            model,
            root,
            camera,
            fixed_dt_s,
            tolerance,
            split,
            drives,
        )
        rows.extend(split_rows)
        summaries[split] = summarize_rows(split_rows)
        per_drive[split] = split_per_drive

    receipt_path, receipt_payload = claim_frozen_test_access(
        checkpoint_path,
        expected_revision,
    )
    test_rows, test_per_drive = evaluate_drive_group(
        model,
        root,
        camera,
        fixed_dt_s,
        tolerance,
        "test",
        FROZEN_TEST_DRIVES,
    )
    rows.extend(test_rows)
    summaries["test"] = summarize_rows(test_rows)
    per_drive["test"] = test_per_drive

    gates = {
        split: evaluate_gate(summaries[split]["metrics"])
        for split in ("train", "val", "test")
    }
    csv_path = output_dir / "per_frame.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "summary.json"
    result = {
        "experiment": "kitti_stage4_multidrive_aligned_temporal_difference",
        "git_revision": expected_revision,
        "device": str(DEVICE),
        "camera": camera,
        "fixed_dt_s": fixed_dt_s,
        "fixed_dt_tolerance_s": tolerance,
        "future_feature_stage": EXPECTED_STAGE,
        "target_flow_top_stage": validation["target_flow_top_stage"],
        "train_drives": TRAIN_DRIVES,
        "val_drives": VAL_DRIVES,
        "frozen_test_drives": FROZEN_TEST_DRIVES,
        "selection_split": "val_only",
        "test_access_policy": (
            "single_access_after_best_validation_checkpoint_selection"
        ),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": receipt_payload["checkpoint_sha256"],
            **validation,
        },
        "summaries": summaries,
        "per_drive": per_drive,
        "same_stage_copy_current_gates": gates,
        "per_frame_csv": "per_frame.csv",
        "frozen_test_receipt": str(receipt_path),
    }
    with summary_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    completed_receipt = finalize_frozen_test_receipt(
        receipt_path,
        receipt_payload,
        summary_path,
    )
    with (output_dir / "frozen_test_receipt.json").open("w") as handle:
        json.dump(completed_receipt, handle, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "row_count": len(rows),
                "val_gate": gates["val"],
                "frozen_test_gate": gates["test"],
                "frozen_test_receipt": completed_receipt,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
