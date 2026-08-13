import csv
import json
import os
from pathlib import Path

import torch

from predify2021.mce_scores.evaluate_kitti_aligned_temporal_difference import (
    DEVICE,
    build_model,
    evaluate_gate,
    evaluate_split,
    summarize_rows,
    validate_checkpoint,
)
from predify2021.mce_scores.kitti_pairs import (
    KITTIMultiHorizonFrameDataset,
    build_same_drive_train_val_test_subsets,
)


EXPECTED_STAGE = 4
EXPECTED_SPLIT_NAME = "chronological_raw_frames_60_20_20"


def validate_same_drive_checkpoint(checkpoint, expected_revision, expected_drive):
    validation = validate_checkpoint(
        checkpoint,
        expected_revision=expected_revision,
        expected_stage=EXPECTED_STAGE,
    )
    config = checkpoint["config"]
    required = {
        "formal_split": False,
        "same_drive_split": False,
        "same_drive_three_way_split": True,
        "same_drive_drive": expected_drive,
        "train_fraction": 0.6,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "test_selection_role": (
            "unseen_until_after_best_validation_checkpoint_selection"
        ),
        "shuffle_train_pairs": False,
        "shuffle_val_pairs": False,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Expected {key}={expected!r}, got {config.get(key)!r}.")
    if tuple(config.get("train_drives", ())) != (expected_drive,):
        raise ValueError("Checkpoint train drive does not match the same-drive diagnostic.")
    if tuple(config.get("val_drives", ())) != (expected_drive,):
        raise ValueError("Checkpoint validation drive does not match the same-drive diagnostic.")

    metadata = config.get("same_drive_split_metadata")
    if not isinstance(metadata, dict) or metadata.get("split_name") != EXPECTED_SPLIT_NAME:
        raise ValueError("Checkpoint lacks the chronological 60/20/20 split metadata.")
    if tuple(metadata.get("chronological_order", ())) != (
        "train",
        "val",
        "test",
    ):
        raise ValueError("Checkpoint split does not preserve train/val/test time order.")
    if metadata.get("shared_raw_frame_count") != 0 or metadata.get("shuffle") is not False:
        raise ValueError("Checkpoint split must have disjoint raw frames and no shuffling.")
    return {**validation, "split_metadata": metadata}


def main():
    output_dir = Path(os.environ["PREDIFY_SAME_DRIVE_DIAGNOSTIC_OUTPUT_DIR"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_revision = os.environ["PREDIFY_GIT_REVISION"]
    expected_drive = os.environ["PREDIFY_KITTI_DRIVE"]
    checkpoint_path = Path(os.environ["PREDIFY_SAME_DRIVE_DIAGNOSTIC_CHECKPOINT"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validation = validate_same_drive_checkpoint(
        checkpoint,
        expected_revision=expected_revision,
        expected_drive=expected_drive,
    )
    config = checkpoint["config"]

    root = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
    dataset = KITTIMultiHorizonFrameDataset(
        root,
        expected_drive,
        camera=camera,
        horizons=(1,),
        fixed_dt_s=fixed_dt_s,
        dt_tolerance_s=tolerance,
    )
    train_dataset, val_dataset, test_dataset, split_metadata = (
        build_same_drive_train_val_test_subsets(dataset)
    )
    if split_metadata != validation["split_metadata"]:
        raise ValueError("Reconstructed 60/20/20 split differs from checkpoint metadata.")

    model = build_model(checkpoint, validation)
    rows = []
    summaries = {}
    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    for split, split_dataset in datasets.items():
        split_rows = evaluate_split(
            model,
            split_dataset,
            split=split,
            drive=expected_drive,
            fixed_dt_s=fixed_dt_s,
        )
        rows.extend(split_rows)
        summaries[split] = summarize_rows(split_rows)

    gates = {
        split: evaluate_gate(summaries[split]["metrics"])
        for split in ("val", "test")
    }
    csv_path = output_dir / "per_frame.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "experiment": "kitti_stage4_same_drive_60_20_20_diagnostic",
        "git_revision": expected_revision,
        "device": str(DEVICE),
        "drive": expected_drive,
        "camera": camera,
        "fixed_dt_s": fixed_dt_s,
        "fixed_dt_tolerance_s": tolerance,
        "future_feature_stage": EXPECTED_STAGE,
        "target_flow_top_stage": validation["target_flow_top_stage"],
        "selection_split": "val_middle_20_percent",
        "test_access_policy": (
            "test_last_20_percent_evaluated_once_after_best_val_checkpoint_selection"
        ),
        "split_metadata": split_metadata,
        "checkpoint": {"path": str(checkpoint_path), **validation},
        "summaries": summaries,
        "same_stage_copy_current_gates": gates,
        "per_frame_csv": "per_frame.csv",
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "row_count": len(rows),
                "val_gate": gates["val"],
                "test_gate": gates["test"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
