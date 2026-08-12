import csv
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from predify2021.mce_scores.kitti_pairs import KITTIMultiHorizonFrameDataset
from predify2021.model_factory.get_model import get_model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-12
COPY_CURRENT_REFERENCE = {
    "feature_mse": 0.06008010,
    "feature_cosine": 0.91990469,
    "normalized_feature_error": 0.37576611,
}
TEMPORAL_ALIGNMENT_DEFINITION = (
    "local_match_F_previous_to_F_current_target_coordinates;"
    "use_matched_source_not_historical_future_warp"
)
ALIGNED_DIFFERENCE_DEFINITION = (
    "D_current=F_current-align(F_previous,F_current);"
    "predictor_input=[F_current,D_current]"
)


def validate_checkpoint(checkpoint, expected_revision):
    config = checkpoint.get("config")
    if not isinstance(config, dict) or config.get("prediction_task") != "future_feature":
        raise ValueError("Checkpoint is not a future-feature model.")
    required_config = {
        "git_revision": expected_revision,
        "future_feature_history_mode": "aligned_difference",
        "future_feature_temporal_fusion_mode": "none",
        "future_feature_temporal_fusion_architecture": "none",
        "future_feature_temporal_alignment": TEMPORAL_ALIGNMENT_DEFINITION,
        "future_feature_history_definition": ALIGNED_DIFFERENCE_DEFINITION,
        "future_feature_prediction_form": "current_residual",
        "future_motion_radius": 1,
        "future_motion_patch_size": 3,
        "pretrained": True,
        "train_backbone": False,
        "feedback_decoder_trainable": False,
        "top_target_source": "student_self",
        "stream_mode": True,
        "reset_each_frame": False,
        "shuffle_train_pairs": False,
        "shuffle_val_pairs": False,
        "local_reconstruction_weight": 0.0,
        "optimizer_created": True,
    }
    for key, required_value in required_config.items():
        if config.get(key) != required_value:
            raise ValueError(
                f"Expected {key}={required_value!r}, got {config.get(key)!r}."
            )
    if checkpoint.get("checkpoint_kind") != "best_val_future_feature_mse":
        raise ValueError("Checkpoint is not selected by best validation feature MSE.")
    names = tuple(config.get("optimized_parameter_names", ()))
    if not names or not all(
        name.startswith("future_feature_predictor.") for name in names
    ):
        raise ValueError(
            "Aligned temporal difference may optimize only the Future Predictor."
        )
    return {
        "history_mode": config["future_feature_history_mode"],
        "fusion_mode": config["future_feature_temporal_fusion_mode"],
        "predictor_kernel_size": int(config["future_feature_predictor_kernel_size"]),
        "motion_radius": int(config["future_motion_radius"]),
        "motion_patch_size": int(config["future_motion_patch_size"]),
        "selected_epoch": checkpoint.get("selected_epoch"),
        "optimized_parameter_names": names,
    }


def build_model(checkpoint, validation):
    config = checkpoint["config"]
    model = get_model(
        "pvgg_tf",
        pretrained=False,
        target_flow_mode=config.get("target_flow_mode", "recursive"),
        temporal_target_mode=config.get("temporal_target_mode", "next_top"),
        temporal_horizons=tuple(config.get("temporal_horizons", (1,))),
        dynamic_error=config.get("dynamic_error", True),
        error_state_mode=config.get("error_state_mode", "ema"),
        local_loss_error_source=config.get("local_loss_error_source", "instant"),
        error_sample_time=config.get("error_sample_time", 0.1035),
        error_time_constant=config.get("error_time_constant", 0.5),
        error_gain=config.get("error_gain", 1.0),
        temporal_error_sample_time=config.get("temporal_error_sample_time", 0.1035),
        temporal_error_time_constant=config.get("temporal_error_time_constant", 0.5),
        temporal_error_gain=config.get("temporal_error_gain", 1.0),
        task="future_feature",
        future_feature_history_mode=validation["history_mode"],
        future_feature_temporal_fusion_mode=validation["fusion_mode"],
        future_feature_predictor_kernel_size=validation["predictor_kernel_size"],
        future_feature_prediction_form="current_residual",
        future_motion_radius=validation["motion_radius"],
        future_motion_patch_size=validation["motion_patch_size"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if model.temporal_fusion_module is not None:
        raise ValueError("Aligned temporal difference must not create Temporal Fusion.")
    model.eval()
    return model.to(DEVICE)


def _distribution(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    quantiles = torch.quantile(
        tensor,
        torch.tensor([0.05, 0.5, 0.9, 0.95], dtype=torch.float64),
    )
    return {
        "count": len(values),
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "minimum": tensor.min().item(),
        "p05": quantiles[0].item(),
        "p50": quantiles[1].item(),
        "p90": quantiles[2].item(),
        "p95": quantiles[3].item(),
        "maximum": tensor.max().item(),
    }


def evaluate_split(model, dataset, split, drive, fixed_dt_s):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.reset()
    rows = []
    with torch.inference_mode():
        for sample_index, (current, futures, current_names, future_names) in enumerate(
            tqdm(loader, desc=f"aligned_temporal_difference:{split}")
        ):
            current = current.to(DEVICE)
            future = futures[:, 0].to(DEVICE)
            model.step_frame(
                current,
                top_target_provider=lambda: model.extract_top_forward_feature(
                    future,
                    detach=True,
                ),
            )
            outputs = model.future_prediction_outputs
            predicted = outputs["predicted_future_top"].float()
            target = outputs["future_top_target"].float()
            current_top = outputs["current_top"].float()
            prediction_base = outputs["prediction_base_top"].float()
            predicted_flat = predicted.flatten(1)
            target_flat = target.flatten(1)
            alignment_applied = bool(outputs["aligned_difference_applied"])
            raw_previous = outputs["aligned_difference_raw_previous_top"]
            aligned_previous = outputs["aligned_difference_previous_top"]
            temporal_difference = outputs["aligned_temporal_difference_top"].float()
            matching_cost = outputs["alignment_patch_matching_cost"]
            motion_dy = outputs["motion_dy"]
            motion_dx = outputs["motion_dx"]
            prediction_base_current_max_abs = float(
                (prediction_base - current_top).abs().max().item()
            )
            if prediction_base_current_max_abs != 0.0:
                raise RuntimeError("Prediction base changed F_t in aligned-difference mode.")
            if outputs["temporal_fusion_applied"]:
                raise RuntimeError("Temporal Fusion ran in aligned-difference mode.")
            if alignment_applied:
                raw_alignment_mse = float(
                    (raw_previous.float() - current_top).square().mean().item()
                )
                aligned_alignment_mse = float(
                    (aligned_previous.float() - current_top).square().mean().item()
                )
                alignment_reduction = (
                    raw_alignment_mse - aligned_alignment_mse
                ) / max(raw_alignment_mse, EPS)
                motion_magnitude = torch.sqrt(
                    motion_dy.float().square() + motion_dx.float().square()
                )
                nonzero_motion_fraction = float(
                    ((motion_dy != 0) | (motion_dx != 0)).float().mean().item()
                )
                mean_motion_magnitude = float(motion_magnitude.mean().item())
                mean_matching_cost = float(matching_cost.float().mean().item())
            else:
                raw_alignment_mse = ""
                aligned_alignment_mse = ""
                alignment_reduction = ""
                nonzero_motion_fraction = ""
                mean_motion_magnitude = ""
                mean_matching_cost = ""
            raw_start = int(dataset.valid_start_indices[sample_index])
            rows.append(
                {
                    "condition": "aligned_temporal_difference",
                    "split": split,
                    "drive": drive,
                    "sample_index": sample_index,
                    "current_raw_frame_index": raw_start,
                    "future_raw_frame_index": raw_start + 1,
                    "current_frame_name": current_names[0],
                    "future_frame_name": future_names[0][0],
                    "horizon_frames": 1,
                    "horizon_seconds": fixed_dt_s,
                    "temporal_fusion_applied": False,
                    "alignment_applied": alignment_applied,
                    "feature_mse": float((predicted - target).square().mean().item()),
                    "feature_cosine": float(
                        F.cosine_similarity(predicted_flat, target_flat, dim=1).mean().item()
                    ),
                    "normalized_feature_error": float(
                        (
                            torch.linalg.vector_norm(predicted_flat - target_flat, dim=1)
                            / torch.linalg.vector_norm(target_flat, dim=1).clamp_min(EPS)
                        ).mean().item()
                    ),
                    "copy_mse": float((current_top - target).square().mean().item()),
                    "raw_previous_to_current_mse": raw_alignment_mse,
                    "aligned_previous_to_current_mse": aligned_alignment_mse,
                    "alignment_mse_reduction_fraction": alignment_reduction,
                    "alignment_patch_matching_mse": mean_matching_cost,
                    "nonzero_motion_fraction": nonzero_motion_fraction,
                    "mean_motion_magnitude_cells": mean_motion_magnitude,
                    "prediction_base_current_max_abs": (
                        prediction_base_current_max_abs
                    ),
                    "aligned_temporal_difference_rms": math.sqrt(
                        float(temporal_difference.square().mean().item())
                    ),
                    "predicted_residual_rms": math.sqrt(
                        float(outputs["predicted_residual_top"].float().square().mean().item())
                    ),
                }
            )
    return rows


def summarize_rows(rows):
    all_metrics = (
        "feature_mse",
        "feature_cosine",
        "normalized_feature_error",
        "copy_mse",
        "prediction_base_current_max_abs",
        "aligned_temporal_difference_rms",
        "predicted_residual_rms",
    )
    aligned_metrics = (
        "raw_previous_to_current_mse",
        "aligned_previous_to_current_mse",
        "alignment_mse_reduction_fraction",
        "alignment_patch_matching_mse",
        "nonzero_motion_fraction",
        "mean_motion_magnitude_cells",
    )
    return {
        "sample_count": len(rows),
        "temporal_fusion_applied_count": 0,
        "alignment_applied_count": sum(bool(row["alignment_applied"]) for row in rows),
        "metrics": {
            **{
                metric: _distribution([float(row[metric]) for row in rows])
                for metric in all_metrics
            },
            **{
                metric: _distribution(
                    [float(row[metric]) for row in rows if row[metric] != ""]
                )
                for metric in aligned_metrics
            },
        },
    }


def evaluate_gate(metrics):
    observed = {
        key: float(metrics[key]["mean"]) for key in COPY_CURRENT_REFERENCE
    }
    checks = {
        "feature_mse_below_copy": (
            observed["feature_mse"] < COPY_CURRENT_REFERENCE["feature_mse"]
        ),
        "feature_cosine_above_copy": (
            observed["feature_cosine"] > COPY_CURRENT_REFERENCE["feature_cosine"]
        ),
        "normalized_error_below_copy": (
            observed["normalized_feature_error"]
            < COPY_CURRENT_REFERENCE["normalized_feature_error"]
        ),
    }
    return {
        "reference": COPY_CURRENT_REFERENCE,
        "observed": observed,
        "checks": checks,
        "all_three_pass": all(checks.values()),
        "deltas_observed_minus_reference": {
            key: observed[key] - COPY_CURRENT_REFERENCE[key] for key in observed
        },
    }


def main():
    output_dir = Path(os.environ["PREDIFY_ALIGNED_DIFFERENCE_EVAL_OUTPUT_DIR"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_revision = os.environ["PREDIFY_GIT_REVISION"]
    checkpoint_path = Path(os.environ["PREDIFY_ALIGNED_DIFFERENCE_CHECKPOINT"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validation = validate_checkpoint(checkpoint, expected_revision)
    config = checkpoint["config"]
    root = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
    train_drive = os.environ["PREDIFY_TRAIN_DRIVES"]
    val_drive = os.environ["PREDIFY_VAL_DRIVES"]
    if tuple(config.get("train_drives", ())) != (train_drive,):
        raise ValueError("Checkpoint training drive does not match the formal replay.")
    if tuple(config.get("val_drives", ())) != (val_drive,):
        raise ValueError("Checkpoint validation drive does not match the formal replay.")
    datasets = {
        "train": KITTIMultiHorizonFrameDataset(
            root,
            train_drive,
            camera=camera,
            horizons=(1,),
            fixed_dt_s=fixed_dt_s,
            dt_tolerance_s=tolerance,
        ),
        "val": KITTIMultiHorizonFrameDataset(
            root,
            val_drive,
            camera=camera,
            horizons=(1,),
            fixed_dt_s=fixed_dt_s,
            dt_tolerance_s=tolerance,
        ),
    }
    model = build_model(checkpoint, validation)
    rows = []
    summaries = {}
    for split, drive in (("train", train_drive), ("val", val_drive)):
        split_rows = evaluate_split(
            model,
            datasets[split],
            split=split,
            drive=drive,
            fixed_dt_s=fixed_dt_s,
        )
        rows.extend(split_rows)
        summaries[split] = summarize_rows(split_rows)
    gate = evaluate_gate(summaries["val"]["metrics"])

    csv_path = output_dir / "per_frame.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "experiment": "kitti_stage5_aligned_temporal_difference_predictor",
        "git_revision": expected_revision,
        "device": str(DEVICE),
        "camera": camera,
        "fixed_dt_s": fixed_dt_s,
        "fixed_dt_tolerance_s": tolerance,
        "train_drive": train_drive,
        "val_drive": val_drive,
        "checkpoint": {"path": str(checkpoint_path), **validation},
        "summaries": summaries,
        "heldout_copy_current_gate": gate,
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
                "heldout_copy_current_gate": gate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
