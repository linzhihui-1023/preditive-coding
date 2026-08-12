import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from predify2021.mce_scores.kitti_pairs import KITTIMultiHorizonFrameDataset
from predify2021.model_factory.get_model import get_model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-12
EXPECTED_FORMS = {
    "copy_current": ("copy_current", "current_residual"),
    "historical_warp": ("none", "historical_warp"),
    "warp_residual": ("none", "historical_warp_residual"),
}


def _parse_checkpoints(value):
    checkpoints = {}
    for item in value.split(","):
        if not item.strip():
            continue
        label, separator, path = item.partition("=")
        if not separator or not label.strip() or not path.strip():
            raise ValueError(f"Invalid checkpoint mapping: {item!r}.")
        checkpoints[label.strip()] = Path(path.strip())
    if set(checkpoints) != set(EXPECTED_FORMS):
        raise ValueError(
            f"Expected checkpoint labels {sorted(EXPECTED_FORMS)}, got {sorted(checkpoints)}."
        )
    return checkpoints


def validate_checkpoint(label, checkpoint, expected_revision):
    config = checkpoint.get("config")
    if not isinstance(config, dict) or config.get("prediction_task") != "future_feature":
        raise ValueError(f"{label} is not a future-feature checkpoint.")
    history_mode = config.get("future_feature_history_mode")
    prediction_form = config.get("future_feature_prediction_form", "current_residual")
    if prediction_form == "residual_Fhat_next=F_current+delta_hat":
        prediction_form = "current_residual"
    expected_history, expected_form = EXPECTED_FORMS[label]
    if (history_mode, prediction_form) != (expected_history, expected_form):
        raise ValueError(
            f"{label} expected history/form {(expected_history, expected_form)}, "
            f"got {(history_mode, prediction_form)}."
        )
    checkpoint_revision = config.get("git_revision")
    if checkpoint_revision != expected_revision:
        raise ValueError(
            f"{label} checkpoint revision {checkpoint_revision!r} != {expected_revision!r}."
        )
    if checkpoint.get("checkpoint_kind") != "best_val_future_feature_mse":
        raise ValueError(f"{label} is not a best-validation future-feature checkpoint.")
    return {
        "history_mode": history_mode,
        "prediction_form": prediction_form,
        "predictor_kernel_size": int(config["future_feature_predictor_kernel_size"]),
        "motion_radius": int(config.get("future_motion_radius", 1)),
        "motion_patch_size": int(config.get("future_motion_patch_size", 3)),
        "selected_epoch": checkpoint.get("selected_epoch"),
        "checkpoint_revision": checkpoint_revision,
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
        future_feature_predictor_kernel_size=validation["predictor_kernel_size"],
        future_feature_prediction_form=validation["prediction_form"],
        future_motion_radius=validation["motion_radius"],
        future_motion_patch_size=validation["motion_patch_size"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model.to(DEVICE)


def _distribution(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    quantiles = torch.quantile(
        tensor,
        torch.tensor([0.05, 0.5, 0.9, 0.95], dtype=torch.float64),
    )
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "minimum": tensor.min().item(),
        "p05": quantiles[0].item(),
        "p50": quantiles[1].item(),
        "p90": quantiles[2].item(),
        "p95": quantiles[3].item(),
        "maximum": tensor.max().item(),
    }


def evaluate_split(model, dataset, label, split, drive, fixed_dt_s):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.reset()
    rows = []
    with torch.inference_mode():
        for sample_index, (current, futures, current_names, future_names) in enumerate(
            tqdm(loader, desc=f"{label}:{split}")
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
            base = outputs["prediction_base_top"].float()
            predicted_residual = outputs["predicted_residual_top"].float()
            target_residual = outputs["target_residual_top"].float()
            predicted_flat = predicted.flatten(1)
            target_flat = target.flatten(1)
            raw_start = int(dataset.valid_start_indices[sample_index])
            coverage = outputs["warp_coverage_fraction"]
            collision = outputs["warp_collision_fraction"]
            rows.append(
                {
                    "condition": label,
                    "split": split,
                    "drive": drive,
                    "sample_index": sample_index,
                    "current_raw_frame_index": raw_start,
                    "future_raw_frame_index": raw_start + 1,
                    "current_frame_name": current_names[0],
                    "future_frame_name": future_names[0][0],
                    "horizon_frames": 1,
                    "horizon_seconds": fixed_dt_s,
                    "bootstrap_without_history": outputs["motion_dy"] is None,
                    "feature_mse": float((predicted - target).square().mean().item()),
                    "copy_mse": float((current_top - target).square().mean().item()),
                    "prediction_base_mse": float((base - target).square().mean().item()),
                    "residual_mse": float(
                        (predicted_residual - target_residual).square().mean().item()
                    ),
                    "feature_cosine": float(
                        F.cosine_similarity(predicted_flat, target_flat, dim=1).mean().item()
                    ),
                    "normalized_feature_error": float(
                        (
                            torch.linalg.vector_norm(predicted_flat - target_flat, dim=1)
                            / torch.linalg.vector_norm(target_flat, dim=1).clamp_min(EPS)
                        ).mean().item()
                    ),
                    "predicted_residual_rms": math.sqrt(
                        float(predicted_residual.square().mean().item())
                    ),
                    "target_residual_rms": math.sqrt(
                        float(target_residual.square().mean().item())
                    ),
                    "warp_coverage_fraction": (
                        "" if coverage is None else float(coverage.mean().item())
                    ),
                    "warp_collision_fraction": (
                        "" if collision is None else float(collision.mean().item())
                    ),
                }
            )
    return rows


def summarize_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["split"])].append(row)
    summary = {}
    metric_names = (
        "feature_mse",
        "copy_mse",
        "prediction_base_mse",
        "residual_mse",
        "feature_cosine",
        "normalized_feature_error",
        "predicted_residual_rms",
        "target_residual_rms",
    )
    for (condition, split), group in sorted(grouped.items()):
        summary[f"{condition}/{split}"] = {
            "condition": condition,
            "split": split,
            "sample_count": len(group),
            "bootstrap_frame_count": sum(
                str(row["bootstrap_without_history"]).lower() == "true" for row in group
            ),
            "metrics": {
                metric: _distribution([float(row[metric]) for row in group])
                for metric in metric_names
            },
        }
    return summary


def main():
    output_dir = Path(os.environ["PREDIFY_WARP_MATRIX_EVAL_OUTPUT_DIR"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_revision = os.environ["PREDIFY_GIT_REVISION"]
    checkpoints = _parse_checkpoints(os.environ["PREDIFY_WARP_MATRIX_CHECKPOINTS"])
    root = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
    train_drive = os.environ["PREDIFY_TRAIN_DRIVES"]
    val_drive = os.environ["PREDIFY_VAL_DRIVES"]
    datasets = {
        "train": KITTIMultiHorizonFrameDataset(
            root, train_drive, camera=camera, horizons=(1,), fixed_dt_s=fixed_dt_s,
            dt_tolerance_s=tolerance,
        ),
        "val": KITTIMultiHorizonFrameDataset(
            root, val_drive, camera=camera, horizons=(1,), fixed_dt_s=fixed_dt_s,
            dt_tolerance_s=tolerance,
        ),
    }

    rows = []
    checkpoint_records = {}
    for label, checkpoint_path in checkpoints.items():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validation = validate_checkpoint(label, checkpoint, expected_revision)
        model = build_model(checkpoint, validation)
        checkpoint_records[label] = {
            "path": str(checkpoint_path),
            **validation,
        }
        for split, drive in (("train", train_drive), ("val", val_drive)):
            rows.extend(
                evaluate_split(
                    model,
                    datasets[split],
                    label=label,
                    split=split,
                    drive=drive,
                    fixed_dt_s=fixed_dt_s,
                )
            )

    csv_path = output_dir / "per_frame.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    matrix = summarize_rows(rows)
    comparisons = {}
    for split in ("train", "val"):
        copy_mse = matrix[f"copy_current/{split}"]["metrics"]["feature_mse"]["mean"]
        warp_mse = matrix[f"historical_warp/{split}"]["metrics"]["feature_mse"]["mean"]
        residual_mse = matrix[f"warp_residual/{split}"]["metrics"]["feature_mse"]["mean"]
        comparisons[split] = {
            "historical_warp_vs_copy_fraction": (copy_mse - warp_mse) / copy_mse,
            "warp_residual_vs_copy_fraction": (copy_mse - residual_mse) / copy_mse,
            "warp_residual_vs_historical_warp_fraction": (warp_mse - residual_mse)
            / warp_mse,
        }
    result = {
        "experiment": "kitti_stage5_warp_residual_matrix",
        "git_revision": expected_revision,
        "device": str(DEVICE),
        "camera": camera,
        "fixed_dt_s": fixed_dt_s,
        "fixed_dt_tolerance_s": tolerance,
        "train_drive": train_drive,
        "val_drive": val_drive,
        "conditions": EXPECTED_FORMS,
        "checkpoints": checkpoint_records,
        "matrix": matrix,
        "comparisons": comparisons,
        "per_frame_csv": "per_frame.csv",
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({"summary_path": str(summary_path), "row_count": len(rows), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
