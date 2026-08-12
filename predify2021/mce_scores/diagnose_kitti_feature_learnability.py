import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights, vgg16
from tqdm import tqdm

from predify2021.mce_scores.kitti_pairs import KITTIMultiHorizonFrameDataset


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-12
STAGE_INDICES = {"stage3": 2, "stage4": 3, "stage5": 4}
STAGE_STRIDES = {"stage3": 4, "stage4": 8, "stage5": 16}


def _parse_ints(value, name):
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers, got {value!r}.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates, got {values}.")
    return values


def _parse_drives(value, name):
    drives = tuple(item.strip() for item in value.split(",") if item.strip())
    if not drives:
        raise ValueError(f"{name} must contain at least one drive.")
    return drives


class VGGStageExtractor(nn.Module):
    """ImageNet VGG16 stages with the same boundaries as PVGG16TargetFlow."""

    def __init__(self, features):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(*features[:4]),
                nn.Sequential(*features[4:9]),
                nn.Sequential(*features[9:16]),
                nn.Sequential(*features[16:23]),
                nn.Sequential(*features[23:30]),
            ]
        )

    def forward(self, images):
        outputs = []
        current = images
        for stage in self.stages:
            current = stage(current)
            outputs.append(current)
        return {name: outputs[index] for name, index in STAGE_INDICES.items()}


def build_vgg_stage_extractor():
    backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model = VGGStageExtractor(backbone.features)
    model.eval()
    return model.to(DEVICE)


def translate_feature(feature, dy, dx):
    """Translate a BCHW feature map with zero fill and no wraparound."""
    if feature.ndim != 4:
        raise ValueError(f"Expected BCHW feature tensor, got shape {tuple(feature.shape)}.")
    dy = int(dy)
    dx = int(dx)
    height, width = feature.shape[-2:]
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError(
            f"Shift {(dy, dx)} must be smaller than feature shape {(height, width)}."
        )

    translated = torch.zeros_like(feature)
    source_y_start = max(0, -dy)
    source_y_stop = min(height, height - dy)
    source_x_start = max(0, -dx)
    source_x_stop = min(width, width - dx)
    target_y_start = max(0, dy)
    target_y_stop = min(height, height + dy)
    target_x_start = max(0, dx)
    target_x_stop = min(width, width + dx)
    translated[..., target_y_start:target_y_stop, target_x_start:target_x_stop] = (
        feature[..., source_y_start:source_y_stop, source_x_start:source_x_stop]
    )
    return translated


def best_oracle_translation_mse(current, target, max_shift_cells=1):
    """Future-selected integer translation oracle over a small square grid."""
    if current.shape != target.shape:
        raise ValueError(
            f"Current and target feature shapes must match: {current.shape} != {target.shape}."
        )
    max_shift_cells = int(max_shift_cells)
    if max_shift_cells < 0:
        raise ValueError("max_shift_cells must be non-negative.")

    batch_size = current.shape[0]
    best_mse = torch.full(
        (batch_size,),
        float("inf"),
        dtype=current.dtype,
        device=current.device,
    )
    best_dy = torch.zeros(batch_size, dtype=torch.int64, device=current.device)
    best_dx = torch.zeros(batch_size, dtype=torch.int64, device=current.device)
    for dy in range(-max_shift_cells, max_shift_cells + 1):
        for dx in range(-max_shift_cells, max_shift_cells + 1):
            shifted = translate_feature(current, dy=dy, dx=dx)
            candidate_mse = (shifted - target).square().flatten(1).mean(dim=1)
            improved = candidate_mse < best_mse
            best_mse = torch.where(improved, candidate_mse, best_mse)
            best_dy = torch.where(improved, dy, best_dy)
            best_dx = torch.where(improved, dx, best_dx)
    return best_mse, best_dy, best_dx


def compute_feature_baselines(previous, current, next_feature, target, horizon, max_shift_cells=1):
    """Compute causal copy/velocity metrics and a noncausal translation oracle."""
    tensors = (previous, current, next_feature, target)
    if any(tensor.shape != current.shape for tensor in tensors):
        raise ValueError("All feature tensors must have the same shape.")
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")

    previous_delta = current - previous
    next_delta = next_feature - current
    velocity_prediction = current + horizon * previous_delta
    copy_mse = (current - target).square().flatten(1).mean(dim=1)
    velocity_mse = (velocity_prediction - target).square().flatten(1).mean(dim=1)
    oracle_mse, oracle_dy, oracle_dx = best_oracle_translation_mse(
        current,
        target,
        max_shift_cells=max_shift_cells,
    )
    delta_cosine = F.cosine_similarity(
        previous_delta.flatten(1),
        next_delta.flatten(1),
        dim=1,
        eps=EPS,
    )
    improvement = copy_mse - velocity_mse
    return {
        "copy_mse": copy_mse,
        "velocity_mse": velocity_mse,
        "oracle_translation_mse": oracle_mse,
        "copy_minus_velocity_mse": improvement,
        "copy_minus_velocity_over_copy": improvement / copy_mse.clamp_min(EPS),
        "copy_minus_oracle_mse": copy_mse - oracle_mse,
        "copy_minus_oracle_over_copy": (copy_mse - oracle_mse)
        / copy_mse.clamp_min(EPS),
        "adjacent_delta_cosine": delta_cosine,
        "oracle_shift_dy_cells": oracle_dy,
        "oracle_shift_dx_cells": oracle_dx,
    }


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


def summarize_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["stage"], int(row["horizon_frames"]))].append(row)

    summary = {}
    metric_names = (
        "copy_mse",
        "velocity_mse",
        "oracle_translation_mse",
        "copy_minus_velocity_mse",
        "copy_minus_velocity_over_copy",
        "copy_minus_oracle_mse",
        "copy_minus_oracle_over_copy",
        "adjacent_delta_cosine",
    )
    for (split, stage, horizon), group in sorted(grouped.items()):
        key = f"{split}/{stage}/h{horizon}"
        copy_mean = sum(float(row["copy_mse"]) for row in group) / len(group)
        velocity_mean = sum(float(row["velocity_mse"]) for row in group) / len(group)
        oracle_mean = sum(float(row["oracle_translation_mse"]) for row in group) / len(group)
        shift_counts = Counter(
            f"dy={int(row['oracle_shift_dy_cells'])},dx={int(row['oracle_shift_dx_cells'])}"
            for row in group
        )
        summary[key] = {
            "split": split,
            "stage": stage,
            "horizon_frames": horizon,
            "sample_count": len(group),
            "metrics": {
                name: _distribution([float(row[name]) for row in group])
                for name in metric_names
            },
            "ratio_of_mean_mse": {
                "copy_minus_velocity_over_copy": (copy_mean - velocity_mean)
                / max(copy_mean, EPS),
                "copy_minus_oracle_over_copy": (copy_mean - oracle_mean)
                / max(copy_mean, EPS),
            },
            "velocity_better_than_copy_fraction": sum(
                float(row["velocity_mse"]) < float(row["copy_mse"]) for row in group
            )
            / len(group),
            "oracle_shift_histogram": dict(sorted(shift_counts.items())),
        }
    return summary


def _row_from_metrics(base, metrics, batch_index):
    row = dict(base)
    for name, values in metrics.items():
        value = values[batch_index].item()
        row[name] = int(value) if name.endswith("_cells") else float(value)
    return row


def evaluate_drive(model, root, drive, split, camera, horizons, fixed_dt_s, tolerance, batch_size, max_shift_cells):
    offsets = tuple(sorted({1, 2, *(horizon + 1 for horizon in horizons)}))
    dataset = KITTIMultiHorizonFrameDataset(
        root,
        drive,
        camera=camera,
        horizons=offsets,
        fixed_dt_s=fixed_dt_s,
        dt_tolerance_s=tolerance,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(os.environ.get("PREDIFY_LEARNABILITY_NUM_WORKERS", "0")),
        pin_memory=DEVICE.type == "cuda",
    )
    offset_to_position = {offset: index + 1 for index, offset in enumerate(offsets)}
    rows = []
    dataset_index = 0

    with torch.inference_mode():
        for previous_images, future_images, previous_names, future_names in tqdm(
            loader,
            desc=f"{split}:{Path(drive).name}",
        ):
            batch_count = previous_images.shape[0]
            frames = torch.cat([previous_images.unsqueeze(1), future_images], dim=1)
            frames = frames.to(DEVICE, non_blocking=True)
            flat_frames = frames.reshape(-1, *frames.shape[2:])
            stage_features = model(flat_frames)

            for stage, flat_feature in stage_features.items():
                feature = flat_feature.reshape(batch_count, len(offsets) + 1, *flat_feature.shape[1:])
                previous = feature[:, 0]
                current = feature[:, offset_to_position[1]]
                next_feature = feature[:, offset_to_position[2]]
                for horizon in horizons:
                    target_position = offset_to_position[horizon + 1]
                    target = feature[:, target_position]
                    metrics = compute_feature_baselines(
                        previous,
                        current,
                        next_feature,
                        target,
                        horizon=horizon,
                        max_shift_cells=max_shift_cells,
                    )
                    for batch_index in range(batch_count):
                        sample_index = dataset_index + batch_index
                        raw_start = int(dataset.valid_start_indices[sample_index])
                        target_name_index = offsets.index(horizon + 1)
                        base = {
                            "split": split,
                            "drive": drive,
                            "camera": camera,
                            "sample_index": sample_index,
                            "previous_raw_frame_index": raw_start,
                            "origin_raw_frame_index": raw_start + 1,
                            "target_raw_frame_index": raw_start + horizon + 1,
                            "previous_frame_name": previous_names[batch_index],
                            "origin_frame_name": future_names[offsets.index(1)][batch_index],
                            "target_frame_name": future_names[target_name_index][batch_index],
                            "stage": stage,
                            "stage_channels": int(current.shape[1]),
                            "stage_height": int(current.shape[2]),
                            "stage_width": int(current.shape[3]),
                            "stage_stride_input_pixels": STAGE_STRIDES[stage],
                            "horizon_frames": horizon,
                            "horizon_seconds": horizon * fixed_dt_s,
                        }
                        rows.append(_row_from_metrics(base, metrics, batch_index))
            dataset_index += batch_count

    return rows, {
        "drive": drive,
        "split": split,
        "sample_count": len(dataset),
        "raw_frame_count": len(dataset.frame_paths),
        "offsets_from_previous_frame": offsets,
        "time_filter_stats": dataset.time_filter_stats,
    }


def main():
    output_dir = Path(os.environ["PREDIFY_LEARNABILITY_OUTPUT_DIR"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    git_revision = os.environ["PREDIFY_GIT_REVISION"]
    root = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    train_drives = _parse_drives(os.environ["PREDIFY_TRAIN_DRIVES"], "PREDIFY_TRAIN_DRIVES")
    val_drives = _parse_drives(os.environ["PREDIFY_VAL_DRIVES"], "PREDIFY_VAL_DRIVES")
    horizons = _parse_ints(
        os.environ.get("PREDIFY_LEARNABILITY_HORIZONS", "1,2,3,5"),
        "PREDIFY_LEARNABILITY_HORIZONS",
    )
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
    batch_size = int(os.environ.get("PREDIFY_LEARNABILITY_BATCH_SIZE", "4"))
    max_shift_cells = int(os.environ.get("PREDIFY_LEARNABILITY_MAX_SHIFT_CELLS", "1"))
    if batch_size <= 0:
        raise ValueError("PREDIFY_LEARNABILITY_BATCH_SIZE must be positive.")
    if max_shift_cells < 0:
        raise ValueError("PREDIFY_LEARNABILITY_MAX_SHIFT_CELLS must be non-negative.")

    model = build_vgg_stage_extractor()
    rows = []
    datasets = []
    for split, drives in (("train", train_drives), ("val", val_drives)):
        for drive in drives:
            drive_rows, dataset_info = evaluate_drive(
                model,
                root=root,
                drive=drive,
                split=split,
                camera=camera,
                horizons=horizons,
                fixed_dt_s=fixed_dt_s,
                tolerance=tolerance,
                batch_size=batch_size,
                max_shift_cells=max_shift_cells,
            )
            rows.extend(drive_rows)
            datasets.append(dataset_info)

    csv_path = output_dir / "per_frame.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "experiment": "kitti_vgg_feature_learnability_matrix",
        "git_revision": git_revision,
        "device": str(DEVICE),
        "backbone": "torchvision VGG16 IMAGENET1K_V1",
        "preprocessing": "Resize(224), CenterCrop(224), ImageNet normalization",
        "camera": camera,
        "fixed_dt_s": fixed_dt_s,
        "fixed_dt_tolerance_s": tolerance,
        "horizons_frames": horizons,
        "horizons_seconds": {str(horizon): horizon * fixed_dt_s for horizon in horizons},
        "stages": {
            stage: {
                "pvgg_stage_index": index + 1,
                "stride_input_pixels": STAGE_STRIDES[stage],
                "oracle_max_shift_cells": max_shift_cells,
                "oracle_max_shift_input_pixels": max_shift_cells * STAGE_STRIDES[stage],
            }
            for stage, index in STAGE_INDICES.items()
        },
        "definitions": {
            "forecast_origin": "F_t; each sample is loaded from t-1 through t+max(h)",
            "copy_current": "Fhat_(t+h)=F_t",
            "constant_velocity": "Fhat_(t+h)=F_t+h*(F_t-F_(t-1))",
            "oracle_translation": (
                "future-selected integer translation of F_t over dy,dx in "
                f"[-{max_shift_cells},{max_shift_cells}] feature cells, zero fill, full-map MSE"
            ),
            "oracle_status": "noncausal diagnostic only; not a prediction result",
            "adjacent_delta_cosine": "cos(F_t-F_(t-1), F_(t+1)-F_t)",
            "velocity_improvement_fraction": "(MSE_copy-MSE_velocity)/MSE_copy",
        },
        "fair_comparison_policy": "all horizons use starts valid through max(h)+1 from t-1",
        "datasets": datasets,
        "per_frame_csv": "per_frame.csv",
        "matrix": summarize_rows(rows),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "per_frame_csv": str(csv_path),
                "row_count": len(rows),
                "matrix_entries": len(result["matrix"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
