import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from predify2021.mce_scores.diagnose_kitti_feature_learnability import (
    DEVICE,
    EPS,
    STAGE_STRIDES,
    build_vgg_stage_extractor,
    translate_feature,
)
from predify2021.mce_scores.kitti_pairs import KITTIMultiHorizonFrameDataset


LOCAL_HORIZONS = (1, 3)
RADII = (1, 2)
PATCH_SIZES = (1, 3)
STAGE_ORDER = ("stage5", "stage4")


def candidate_shifts(radius):
    radius = int(radius)
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}.")
    shifts = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    return tuple(
        sorted(
            shifts,
            key=lambda shift: (
                shift[0] * shift[0] + shift[1] * shift[1],
                abs(shift[0]) + abs(shift[1]),
                shift[0],
                shift[1],
            ),
        )
    )


def patch_descriptors(feature, patch_size):
    patch_size = int(patch_size)
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be a positive odd integer, got {patch_size}.")
    batch_size, channels, height, width = feature.shape
    unfolded = F.unfold(
        feature,
        kernel_size=patch_size,
        padding=patch_size // 2,
    )
    return unfolded.reshape(batch_size, channels * patch_size * patch_size, height, width)


def estimate_local_displacement(source, target, radius, patch_size):
    """Find a source-to-target displacement at every target location."""
    if source.shape != target.shape or source.ndim != 4:
        raise ValueError("source and target must be matching BCHW tensors.")
    source_descriptors = patch_descriptors(source, patch_size)
    target_descriptors = patch_descriptors(target, patch_size)
    batch_size, _, height, width = source.shape
    valid_source = torch.ones(
        batch_size,
        1,
        height,
        width,
        dtype=source.dtype,
        device=source.device,
    )
    best_cost = torch.full(
        (batch_size, height, width),
        float("inf"),
        dtype=source.dtype,
        device=source.device,
    )
    best_dy = torch.zeros_like(best_cost, dtype=torch.int64)
    best_dx = torch.zeros_like(best_cost, dtype=torch.int64)
    matched_source = torch.zeros_like(source)

    for dy, dx in candidate_shifts(radius):
        if abs(dy) >= height or abs(dx) >= width:
            continue
        candidate_descriptors = translate_feature(source_descriptors, dy=dy, dx=dx)
        candidate_source = translate_feature(source, dy=dy, dx=dx)
        valid = translate_feature(valid_source, dy=dy, dx=dx).squeeze(1) > 0.5
        cost = (candidate_descriptors - target_descriptors).square().mean(dim=1)
        cost = torch.where(valid, cost, torch.full_like(cost, float("inf")))
        improved = cost < best_cost
        best_cost = torch.where(improved, cost, best_cost)
        best_dy = torch.where(improved, dy, best_dy)
        best_dx = torch.where(improved, dx, best_dx)
        matched_source = torch.where(improved.unsqueeze(1), candidate_source, matched_source)

    return {
        "dy": best_dy,
        "dx": best_dx,
        "patch_matching_cost": best_cost,
        "matched_source": matched_source,
    }


def forward_splat_discrete(feature, dy_field, dx_field, radius):
    """Move current features forward; average collisions and copy-fill holes."""
    if feature.ndim != 4 or dy_field.shape != feature.shape[:1] + feature.shape[-2:]:
        raise ValueError("Motion fields must have shape BHW matching the BCHW feature.")
    if dx_field.shape != dy_field.shape:
        raise ValueError("dy_field and dx_field must have the same shape.")

    accumulated = torch.zeros_like(feature)
    weights = torch.zeros(
        feature.shape[0],
        1,
        feature.shape[2],
        feature.shape[3],
        dtype=feature.dtype,
        device=feature.device,
    )
    for dy, dx in candidate_shifts(radius):
        if abs(dy) >= feature.shape[2] or abs(dx) >= feature.shape[3]:
            continue
        mask = ((dy_field == dy) & (dx_field == dx)).unsqueeze(1).to(feature.dtype)
        accumulated = accumulated + translate_feature(feature * mask, dy=dy, dx=dx)
        weights = weights + translate_feature(mask, dy=dy, dx=dx)

    covered = weights > 0
    warped = torch.where(covered, accumulated / weights.clamp_min(1.0), feature)
    return {
        "warped": warped,
        "coverage_fraction": covered.flatten(1).float().mean(dim=1),
        "collision_fraction": (weights > 1).flatten(1).float().mean(dim=1),
    }


def _motion_statistics(dy, dx):
    magnitude = torch.sqrt(dy.float().square() + dx.float().square())
    return {
        "nonzero_motion_fraction": ((dy != 0) | (dx != 0)).flatten(1).float().mean(dim=1),
        "mean_motion_magnitude_cells": magnitude.flatten(1).mean(dim=1),
    }


def compute_local_future_matching(current, target, radius, patch_size):
    match = estimate_local_displacement(
        current,
        target,
        radius=radius,
        patch_size=patch_size,
    )
    copy_mse = (current - target).square().flatten(1).mean(dim=1)
    result_mse = (match["matched_source"] - target).square().flatten(1).mean(dim=1)
    improvement = copy_mse - result_mse
    return {
        "copy_mse": copy_mse,
        "result_mse": result_mse,
        "copy_minus_result_mse": improvement,
        "copy_minus_result_over_copy": improvement / copy_mse.clamp_min(EPS),
        "matching_patch_mse": match["patch_matching_cost"].flatten(1).mean(dim=1),
        "coverage_fraction": torch.ones_like(copy_mse),
        "collision_fraction": torch.zeros_like(copy_mse),
        **_motion_statistics(match["dy"], match["dx"]),
    }


def compute_causal_historical_warp(previous, current, target, radius, patch_size):
    motion = estimate_local_displacement(
        previous,
        current,
        radius=radius,
        patch_size=patch_size,
    )
    splat = forward_splat_discrete(
        current,
        motion["dy"],
        motion["dx"],
        radius=radius,
    )
    copy_mse = (current - target).square().flatten(1).mean(dim=1)
    result_mse = (splat["warped"] - target).square().flatten(1).mean(dim=1)
    improvement = copy_mse - result_mse
    return {
        "copy_mse": copy_mse,
        "result_mse": result_mse,
        "copy_minus_result_mse": improvement,
        "copy_minus_result_over_copy": improvement / copy_mse.clamp_min(EPS),
        "matching_patch_mse": motion["patch_matching_cost"].flatten(1).mean(dim=1),
        "coverage_fraction": splat["coverage_fraction"],
        "collision_fraction": splat["collision_fraction"],
        **_motion_statistics(motion["dy"], motion["dx"]),
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
        key = (
            row["method"],
            row["split"],
            row["stage"],
            int(row["horizon_frames"]),
            int(row["radius_cells"]),
            int(row["patch_size"]),
        )
        grouped[key].append(row)

    summary = {}
    metric_names = (
        "copy_mse",
        "result_mse",
        "copy_minus_result_mse",
        "copy_minus_result_over_copy",
        "matching_patch_mse",
        "nonzero_motion_fraction",
        "mean_motion_magnitude_cells",
        "coverage_fraction",
        "collision_fraction",
    )
    for key, group in sorted(grouped.items()):
        method, split, stage, horizon, radius, patch_size = key
        copy_mean = sum(float(row["copy_mse"]) for row in group) / len(group)
        result_mean = sum(float(row["result_mse"]) for row in group) / len(group)
        name = f"{method}/{split}/{stage}/h{horizon}/r{radius}/p{patch_size}"
        summary[name] = {
            "method": method,
            "split": split,
            "stage": stage,
            "horizon_frames": horizon,
            "radius_cells": radius,
            "patch_size": patch_size,
            "sample_count": len(group),
            "metrics": {
                metric: _distribution([float(row[metric]) for row in group])
                for metric in metric_names
            },
            "ratio_of_mean_mse": (copy_mean - result_mean) / max(copy_mean, EPS),
            "result_better_than_copy_fraction": sum(
                float(row["result_mse"]) < float(row["copy_mse"]) for row in group
            )
            / len(group),
        }
    return summary


def build_causal_gate(matrix):
    configurations = {}
    for radius in RADII:
        for patch_size in PATCH_SIZES:
            gains = {}
            for split in ("train", "val"):
                key = f"causal_historical_warp/{split}/stage5/h1/r{radius}/p{patch_size}"
                gains[split] = matrix[key]["ratio_of_mean_mse"]
            config_name = f"r{radius}/p{patch_size}"
            configurations[config_name] = {
                "train_gain_fraction": gains["train"],
                "val_gain_fraction": gains["val"],
                "passes_both_drives": gains["train"] > 0 and gains["val"] > 0,
            }
    return {
        "definition": (
            "A single radius/patch configuration must have aggregate stage5 h=1 "
            "causal-warp MSE below Copy-current on both drives."
        ),
        "configurations": configurations,
        "passed": any(config["passes_both_drives"] for config in configurations.values()),
    }


CSV_FIELDS = (
    "method",
    "causal",
    "split",
    "drive",
    "camera",
    "sample_index",
    "previous_raw_frame_index",
    "origin_raw_frame_index",
    "target_raw_frame_index",
    "previous_frame_name",
    "origin_frame_name",
    "target_frame_name",
    "stage",
    "stage_channels",
    "stage_height",
    "stage_width",
    "stage_stride_input_pixels",
    "horizon_frames",
    "horizon_seconds",
    "radius_cells",
    "radius_input_pixels",
    "patch_size",
    "copy_mse",
    "result_mse",
    "copy_minus_result_mse",
    "copy_minus_result_over_copy",
    "matching_patch_mse",
    "nonzero_motion_fraction",
    "mean_motion_magnitude_cells",
    "coverage_fraction",
    "collision_fraction",
)


def _tensor_metrics_to_row(base, metrics, batch_index):
    row = dict(base)
    for name, value in metrics.items():
        row[name] = float(value[batch_index].item())
    return row


def evaluate_phase(
    model,
    root,
    drives_by_split,
    camera,
    fixed_dt_s,
    tolerance,
    batch_size,
    method,
    stages,
):
    rows = []
    datasets = []
    offsets = (1, 2, 4)
    offset_to_position = {offset: index + 1 for index, offset in enumerate(offsets)}
    for split, drives in drives_by_split:
        for drive in drives:
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
                num_workers=int(os.environ.get("PREDIFY_LOCAL_MOTION_NUM_WORKERS", "0")),
                pin_memory=DEVICE.type == "cuda",
            )
            datasets.append(
                {
                    "method": method,
                    "stages": stages,
                    "split": split,
                    "drive": drive,
                    "sample_count": len(dataset),
                    "raw_frame_count": len(dataset.frame_paths),
                    "time_filter_stats": dataset.time_filter_stats,
                }
            )
            dataset_index = 0
            with torch.inference_mode():
                for previous_images, future_images, previous_names, future_names in tqdm(
                    loader,
                    desc=f"{method}:{'+'.join(stages)}:{split}",
                ):
                    batch_count = previous_images.shape[0]
                    frames = torch.cat([previous_images.unsqueeze(1), future_images], dim=1)
                    frames = frames.to(DEVICE, non_blocking=True)
                    flat_frames = frames.reshape(-1, *frames.shape[2:])
                    all_stage_features = model(flat_frames)
                    for stage in stages:
                        flat_feature = all_stage_features[stage]
                        feature = flat_feature.reshape(
                            batch_count,
                            len(offsets) + 1,
                            *flat_feature.shape[1:],
                        )
                        previous = feature[:, 0]
                        current = feature[:, offset_to_position[1]]
                        horizons = LOCAL_HORIZONS if method == "local_future_matching" else (1,)
                        for horizon in horizons:
                            target = feature[:, offset_to_position[horizon + 1]]
                            for radius in RADII:
                                for patch_size in PATCH_SIZES:
                                    if method == "local_future_matching":
                                        metrics = compute_local_future_matching(
                                            current,
                                            target,
                                            radius=radius,
                                            patch_size=patch_size,
                                        )
                                    else:
                                        metrics = compute_causal_historical_warp(
                                            previous,
                                            current,
                                            target,
                                            radius=radius,
                                            patch_size=patch_size,
                                        )
                                    for batch_index in range(batch_count):
                                        sample_index = dataset_index + batch_index
                                        raw_start = int(dataset.valid_start_indices[sample_index])
                                        base = {
                                            "method": method,
                                            "causal": method == "causal_historical_warp",
                                            "split": split,
                                            "drive": drive,
                                            "camera": camera,
                                            "sample_index": sample_index,
                                            "previous_raw_frame_index": raw_start,
                                            "origin_raw_frame_index": raw_start + 1,
                                            "target_raw_frame_index": raw_start + horizon + 1,
                                            "previous_frame_name": previous_names[batch_index],
                                            "origin_frame_name": future_names[0][batch_index],
                                            "target_frame_name": future_names[
                                                offsets.index(horizon + 1)
                                            ][batch_index],
                                            "stage": stage,
                                            "stage_channels": int(current.shape[1]),
                                            "stage_height": int(current.shape[2]),
                                            "stage_width": int(current.shape[3]),
                                            "stage_stride_input_pixels": STAGE_STRIDES[stage],
                                            "horizon_frames": horizon,
                                            "horizon_seconds": horizon * fixed_dt_s,
                                            "radius_cells": radius,
                                            "radius_input_pixels": radius * STAGE_STRIDES[stage],
                                            "patch_size": patch_size,
                                        }
                                        rows.append(
                                            _tensor_metrics_to_row(base, metrics, batch_index)
                                        )
                    dataset_index += batch_count
    return rows, datasets


def _parse_drives(value, name):
    drives = tuple(item.strip() for item in value.split(",") if item.strip())
    if not drives:
        raise ValueError(f"{name} must contain at least one drive.")
    return drives


def main():
    output_dir = Path(os.environ["PREDIFY_LOCAL_MOTION_OUTPUT_DIR"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    git_revision = os.environ["PREDIFY_GIT_REVISION"]
    root = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
    batch_size = int(os.environ.get("PREDIFY_LOCAL_MOTION_BATCH_SIZE", "4"))
    train_drives = _parse_drives(os.environ["PREDIFY_TRAIN_DRIVES"], "PREDIFY_TRAIN_DRIVES")
    val_drives = _parse_drives(os.environ["PREDIFY_VAL_DRIVES"], "PREDIFY_VAL_DRIVES")
    drives_by_split = (("train", train_drives), ("val", val_drives))

    model = build_vgg_stage_extractor()
    rows = []
    dataset_records = []
    phases = (
        ("local_future_matching", ("stage5",)),
        ("local_future_matching", ("stage4",)),
        ("causal_historical_warp", STAGE_ORDER),
    )
    for method, stages in phases:
        phase_rows, phase_datasets = evaluate_phase(
            model,
            root=root,
            drives_by_split=drives_by_split,
            camera=camera,
            fixed_dt_s=fixed_dt_s,
            tolerance=tolerance,
            batch_size=batch_size,
            method=method,
            stages=stages,
        )
        rows.extend(phase_rows)
        dataset_records.extend(phase_datasets)

    csv_path = output_dir / "per_frame.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    matrix = summarize_rows(rows)
    causal_gate = build_causal_gate(matrix)
    result = {
        "experiment": "kitti_vgg_local_matching_and_causal_motion",
        "git_revision": git_revision,
        "device": str(DEVICE),
        "backbone": "torchvision VGG16 IMAGENET1K_V1",
        "preprocessing": "Resize(224), CenterCrop(224), ImageNet normalization",
        "camera": camera,
        "fixed_dt_s": fixed_dt_s,
        "fixed_dt_tolerance_s": tolerance,
        "execution_order": [
            "local_future_matching:stage5:h1,h3",
            "local_future_matching:stage4:h1,h3",
            "causal_historical_warp:stage5,stage4:h1",
        ],
        "radii_cells": RADII,
        "patch_sizes": PATCH_SIZES,
        "definitions": {
            "local_future_matching": (
                "At every target location, future target descriptors select the best "
                "nearby F_t descriptor; selected source-center features form the result."
            ),
            "local_future_matching_status": "noncausal oracle diagnostic only",
            "causal_motion_estimation": (
                "F_t descriptors match nearby F_(t-1) descriptors to estimate a "
                "source-to-current discrete displacement at each current location."
            ),
            "causal_warp": (
                "Forward-splat F_t with the historical displacement; average collisions "
                "and use Copy-current at holes."
            ),
            "patch_matching": "zero-padded 1x1 or 3x3 feature descriptors",
            "candidate_validity": "candidate centers outside the feature map are excluded",
            "gain_fraction": "(MSE_copy-MSE_result)/MSE_copy using aggregate means",
        },
        "stages": {
            stage: {
                "stride_input_pixels": STAGE_STRIDES[stage],
                "radius_input_pixels": {
                    str(radius): radius * STAGE_STRIDES[stage] for radius in RADII
                },
            }
            for stage in STAGE_ORDER
        },
        "fair_comparison_policy": "all methods use origins valid from t-1 through t+3",
        "datasets": dataset_records,
        "per_frame_csv": "per_frame.csv",
        "matrix": matrix,
        "causal_copy_gate": causal_gate,
        "predictor_change_allowed": causal_gate["passed"],
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
                "matrix_entries": len(matrix),
                "causal_copy_gate": causal_gate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
