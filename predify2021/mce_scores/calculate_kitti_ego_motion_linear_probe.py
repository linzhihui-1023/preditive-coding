import os
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from predify2021.model_factory.get_model import get_model

from .kitti_pairs import build_kitti_ego_motion_pair_dataset, collect_time_filter_stats


KITTI_ROOT = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
TRAIN_DRIVES = tuple(
    value.strip()
    for value in os.environ.get("PREDIFY_TRAIN_DRIVES", "2011_09_26/2011_09_26_drive_0005_sync").split(",")
    if value.strip()
)
VAL_DRIVES = tuple(
    value.strip()
    for value in os.environ.get("PREDIFY_VAL_DRIVES", "2011_09_26/2011_09_26_drive_0011_sync").split(",")
    if value.strip()
)
KITTI_CAMERA = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
MAX_TRAIN_PAIRS = int(os.environ.get("PREDIFY_MAX_TRAIN_PAIRS", "0"))
MAX_VAL_PAIRS = int(os.environ.get("PREDIFY_MAX_VAL_PAIRS", "0"))
BATCH_SIZE = int(os.environ.get("PREDIFY_BATCHSIZE", "16"))
NUM_WORKERS = int(os.environ.get("PREDIFY_NUM_WORKERS", "4"))
RIDGE_LAMBDA = float(os.environ.get("PREDIFY_PROBE_RIDGE", "1e-3"))
USE_PRETRAINED = os.environ.get("PREDIFY_PRETRAINED", "1") == "1"
PCODER_WEIGHTS = os.environ.get("PREDIFY_PCODER_WEIGHTS", "/home/lin/predify/weights_pvgg16_imagenet")
MODEL_CHECKPOINT = os.environ.get("PREDIFY_MODEL_CHECKPOINT", "")
OUTPUT_PATH = os.environ.get("PREDIFY_OUTPUT_PATH", "kitti_ego_motion_linear_probe.p")
MODEL_LABEL = os.environ.get("PREDIFY_MODEL_LABEL", "pvgg_tf")
TEMPORAL_TARGET_MODE_OVERRIDE = os.environ.get("PREDIFY_TEMPORAL_TARGET_MODE", "").strip()
TEMPORAL_HORIZONS_OVERRIDE = os.environ.get("PREDIFY_TEMPORAL_HORIZONS", "").strip()
FIXED_TS_RAW = os.environ.get("PREDIFY_FIXED_TS_S", "0.1035").strip()
FIXED_TS_TOL_S = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_optional_float(raw_value):
    if raw_value == "" or raw_value.lower() == "none":
        return None
    return float(raw_value)


FIXED_TS_S = parse_optional_float(FIXED_TS_RAW)


def parse_temporal_horizons(raw_value):
    values = tuple(int(value) for value in raw_value.split(",") if value.strip())
    if not values:
        raise ValueError("Temporal horizons must contain at least one positive integer.")
    return values


def build_dataloader(drives, max_pairs):
    dataset = build_kitti_ego_motion_pair_dataset(
        KITTI_ROOT,
        drives,
        camera=KITTI_CAMERA,
        fixed_dt_s=FIXED_TS_S,
        dt_tolerance_s=FIXED_TS_TOL_S,
        max_pairs=max_pairs,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )


def build_model():
    checkpoint = None
    if MODEL_CHECKPOINT:
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location="cpu")

    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    temporal_target_mode = TEMPORAL_TARGET_MODE_OVERRIDE or checkpoint_config.get(
        "temporal_target_mode", "next_top"
    )
    temporal_horizons = (
        parse_temporal_horizons(TEMPORAL_HORIZONS_OVERRIDE)
        if TEMPORAL_HORIZONS_OVERRIDE
        else tuple(checkpoint_config.get("temporal_horizons", (1,)))
    )

    model = get_model(
        "pvgg_tf",
        pretrained=USE_PRETRAINED,
        pcoder_weights=PCODER_WEIGHTS if USE_PRETRAINED else None,
        target_flow_mode="quasi_steady",
        compute_local_param_grads=False,
        temporal_target_mode=temporal_target_mode,
        temporal_horizons=temporal_horizons,
    ).to(device)

    if checkpoint is not None:
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded model checkpoint from {MODEL_CHECKPOINT}", flush=True)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, temporal_target_mode, temporal_horizons


def build_pair_features(model, dataloader, desc):
    feature_batches = []
    target_batches = []

    with torch.no_grad():
        for current_frames, next_frames, targets, current_names, next_names in tqdm(dataloader, desc=desc):
            current_frames = current_frames.to(device, non_blocking=device.type == "cuda")
            next_frames = next_frames.to(device, non_blocking=device.type == "cuda")

            current_top = model.extract_top_forward_feature(current_frames, detach=True)
            next_top = model.extract_top_forward_feature(next_frames, detach=True)

            current_pooled = current_top.mean(dim=(-1, -2))
            next_pooled = next_top.mean(dim=(-1, -2))
            delta_feature = next_pooled - current_pooled

            feature_batches.append(delta_feature.detach().cpu().double())
            target_batches.append(targets.detach().cpu().double())

    features = torch.cat(feature_batches, dim=0)
    targets = torch.cat(target_batches, dim=0)
    return features, targets


def standardize_train_tensor(tensor):
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    standardized = (tensor - mean) / std
    return standardized, mean, std


def apply_standardization(tensor, mean, std):
    return (tensor - mean) / std


def fit_ridge_regression(train_features, train_targets, ridge_lambda):
    feature_dim = train_features.shape[1]
    xtx = train_features.T @ train_features
    ridge = ridge_lambda * torch.eye(feature_dim, dtype=train_features.dtype)
    xty = train_features.T @ train_targets
    weights = torch.linalg.solve(xtx + ridge, xty)
    return weights


def compute_regression_metrics(predictions, targets):
    errors = predictions - targets
    mae = torch.mean(torch.abs(errors), dim=0)
    rmse = torch.sqrt(torch.mean(errors.pow(2), dim=0))
    return {
        "mae_delta_forward_m": float(mae[0].item()),
        "mae_delta_yaw_rad": float(mae[1].item()),
        "rmse_delta_forward_m": float(rmse[0].item()),
        "rmse_delta_yaw_rad": float(rmse[1].item()),
    }


def feature_stats(features):
    return {
        "mean_abs": float(torch.mean(torch.abs(features)).item()),
        "std_mean": float(torch.mean(torch.std(features, dim=0, unbiased=False)).item()),
        "std_global": float(torch.std(features, unbiased=False).item()),
    }


def main():
    train_loader = build_dataloader(TRAIN_DRIVES, MAX_TRAIN_PAIRS)
    val_loader = build_dataloader(VAL_DRIVES, MAX_VAL_PAIRS)
    train_time_filter_stats = collect_time_filter_stats(train_loader.dataset)
    val_time_filter_stats = collect_time_filter_stats(val_loader.dataset)

    model, temporal_target_mode, temporal_horizons = build_model()
    print(
        f"Starting ego-motion linear probe with model_label={MODEL_LABEL}, "
        f"train_drives={TRAIN_DRIVES}, val_drives={VAL_DRIVES}, batchsize={BATCH_SIZE}, "
        f"ridge_lambda={RIDGE_LAMBDA}, device={device}, "
        f"temporal_target_mode={temporal_target_mode}, temporal_horizons={temporal_horizons}, "
        f"fixed_ts_s={FIXED_TS_S}, fixed_ts_tol_s={FIXED_TS_TOL_S}",
        flush=True,
    )
    if train_time_filter_stats:
        print(f"Train time filter stats: {train_time_filter_stats}", flush=True)
    if val_time_filter_stats:
        print(f"Val time filter stats: {val_time_filter_stats}", flush=True)

    train_features, train_targets = build_pair_features(model, train_loader, "probe_train_features")
    val_features, val_targets = build_pair_features(model, val_loader, "probe_val_features")

    train_features_std, feature_mean, feature_std = standardize_train_tensor(train_features)
    train_targets_std, target_mean, target_std = standardize_train_tensor(train_targets)
    val_features_std = apply_standardization(val_features, feature_mean, feature_std)

    weights = fit_ridge_regression(train_features_std, train_targets_std, RIDGE_LAMBDA)

    train_predictions_std = train_features_std @ weights
    val_predictions_std = val_features_std @ weights

    train_predictions = train_predictions_std * target_std + target_mean
    val_predictions = val_predictions_std * target_std + target_mean

    results = {
        "config": {
            "kitti_root": KITTI_ROOT,
            "train_drives": TRAIN_DRIVES,
            "val_drives": VAL_DRIVES,
            "kitti_camera": KITTI_CAMERA,
            "max_train_pairs": MAX_TRAIN_PAIRS,
            "max_val_pairs": MAX_VAL_PAIRS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "ridge_lambda": RIDGE_LAMBDA,
            "pretrained": USE_PRETRAINED,
            "pcoder_weights": PCODER_WEIGHTS,
            "model_checkpoint": MODEL_CHECKPOINT,
            "model_label": MODEL_LABEL,
            "temporal_target_mode": temporal_target_mode,
            "temporal_horizons": temporal_horizons,
            "fixed_ts_s": FIXED_TS_S,
            "fixed_ts_tol_s": FIXED_TS_TOL_S,
            "train_time_filter_stats": train_time_filter_stats,
            "val_time_filter_stats": val_time_filter_stats,
            "feature_definition": "mean(stage5_next - stage5_current)",
            "target_definition": ("delta_forward_m", "delta_yaw_rad"),
        },
        "train_pairs": train_features.shape[0],
        "val_pairs": val_features.shape[0],
        "feature_dim": train_features.shape[1],
        "train_feature_stats": feature_stats(train_features),
        "val_feature_stats": feature_stats(val_features),
        "train_metrics": compute_regression_metrics(train_predictions, train_targets),
        "val_metrics": compute_regression_metrics(val_predictions, val_targets),
    }

    print(f"Train pairs: {results['train_pairs']}", flush=True)
    print(f"Val pairs: {results['val_pairs']}", flush=True)
    print(f"Feature dim: {results['feature_dim']}", flush=True)
    print(f"Train metrics: {results['train_metrics']}", flush=True)
    print(f"Val metrics: {results['val_metrics']}", flush=True)
    print(f"Train feature stats: {results['train_feature_stats']}", flush=True)
    print(f"Val feature stats: {results['val_feature_stats']}", flush=True)

    with open(OUTPUT_PATH, "wb") as handle:
        pickle.dump(results, handle)
    print(f"Saved probe results to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
