import os
import pickle
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from .kitti_pairs import (
    ShuffledFuturePairDataset,
    build_kitti_ego_motion_pair_dataset,
    collect_time_filter_stats,
)


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
EPOCHS = int(os.environ.get("PREDIFY_EPOCHS", "10"))
LEARNING_RATE = float(os.environ.get("PREDIFY_LR", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("PREDIFY_WEIGHT_DECAY", "0.0"))
USE_PRETRAINED = os.environ.get("PREDIFY_PRETRAINED", "1") == "1"
FREEZE_BACKBONE = os.environ.get("PREDIFY_FREEZE_BACKBONE", "1") == "1"
OUTPUT_PATH = os.environ.get("PREDIFY_OUTPUT_PATH", "kitti_vgg_motion_baseline_train.p")
SAVE_MODEL_PATH = os.environ.get("PREDIFY_SAVE_MODEL_PATH", "")
FIXED_TS_RAW = os.environ.get("PREDIFY_FIXED_TS_S", "0.1035").strip()
FIXED_TS_TOL_S = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
SHUFFLE_TRAIN_PAIRS = os.environ.get("PREDIFY_SHUFFLE_TRAIN_PAIRS", "0") == "1"
SHUFFLE_VAL_PAIRS = os.environ.get("PREDIFY_SHUFFLE_VAL_PAIRS", "0") == "1"
SHUFFLE_SEED = int(os.environ.get("PREDIFY_SHUFFLE_SEED", "0"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_optional_float(raw_value):
    if raw_value == "" or raw_value.lower() == "none":
        return None
    return float(raw_value)


FIXED_TS_S = parse_optional_float(FIXED_TS_RAW)


def resolve_save_model_path(output_path):
    if SAVE_MODEL_PATH:
        return SAVE_MODEL_PATH
    path = Path(output_path)
    base_name = path.stem if path.suffix else path.name
    return str(path.with_name(f"{base_name}_model.pt"))


class VGGMotionBaseline(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()
        if pretrained:
            try:
                weights = models.VGG16_Weights.IMAGENET1K_V1
                vgg = models.vgg16(weights=weights)
            except AttributeError:
                vgg = models.vgg16(pretrained=True)
        else:
            try:
                vgg = models.vgg16(weights=None)
            except TypeError:
                vgg = models.vgg16(pretrained=False)

        self.backbone = vgg.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    def extract_features(self, x):
        features = self.backbone(x)
        pooled = self.pool(features).flatten(1)
        return pooled

    def forward(self, x):
        features = self.extract_features(x)
        return self.regressor(features)


def build_dataloader(drives, max_pairs, shuffle):
    dataset = build_kitti_ego_motion_pair_dataset(
        KITTI_ROOT,
        drives,
        camera=KITTI_CAMERA,
        fixed_dt_s=FIXED_TS_S,
        dt_tolerance_s=FIXED_TS_TOL_S,
        max_pairs=max_pairs,
    )
    if shuffle and SHUFFLE_TRAIN_PAIRS:
        dataset = ShuffledFuturePairDataset(dataset, seed=SHUFFLE_SEED)
    elif (not shuffle) and SHUFFLE_VAL_PAIRS:
        dataset = ShuffledFuturePairDataset(dataset, seed=SHUFFLE_SEED + 1)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    return loader


def collect_target_stats(dataset):
    targets = []
    for sample in dataset:
        targets.append(sample[2].float())
    targets = torch.stack(targets, dim=0)
    mean = targets.mean(dim=0)
    std = targets.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return mean, std


def compute_feature_stats(feature_batches):
    features = torch.cat(feature_batches, dim=0)
    return {
        "mean_abs": float(torch.mean(torch.abs(features)).item()),
        "std_mean": float(torch.mean(torch.std(features, dim=0, unbiased=False)).item()),
        "std_global": float(torch.std(features, unbiased=False).item()),
    }


def compute_metrics(predictions, targets):
    errors = predictions - targets
    mae = torch.mean(torch.abs(errors), dim=0)
    rmse = torch.sqrt(torch.mean(errors.pow(2), dim=0))
    return {
        "mae_delta_forward_m": float(mae[0].item()),
        "mae_delta_yaw_rad": float(mae[1].item()),
        "rmse_delta_forward_m": float(rmse[0].item()),
        "rmse_delta_yaw_rad": float(rmse[1].item()),
    }


def run_epoch(model, dataloader, target_mean, target_std, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    if FREEZE_BACKBONE:
        model.backbone.eval()

    loss_fn = nn.MSELoss()
    loss_history = []
    prediction_batches = []
    target_batches = []
    feature_batches = []

    iterator = tqdm(dataloader, desc="train" if is_train else "val")
    for current_frames, next_frames, targets, current_names, next_names in iterator:
        current_frames = current_frames.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        standardized_targets = (targets - target_mean) / target_std

        with torch.set_grad_enabled(is_train):
            features = model.extract_features(current_frames)
            predictions_std = model.regressor(features)
            loss = loss_fn(predictions_std, standardized_targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        predictions = predictions_std.detach() * target_std + target_mean
        loss_history.append(float(loss.detach().cpu().item()))
        prediction_batches.append(predictions.detach().cpu())
        target_batches.append(targets.detach().cpu())
        feature_batches.append(features.detach().cpu())

    predictions = torch.cat(prediction_batches, dim=0)
    targets = torch.cat(target_batches, dim=0)
    feature_stats = compute_feature_stats(feature_batches)
    metrics = compute_metrics(predictions, targets)
    metrics.update(
        {
            "loss": float(sum(loss_history) / len(loss_history)) if loss_history else 0.0,
            "feature_std_global": feature_stats["std_global"],
            "feature_stats": feature_stats,
        }
    )
    return metrics


def main():
    train_loader = build_dataloader(TRAIN_DRIVES, MAX_TRAIN_PAIRS, shuffle=True)
    val_loader = build_dataloader(VAL_DRIVES, MAX_VAL_PAIRS, shuffle=False)
    train_time_filter_stats = collect_time_filter_stats(train_loader.dataset)
    val_time_filter_stats = collect_time_filter_stats(val_loader.dataset)

    target_mean, target_std = collect_target_stats(train_loader.dataset)
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

    print(
        f"Starting VGG motion baseline with pretrained={USE_PRETRAINED}, "
        f"freeze_backbone={FREEZE_BACKBONE}, epochs={EPOCHS}, batchsize={BATCH_SIZE}, "
        f"lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, device={device}, "
        f"fixed_ts_s={FIXED_TS_S}, fixed_ts_tol_s={FIXED_TS_TOL_S}, "
        f"shuffle_train_pairs={SHUFFLE_TRAIN_PAIRS}, shuffle_val_pairs={SHUFFLE_VAL_PAIRS}, "
        f"shuffle_seed={SHUFFLE_SEED}",
        flush=True,
    )
    print(f"Train drives={TRAIN_DRIVES}, Val drives={VAL_DRIVES}", flush=True)
    print(f"Train time filter stats: {train_time_filter_stats}", flush=True)
    print(f"Val time filter stats: {val_time_filter_stats}", flush=True)
    print(
        f"Target normalization: mean={target_mean.detach().cpu().tolist()}, "
        f"std={target_std.detach().cpu().tolist()}",
        flush=True,
    )

    model = VGGMotionBaseline(pretrained=USE_PRETRAINED, freeze_backbone=FREEZE_BACKBONE).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = {
        "config": {
            "kitti_root": KITTI_ROOT,
            "train_drives": TRAIN_DRIVES,
            "val_drives": VAL_DRIVES,
            "kitti_camera": KITTI_CAMERA,
            "max_train_pairs": MAX_TRAIN_PAIRS,
            "max_val_pairs": MAX_VAL_PAIRS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "pretrained": USE_PRETRAINED,
            "freeze_backbone": FREEZE_BACKBONE,
            "fixed_ts_s": FIXED_TS_S,
            "fixed_ts_tol_s": FIXED_TS_TOL_S,
            "shuffle_train_pairs": SHUFFLE_TRAIN_PAIRS,
            "shuffle_val_pairs": SHUFFLE_VAL_PAIRS,
            "shuffle_seed": SHUFFLE_SEED,
            "train_time_filter_stats": train_time_filter_stats,
            "val_time_filter_stats": val_time_filter_stats,
            "target_mean": target_mean.detach().cpu().tolist(),
            "target_std": target_std.detach().cpu().tolist(),
        },
        "epochs": [],
    }

    start = datetime.now()
    print(f"STARTING AT : {start}", flush=True)

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, target_mean, target_std, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, target_mean, target_std, optimizer=None)

        history["epochs"].append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        print(
            f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.6f} | "
            f"train_mae_fwd={train_metrics['mae_delta_forward_m']:.6f} | "
            f"train_mae_yaw={train_metrics['mae_delta_yaw_rad']:.6f} | "
            f"train_feat_std={train_metrics['feature_std_global']:.6f}",
            flush=True,
        )
        print(
            f"Epoch {epoch:03d} | val_loss={val_metrics['loss']:.6f} | "
            f"val_mae_fwd={val_metrics['mae_delta_forward_m']:.6f} | "
            f"val_mae_yaw={val_metrics['mae_delta_yaw_rad']:.6f} | "
            f"val_rmse_fwd={val_metrics['rmse_delta_forward_m']:.6f} | "
            f"val_rmse_yaw={val_metrics['rmse_delta_yaw_rad']:.6f} | "
            f"val_feat_std={val_metrics['feature_std_global']:.6f}",
            flush=True,
        )

    end = datetime.now()
    print(f"TOTAL TIME TAKEN : {end-start}", flush=True)

    with open(OUTPUT_PATH, "wb") as handle:
        pickle.dump(history, handle)
    print(f"Saved training history to {OUTPUT_PATH}", flush=True)

    model_path = resolve_save_model_path(OUTPUT_PATH)
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": history["config"],
    }
    torch.save(checkpoint, model_path)
    print(f"Saved model checkpoint to {model_path}", flush=True)


if __name__ == "__main__":
    main()
