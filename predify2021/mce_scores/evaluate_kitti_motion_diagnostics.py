import json
import os
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from predify2021.model_factory.get_model import get_model

from .kitti_pairs import KITTIEgoMotionMultiHorizonDataset, get_motion_target
from .motion_metrics import compute_motion_diagnostics


MATRIX_ROOT = Path(
    os.environ.get(
        "PREDIFY_MATRIX_ROOT",
        "/home/lin/predify/experiments/seed0_frozen_matrix_6c446d9",
    )
)
OUTPUT_PATH = Path(
    os.environ.get(
        "PREDIFY_OUTPUT_PATH",
        str(MATRIX_ROOT / "seed0_motion_diagnostics.json"),
    )
)
NUM_WORKERS = int(os.environ.get("PREDIFY_NUM_WORKERS", "0"))
GROUPS = tuple(
    value.strip().upper()
    for value in os.environ.get("PREDIFY_GROUPS", "A,B,C,D,E").split(",")
    if value.strip()
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_history(group):
    history_path = MATRIX_ROOT / f"{group}_seed0.p"
    with history_path.open("rb") as handle:
        return pickle.load(handle)


def _motion_stats(config):
    stats = config["motion_target_stats"]
    return (
        torch.tensor(stats["mean"], dtype=torch.float32),
        torch.tensor(stats["std"], dtype=torch.float32),
    )


def _build_sequence_records(config, drives_key):
    records = []
    for drive in config[drives_key]:
        dataset = KITTIEgoMotionMultiHorizonDataset(
            config["kitti_root"],
            drive,
            config["kitti_camera"],
            horizons=tuple(config["temporal_horizons"]),
            fixed_dt_s=config["fixed_ts_s"],
            dt_tolerance_s=config["fixed_ts_tol_s"],
        )
        for segment_index, indices in enumerate(dataset.valid_sample_segments):
            subset = Subset(dataset, list(indices))
            records.append(
                {
                    "name": f"{drive}:segment_{segment_index:04d}",
                    "dataset": subset,
                    "loader": DataLoader(
                        subset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=device.type == "cuda",
                    ),
                }
            )
    return records


def _collect_targets(records):
    return torch.stack(
        [
            get_motion_target(record["dataset"], index).float()
            for record in records
            for index in range(len(record["dataset"]))
        ],
        dim=0,
    )


def _build_model(config, checkpoint_path):
    model = get_model(
        "pvgg_tf",
        pretrained=False,
        target_flow_mode=config["target_flow_mode"],
        compute_local_param_grads=False,
        temporal_target_mode=config["temporal_target_mode"],
        temporal_horizons=tuple(config["temporal_horizons"]),
        dynamic_error=config["dynamic_error"],
        error_state_mode=config["error_state_mode"],
        local_loss_error_source=config["local_loss_error_source"],
        error_sample_time=config["error_sample_time"],
        error_time_constant=config["error_time_constant"],
        error_gain=config["error_gain"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _evaluate_group(group, history, records):
    config = history["config"]
    checkpoint_path = Path(history["best_checkpoint"]["path"])
    model = _build_model(config, checkpoint_path)
    target_mean, target_std = _motion_stats(config)
    target_mean_device = target_mean.to(device)
    target_std_device = target_std.to(device)
    prediction_batches = []
    target_batches = []

    with torch.no_grad():
        for record in records:
            model.reset()
            for batch in tqdm(record["loader"], desc=f"diagnostic_{group}"):
                if config["reset_each_frame"]:
                    model.reset()
                current_frames, future_frames, raw_targets, _, _ = batch
                current_frames = current_frames.to(device, non_blocking=device.type == "cuda")
                future_frames = future_frames.to(device, non_blocking=device.type == "cuda")
                raw_targets = raw_targets.to(device, non_blocking=device.type == "cuda")
                standardized_targets = (raw_targets - target_mean_device) / target_std_device
                top_target = model.extract_top_forward_feature(future_frames[:, 0], detach=True)
                model.step_frame(
                    current_frames,
                    top_target=top_target,
                    temporal_target_override=standardized_targets,
                    duplicate_current_top_context=config["current_top_duplicate"],
                )
                physical_prediction = (
                    model.temporal_prediction * target_std_device + target_mean_device
                )
                prediction_batches.append(physical_prediction.detach().cpu())
                target_batches.append(raw_targets.detach().cpu())

    predictions = torch.cat(prediction_batches, dim=0)
    targets = torch.cat(target_batches, dim=0)
    metrics = compute_motion_diagnostics(
        predictions,
        targets,
        target_mean=target_mean,
        target_std=target_std,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "checkpoint": str(checkpoint_path),
        "selected_epoch": history["best_checkpoint"]["epoch"],
        "selected_validation_mse": history["best_checkpoint"]["value"],
        "metrics": metrics,
    }


def main():
    if not GROUPS:
        raise ValueError("PREDIFY_GROUPS must name at least one matrix group.")
    histories = {group: _load_history(group) for group in GROUPS}
    reference_config = histories[GROUPS[0]]["config"]
    records = _build_sequence_records(reference_config, "val_drives")
    validation_targets = _collect_targets(records)
    target_mean, target_std = _motion_stats(reference_config)
    constant_predictions = target_mean.reshape(1, 1, 2).expand_as(validation_targets)

    results = {
        "matrix_root": str(MATRIX_ROOT),
        "device": str(device),
        "git_revision": reference_config["git_revision"],
        "train_drives": reference_config["train_drives"],
        "val_drives": reference_config["val_drives"],
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "train_mean_constant": compute_motion_diagnostics(
            constant_predictions,
            validation_targets,
            target_mean=target_mean,
            target_std=target_std,
        ),
        "groups": {},
    }
    print(f"Train-mean constant: {results['train_mean_constant']}", flush=True)

    for group in GROUPS:
        group_config = histories[group]["config"]
        for key in (
            "git_revision",
            "kitti_root",
            "train_drives",
            "val_drives",
            "kitti_camera",
            "temporal_horizons",
            "motion_target_stats",
        ):
            if group_config[key] != reference_config[key]:
                raise ValueError(f"Group {group} differs from the reference config at {key}.")
        results["groups"][group] = _evaluate_group(group, histories[group], records)
        print(f"Group {group}: {results['groups'][group]['metrics']}", flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    print(f"Saved motion diagnostics to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
