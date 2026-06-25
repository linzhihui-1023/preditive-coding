import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from predify2021.model_factory.get_model import get_model

from .kitti_pairs import (
    KITTIEgoMotionMultiHorizonDataset,
    build_kitti_multi_horizon_dataloader,
    collect_time_filter_stats,
)


KITTI_ROOT = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
KITTI_DRIVE = os.environ.get("PREDIFY_KITTI_DRIVE", "2011_09_26/2011_09_26_drive_0005_sync")
KITTI_CAMERA = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
MAX_PAIRS = int(os.environ.get("PREDIFY_MAX_PAIRS", "0"))
BATCH_SIZE = int(os.environ.get("PREDIFY_BATCHSIZE", "1"))
NUM_WORKERS = int(os.environ.get("PREDIFY_NUM_WORKERS", "0"))
TARGET_FLOW_MODE = os.environ.get("PREDIFY_TARGET_FLOW_MODE", "quasi_steady")
TEMPORAL_TARGET_MODE = os.environ.get("PREDIFY_TEMPORAL_TARGET_MODE", "next_top")
TASK_ALIGNED_TARGET = os.environ.get("PREDIFY_TASK_ALIGNED_TARGET", "").strip()
USE_DYNAMIC_ERROR = os.environ.get("PREDIFY_DYNAMIC_ERROR", "1") == "1"
ERROR_TS_RAW = os.environ.get("PREDIFY_ERROR_TS", "").strip()
ERROR_TAU_RAW = os.environ.get("PREDIFY_ERROR_TAU", "1.0").strip()
ERROR_GAIN_RAW = os.environ.get("PREDIFY_ERROR_GAIN", "1.0").strip()
TEMPORAL_HORIZONS = tuple(
    int(value)
    for value in os.environ.get("PREDIFY_TEMPORAL_HORIZONS", "1").split(",")
    if value.strip()
)
USE_PRETRAINED = os.environ.get("PREDIFY_PRETRAINED", "0") == "1"
COMPUTE_LOCAL_PARAM_GRADS = os.environ.get("PREDIFY_COMPUTE_LOCAL_PARAM_GRADS", "1") == "1"
PCODER_WEIGHTS = os.environ.get("PREDIFY_PCODER_WEIGHTS")
FIXED_TS_RAW = os.environ.get("PREDIFY_FIXED_TS_S", "0.1035").strip()
FIXED_TS_TOL_S = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_optional_float(raw_value):
    if raw_value == "" or raw_value.lower() == "none":
        return None
    return float(raw_value)


FIXED_TS_S = parse_optional_float(FIXED_TS_RAW)


def parse_float_or_float_list(raw_value):
    values = [value.strip() for value in str(raw_value).split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    if len(values) == 1:
        return float(values[0])
    return tuple(float(value) for value in values)


ERROR_SAMPLE_TIME = parse_optional_float(ERROR_TS_RAW)
if ERROR_SAMPLE_TIME is None:
    ERROR_SAMPLE_TIME = FIXED_TS_S if FIXED_TS_S is not None else 1.0
ERROR_TIME_CONSTANT = parse_float_or_float_list(ERROR_TAU_RAW)
ERROR_GAIN = parse_float_or_float_list(ERROR_GAIN_RAW)


def build_model():
    model = get_model(
        "pvgg_tf",
        pretrained=USE_PRETRAINED,
        pcoder_weights=PCODER_WEIGHTS,
        target_flow_mode=TARGET_FLOW_MODE,
        compute_local_param_grads=COMPUTE_LOCAL_PARAM_GRADS,
        temporal_target_mode=TEMPORAL_TARGET_MODE,
        temporal_horizons=TEMPORAL_HORIZONS,
        dynamic_error=USE_DYNAMIC_ERROR,
        error_sample_time=ERROR_SAMPLE_TIME,
        error_time_constant=ERROR_TIME_CONSTANT,
        error_gain=ERROR_GAIN,
    )
    return model.to(device)


def evaluate_pair_smoke(net, dataloader):
    sample_count = 0
    top_local_loss_sum = 0.0
    top_grad_rms_sum = 0.0
    top_error_rms_sum = 0.0
    top_instant_error_rms_sum = 0.0
    total_local_loss_sum = 0.0
    temporal_loss_sum = 0.0
    temporal_mae_sum = 0.0
    temporal_cosine_sum = 0.0

    for batch in tqdm(
        dataloader,
        desc="kitti_adjacent_pair_smoke",
    ):
        if TASK_ALIGNED_TARGET == "ego_motion":
            current_frames, future_frames, temporal_targets, current_names, future_names = batch
            temporal_targets = temporal_targets.to(device, non_blocking=device.type == "cuda")
        else:
            current_frames, future_frames, current_names, future_names = batch
            temporal_targets = None
        batch_pairs = current_frames.size(0)
        current_frames = current_frames.to(device, non_blocking=device.type == "cuda")
        future_frames = future_frames.to(device, non_blocking=device.type == "cuda")
        next_frames = future_frames[:, 0]

        outputs = net.forward(
            current_frames,
            next_x=next_frames,
            future_x=future_frames,
            temporal_target_override=temporal_targets,
        )
        per_layer_losses, total_local_loss = net.collect_learn_flow_losses()
        top_state = net.layer_states[-1]

        top_local_loss = float(top_state.local_loss.detach().cpu().item())
        top_error_rms = float(
            torch.sqrt(torch.mean(top_state.error.detach().float().pow(2))).cpu().item()
        )
        top_instant_error_rms = float(
            torch.sqrt(torch.mean(top_state.instant_error.detach().float().pow(2))).cpu().item()
        )
        temporal_loss = net.collect_temporal_prediction_loss()
        if temporal_loss is None:
            temporal_loss = top_state.local_loss.new_zeros(())

        if net.temporal_prediction is not None and net.temporal_target is not None:
            temporal_mae = torch.mean(
                torch.abs(net.temporal_prediction.detach() - net.temporal_target.detach())
            )
            temporal_cosine = F.cosine_similarity(
                net.temporal_prediction.detach().float(),
                net.temporal_target.detach().float(),
                dim=-1,
            ).mean()
        else:
            temporal_mae = top_state.local_loss.new_zeros(())
            temporal_cosine = top_state.local_loss.new_zeros(())

        grad_stats = top_state.parameter_grad_stats or {}
        top_grad_rms = float(grad_stats.get("rms", 0.0))
        total_local_loss_value = float(total_local_loss.detach().cpu().item())

        sample_count += batch_pairs
        top_local_loss_sum += top_local_loss * batch_pairs
        top_error_rms_sum += top_error_rms * batch_pairs
        top_instant_error_rms_sum += top_instant_error_rms * batch_pairs
        top_grad_rms_sum += top_grad_rms * batch_pairs
        total_local_loss_sum += total_local_loss_value * batch_pairs
        temporal_loss_sum += float(temporal_loss.detach().cpu().item()) * batch_pairs
        temporal_mae_sum += float(temporal_mae.detach().cpu().item()) * batch_pairs
        temporal_cosine_sum += float(temporal_cosine.detach().cpu().item()) * batch_pairs

    return {
        "pairs": sample_count,
        "mean_top_local_loss": (top_local_loss_sum / sample_count) if sample_count else 0.0,
        "mean_top_error_rms": (top_error_rms_sum / sample_count) if sample_count else 0.0,
        "mean_top_instant_error_rms": (top_instant_error_rms_sum / sample_count) if sample_count else 0.0,
        "mean_top_grad_rms": (top_grad_rms_sum / sample_count) if sample_count else 0.0,
        "mean_total_local_loss": (total_local_loss_sum / sample_count) if sample_count else 0.0,
        "mean_temporal_loss": (temporal_loss_sum / sample_count) if sample_count else 0.0,
        "mean_temporal_mae": (temporal_mae_sum / sample_count) if sample_count else 0.0,
        "mean_temporal_cosine": (temporal_cosine_sum / sample_count) if sample_count else 0.0,
    }


def main():
    if TASK_ALIGNED_TARGET not in {"", "ego_motion"}:
        raise ValueError("PREDIFY_TASK_ALIGNED_TARGET must be empty or 'ego_motion'.")
    if TASK_ALIGNED_TARGET == "ego_motion" and TEMPORAL_TARGET_MODE != "ego_motion":
        raise ValueError(
            "When PREDIFY_TASK_ALIGNED_TARGET=ego_motion, set PREDIFY_TEMPORAL_TARGET_MODE=ego_motion."
        )

    if TASK_ALIGNED_TARGET == "ego_motion":
        dataset = KITTIEgoMotionMultiHorizonDataset(
            KITTI_ROOT,
            KITTI_DRIVE,
            KITTI_CAMERA,
            horizons=TEMPORAL_HORIZONS,
            fixed_dt_s=FIXED_TS_S,
            dt_tolerance_s=FIXED_TS_TOL_S,
        )
        if MAX_PAIRS > 0 and MAX_PAIRS < len(dataset):
            from torch.utils.data import Subset, DataLoader

            dataset = Subset(dataset, list(range(MAX_PAIRS)))
            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=device.type == "cuda",
            )
        else:
            from torch.utils.data import DataLoader

            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=device.type == "cuda",
            )
    else:
        dataloader = build_kitti_multi_horizon_dataloader(
            root=KITTI_ROOT,
            drive=KITTI_DRIVE,
            camera=KITTI_CAMERA,
            horizons=TEMPORAL_HORIZONS,
            fixed_dt_s=FIXED_TS_S,
            dt_tolerance_s=FIXED_TS_TOL_S,
            max_pairs=MAX_PAIRS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )

    pair_count = len(dataloader.dataset)
    time_filter_stats = collect_time_filter_stats(dataloader.dataset)
    sequence_dir = Path(KITTI_ROOT) / KITTI_DRIVE / KITTI_CAMERA / "data"
    print(
        f"Using {pair_count} KITTI temporal samples from {sequence_dir} with horizons={TEMPORAL_HORIZONS}, "
        f"fixed_ts_s={FIXED_TS_S}, fixed_ts_tol_s={FIXED_TS_TOL_S}",
        flush=True,
    )
    if time_filter_stats:
        print(f"Time filter stats: {time_filter_stats}", flush=True)
    print(
        f"Starting pair smoke with model=pvgg_tf, pretrained={USE_PRETRAINED}, "
        f"target_flow_mode={TARGET_FLOW_MODE}, batchsize={BATCH_SIZE}, "
        f"temporal_target_mode={TEMPORAL_TARGET_MODE}, "
        f"temporal_horizons={TEMPORAL_HORIZONS}, "
        f"task_aligned_target={TASK_ALIGNED_TARGET or 'none'}, "
        f"compute_local_param_grads={COMPUTE_LOCAL_PARAM_GRADS}, device={device}, "
        f"dynamic_error={USE_DYNAMIC_ERROR}, error_sample_time={ERROR_SAMPLE_TIME}, "
        f"error_time_constant={ERROR_TIME_CONSTANT}, error_gain={ERROR_GAIN}",
        flush=True,
    )

    net = build_model()
    net.train()

    start = datetime.now()
    print(f"STARTING AT : {start}", flush=True)
    results = evaluate_pair_smoke(net, dataloader)
    end = datetime.now()

    print(f"Pairs: {results['pairs']}", flush=True)
    print(f"Mean top local loss: {results['mean_top_local_loss']:.6f}", flush=True)
    print(f"Mean top error rms: {results['mean_top_error_rms']:.6f}", flush=True)
    print(f"Mean top instant error rms: {results['mean_top_instant_error_rms']:.6f}", flush=True)
    print(f"Mean top grad rms: {results['mean_top_grad_rms']:.6f}", flush=True)
    print(f"Mean total local loss: {results['mean_total_local_loss']:.6f}", flush=True)
    print(f"Mean temporal loss: {results['mean_temporal_loss']:.6f}", flush=True)
    print(f"Mean temporal mae: {results['mean_temporal_mae']:.6f}", flush=True)
    print(f"Mean temporal cosine: {results['mean_temporal_cosine']:.6f}", flush=True)
    print(f"TOTAL TIME TAKEN : {end-start}", flush=True)


if __name__ == "__main__":
    main()
