import copy
import math
import os
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from predify2021.model_factory.get_model import get_model

from .kitti_pairs import (
    KITTIEgoMotionMultiHorizonDataset,
    KITTIMultiHorizonFrameDataset,
    ShuffledFuturePairDataset,
    build_kitti_ego_motion_multi_horizon_dataset,
    build_kitti_multi_horizon_dataset,
    collect_time_filter_stats,
    get_motion_target,
)


KITTI_ROOT = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
KITTI_DRIVE = os.environ.get("PREDIFY_KITTI_DRIVE", "2011_09_26/2011_09_26_drive_0005_sync")
TRAIN_DRIVES_ENV = os.environ.get("PREDIFY_TRAIN_DRIVES", "")
VAL_DRIVES_ENV = os.environ.get("PREDIFY_VAL_DRIVES", "")
KITTI_CAMERA = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
MAX_PAIRS = int(os.environ.get("PREDIFY_MAX_PAIRS", "0"))
MAX_TRAIN_PAIRS = int(os.environ.get("PREDIFY_MAX_TRAIN_PAIRS", str(MAX_PAIRS)))
MAX_VAL_PAIRS = int(os.environ.get("PREDIFY_MAX_VAL_PAIRS", "0"))
BATCH_SIZE = int(os.environ.get("PREDIFY_BATCHSIZE", "1"))
NUM_WORKERS = int(os.environ.get("PREDIFY_NUM_WORKERS", "0"))
EPOCHS = int(os.environ.get("PREDIFY_EPOCHS", "1"))
LEARNING_RATE = float(os.environ.get("PREDIFY_LR", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("PREDIFY_WEIGHT_DECAY", "0.0"))
EMA_DECAY = float(os.environ.get("PREDIFY_EMA_DECAY", "0.99"))
TRAIN_FRACTION = float(os.environ.get("PREDIFY_TRAIN_FRACTION", "0.8"))
TARGET_FLOW_MODE = os.environ.get("PREDIFY_TARGET_FLOW_MODE", "recursive")
TOP_TARGET_SOURCE = os.environ.get("PREDIFY_TOP_TARGET_SOURCE", "ema_teacher")
TEMPORAL_TARGET_MODE = os.environ.get("PREDIFY_TEMPORAL_TARGET_MODE", "next_top")
TASK_ALIGNED_TARGET = os.environ.get("PREDIFY_TASK_ALIGNED_TARGET", "").strip()
USE_DYNAMIC_ERROR = os.environ.get("PREDIFY_DYNAMIC_ERROR", "1") == "1"
ERROR_STATE_MODE = os.environ.get("PREDIFY_ERROR_STATE_MODE", "").strip().lower()
if not ERROR_STATE_MODE:
    ERROR_STATE_MODE = "ema" if USE_DYNAMIC_ERROR else "instant"
LOCAL_LOSS_ERROR_SOURCE = os.environ.get(
    "PREDIFY_LOCAL_LOSS_ERROR_SOURCE",
    "instant",
).strip().lower()
ERROR_TS_RAW = os.environ.get("PREDIFY_ERROR_TS", "").strip()
ERROR_TAU_RAW = os.environ.get("PREDIFY_ERROR_TAU", "1.0").strip()
ERROR_GAIN_RAW = os.environ.get("PREDIFY_ERROR_GAIN", "1.0").strip()
TEMPORAL_HORIZONS = tuple(
    int(value)
    for value in os.environ.get("PREDIFY_TEMPORAL_HORIZONS", "1").split(",")
    if value.strip()
)
USE_PRETRAINED = os.environ.get("PREDIFY_PRETRAINED", "1") == "1"
PCODER_WEIGHTS = os.environ.get("PREDIFY_PCODER_WEIGHTS", "/home/lin/predify/weights_pvgg16_imagenet")
TOP_VARIANCE_WEIGHT = float(os.environ.get("PREDIFY_TOP_VARIANCE_WEIGHT", "0.0"))
TOP_VARIANCE_TARGET = float(os.environ.get("PREDIFY_TOP_VARIANCE_TARGET", "0.01"))
TOP_VARIANCE_EPS = float(os.environ.get("PREDIFY_TOP_VARIANCE_EPS", "1e-6"))
TOP_VARIANCE_WINDOW = int(os.environ.get("PREDIFY_TOP_VARIANCE_WINDOW", "16"))
MOTION_STD_EPS = float(os.environ.get("PREDIFY_MOTION_STD_EPS", "1e-6"))
TEMPORAL_PREDICTION_WEIGHT = float(os.environ.get("PREDIFY_TEMPORAL_PREDICTION_WEIGHT", "1.0"))
LAYER_LOSS_WEIGHTS = tuple(
    float(value)
    for value in os.environ.get("PREDIFY_LAYER_LOSS_WEIGHTS", "0.1,0.1,0.2,0.2,1.0").split(",")
    if value.strip()
)
OUTPUT_PATH = os.environ.get("PREDIFY_OUTPUT_PATH", "kitti_targetflow_adjacent_pairs_train.p")
SAVE_STUDENT_PATH = os.environ.get("PREDIFY_SAVE_STUDENT_PATH", "")
SAVE_TEACHER_PATH = os.environ.get("PREDIFY_SAVE_TEACHER_PATH", "")
SAVE_BEST_STUDENT_PATH = os.environ.get("PREDIFY_SAVE_BEST_STUDENT_PATH", "")
FIXED_TS_RAW = os.environ.get("PREDIFY_FIXED_TS_S", "0.1035").strip()
FIXED_TS_TOL_S = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
SHUFFLE_TRAIN_PAIRS = os.environ.get("PREDIFY_SHUFFLE_TRAIN_PAIRS", "0") == "1"
SHUFFLE_VAL_PAIRS = os.environ.get("PREDIFY_SHUFFLE_VAL_PAIRS", "0") == "1"
SHUFFLE_SEED = int(os.environ.get("PREDIFY_SHUFFLE_SEED", "0"))
STREAM_MODE = os.environ.get("PREDIFY_STREAM_MODE", "1") == "1"
RESET_EACH_FRAME = os.environ.get("PREDIFY_RESET_EACH_FRAME", "0") == "1"
RANDOM_SEED = int(os.environ.get("PREDIFY_SEED", "0"))

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


def parse_drive_list(raw_value, default_drive):
    drives = [value.strip() for value in raw_value.split(",") if value.strip()]
    return drives if drives else [default_drive]


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_data_worker(worker_id):
    worker_seed = (RANDOM_SEED + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _loader_generator():
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    return generator


def _single_stream_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_data_worker,
        generator=_loader_generator(),
    )


def _make_sequence_record(name, dataset):
    return {
        "name": name,
        "dataset": dataset,
        "loader": _single_stream_loader(dataset),
        "time_filter_stats": collect_time_filter_stats(dataset),
    }


def _make_stream_dataset(drive):
    dataset_class = (
        KITTIEgoMotionMultiHorizonDataset if TASK_ALIGNED_TARGET == "ego_motion" else KITTIMultiHorizonFrameDataset
    )
    dataset = dataset_class(
        KITTI_ROOT,
        drive,
        KITTI_CAMERA,
        horizons=TEMPORAL_HORIZONS,
        fixed_dt_s=FIXED_TS_S,
        dt_tolerance_s=FIXED_TS_TOL_S,
    )
    return dataset


def _make_stream_sequence_records(role, drive, max_pairs):
    dataset = _make_stream_dataset(drive)
    remaining = max_pairs if max_pairs > 0 else len(dataset)
    records = []
    for segment_index, sample_indices in enumerate(dataset.valid_sample_segments):
        if remaining <= 0:
            break
        selected_indices = list(sample_indices[:remaining])
        if not selected_indices:
            continue
        records.append(
            _make_sequence_record(
                f"{role}:{drive}:segment_{segment_index:04d}",
                Subset(dataset, selected_indices),
            )
        )
        remaining -= len(selected_indices)
    return records


def _partition_stream_sequence_records(records, train_count, drive):
    train_records = []
    val_records = []
    remaining_train = train_count
    for record_index, record in enumerate(records):
        dataset = record["dataset"]
        take_train = min(len(dataset), remaining_train)
        if take_train > 0:
            train_records.append(
                _make_sequence_record(
                    f"train:{drive}:segment_{record_index:04d}",
                    Subset(dataset, list(range(take_train))),
                )
            )
            remaining_train -= take_train
        if take_train < len(dataset):
            val_records.append(
                _make_sequence_record(
                    f"val:{drive}:segment_{record_index:04d}",
                    Subset(dataset, list(range(take_train, len(dataset)))),
                )
            )
    return train_records, val_records


def _count_sequence_samples(sequences):
    return sum(len(sequence["dataset"]) for sequence in sequences)


def _collect_sequence_time_filter_stats(sequences):
    return {
        sequence["name"]: sequence["time_filter_stats"]
        for sequence in sequences
        if sequence["time_filter_stats"]
    }


def compute_motion_target_stats(training_data):
    datasets = (
        [sequence["dataset"] for sequence in training_data]
        if STREAM_MODE
        else [training_data.dataset]
    )
    targets = [
        get_motion_target(dataset, index).float()
        for dataset in datasets
        for index in range(len(dataset))
    ]
    if not targets:
        raise ValueError("Cannot estimate motion normalization without training targets.")

    stacked = torch.stack(targets, dim=0)
    if stacked.dim() == 2:
        stacked = stacked.unsqueeze(1)
    if stacked.shape[-1] != 2:
        raise ValueError(f"Expected longitudinal-yaw targets with 2 components, got {stacked.shape}.")

    mean = stacked.mean(dim=0)
    raw_std = stacked.std(dim=0, unbiased=False)
    std = raw_std.clamp_min(MOTION_STD_EPS)
    return {
        "name": "longitudinal_yaw_2dof",
        "components": ("forward_displacement_m", "yaw_change_rad"),
        "count": int(stacked.shape[0]),
        "mean": mean,
        "std": std,
        "raw_std": raw_std,
        "std_eps": MOTION_STD_EPS,
    }


def serialize_motion_target_stats(stats):
    if stats is None:
        return None
    return {
        "name": stats["name"],
        "components": stats["components"],
        "count": stats["count"],
        "mean": stats["mean"].tolist(),
        "std": stats["std"].tolist(),
        "raw_std": stats["raw_std"].tolist(),
        "std_eps": stats["std_eps"],
    }


def normalize_motion_targets(targets, stats):
    mean = stats["mean"].to(targets.device, targets.dtype)
    std = stats["std"].to(targets.device, targets.dtype)
    return (targets - mean) / std


def denormalize_motion_targets(targets, stats):
    mean = stats["mean"].to(targets.device, targets.dtype)
    std = stats["std"].to(targets.device, targets.dtype)
    return targets * std + mean


def derive_checkpoint_path(output_path, suffix):
    path = Path(output_path)
    base_name = path.stem if path.suffix else path.name
    return str(path.with_name(f"{base_name}{suffix}"))


def save_model_checkpoint(model, path, config, epoch_record, checkpoint_kind):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "selected_epoch": epoch_record,
            "checkpoint_kind": checkpoint_kind,
        },
        path,
    )


def freeze_teacher(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def build_student_model():
    return get_model(
        "pvgg_tf",
        pretrained=USE_PRETRAINED,
        pcoder_weights=PCODER_WEIGHTS if USE_PRETRAINED else None,
        target_flow_mode=TARGET_FLOW_MODE,
        compute_local_param_grads=False,
        temporal_target_mode=TEMPORAL_TARGET_MODE,
        temporal_horizons=TEMPORAL_HORIZONS,
        dynamic_error=ERROR_STATE_MODE != "instant",
        error_state_mode=ERROR_STATE_MODE,
        local_loss_error_source=LOCAL_LOSS_ERROR_SOURCE,
        error_sample_time=ERROR_SAMPLE_TIME,
        error_time_constant=ERROR_TIME_CONSTANT,
        error_gain=ERROR_GAIN,
    ).to(device)


def build_teacher_model(student):
    if TOP_TARGET_SOURCE != "ema_teacher":
        return None
    teacher = copy.deepcopy(student).to(device)
    freeze_teacher(teacher)
    return teacher


def build_train_val_loaders():
    train_drives = parse_drive_list(TRAIN_DRIVES_ENV, KITTI_DRIVE)
    val_drives = parse_drive_list(VAL_DRIVES_ENV, "") if VAL_DRIVES_ENV.strip() else []

    if STREAM_MODE:
        if BATCH_SIZE != 1:
            raise ValueError(
                "PREDIFY_STREAM_MODE=1 requires PREDIFY_BATCHSIZE=1 so each video frame "
                "inherits exactly one previous frame state."
            )
        if SHUFFLE_TRAIN_PAIRS or SHUFFLE_VAL_PAIRS:
            raise ValueError(
                "PREDIFY_STREAM_MODE=1 keeps true video order; shuffled-pair controls are disabled."
            )

        if val_drives:
            train_sequences = [
                sequence
                for drive in train_drives
                for sequence in _make_stream_sequence_records(
                    "train",
                    drive,
                    MAX_TRAIN_PAIRS,
                )
            ]
            val_sequences = [
                sequence
                for drive in val_drives
                for sequence in _make_stream_sequence_records(
                    "val",
                    drive,
                    MAX_VAL_PAIRS,
                )
            ]
        else:
            all_sequences = _make_stream_sequence_records("all", KITTI_DRIVE, MAX_PAIRS)
            pair_count = _count_sequence_samples(all_sequences)
            train_count = max(1, int(pair_count * TRAIN_FRACTION))
            train_count = min(train_count, pair_count - 1) if pair_count > 1 else 1
            train_sequences, val_sequences = _partition_stream_sequence_records(
                all_sequences,
                train_count,
                KITTI_DRIVE,
            )

        return train_sequences, val_sequences, train_drives, val_drives

    if val_drives:
        if TASK_ALIGNED_TARGET == "ego_motion":
            train_dataset = build_kitti_ego_motion_multi_horizon_dataset(
                KITTI_ROOT,
                train_drives,
                camera=KITTI_CAMERA,
                horizons=TEMPORAL_HORIZONS,
                fixed_dt_s=FIXED_TS_S,
                dt_tolerance_s=FIXED_TS_TOL_S,
                max_pairs=MAX_TRAIN_PAIRS,
            )
            val_dataset = build_kitti_ego_motion_multi_horizon_dataset(
                KITTI_ROOT,
                val_drives,
                camera=KITTI_CAMERA,
                horizons=TEMPORAL_HORIZONS,
                fixed_dt_s=FIXED_TS_S,
                dt_tolerance_s=FIXED_TS_TOL_S,
                max_pairs=MAX_VAL_PAIRS,
            )
        else:
            train_dataset = build_kitti_multi_horizon_dataset(
                KITTI_ROOT,
                train_drives,
                camera=KITTI_CAMERA,
                horizons=TEMPORAL_HORIZONS,
                fixed_dt_s=FIXED_TS_S,
                dt_tolerance_s=FIXED_TS_TOL_S,
                max_pairs=MAX_TRAIN_PAIRS,
            )
            val_dataset = build_kitti_multi_horizon_dataset(
                KITTI_ROOT,
                val_drives,
                camera=KITTI_CAMERA,
                horizons=TEMPORAL_HORIZONS,
                fixed_dt_s=FIXED_TS_S,
                dt_tolerance_s=FIXED_TS_TOL_S,
                max_pairs=MAX_VAL_PAIRS,
            )
    else:
        dataset_class = (
            KITTIEgoMotionMultiHorizonDataset if TASK_ALIGNED_TARGET == "ego_motion" else KITTIMultiHorizonFrameDataset
        )
        dataset = dataset_class(
            KITTI_ROOT,
            KITTI_DRIVE,
            KITTI_CAMERA,
            horizons=TEMPORAL_HORIZONS,
            fixed_dt_s=FIXED_TS_S,
            dt_tolerance_s=FIXED_TS_TOL_S,
        )
        if MAX_PAIRS > 0 and MAX_PAIRS < len(dataset):
            dataset = Subset(dataset, list(range(MAX_PAIRS)))

        pair_count = len(dataset)
        train_count = max(1, int(pair_count * TRAIN_FRACTION))
        train_count = min(train_count, pair_count - 1) if pair_count > 1 else 1
        val_count = pair_count - train_count

        if isinstance(dataset, Subset):
            indices = dataset.indices
            base_dataset = dataset.dataset
        else:
            indices = list(range(pair_count))
            base_dataset = dataset

        train_dataset = Subset(base_dataset, indices[:train_count])
        val_dataset = Subset(base_dataset, indices[train_count:]) if val_count > 0 else None

    train_loader = DataLoader(
        ShuffledFuturePairDataset(train_dataset, seed=SHUFFLE_SEED) if SHUFFLE_TRAIN_PAIRS else train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_data_worker,
        generator=_loader_generator(),
    )

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            ShuffledFuturePairDataset(val_dataset, seed=SHUFFLE_SEED + 1) if SHUFFLE_VAL_PAIRS else val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_data_worker,
            generator=_loader_generator(),
        )

    return train_loader, val_loader, train_drives, val_drives


def build_optimizer(student):
    trainable_parameters = []
    trainable_parameters.extend(student.forward_stages.parameters())
    trainable_parameters.extend(student.feedback_modules.parameters())
    trainable_parameters.extend(student.temporal_predictor.parameters())
    return torch.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def update_ema(student, teacher, decay):
    if teacher is None:
        return
    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
            teacher_parameter.data.mul_(decay).add_(student_parameter.data, alpha=1.0 - decay)


def combine_local_losses(per_layer_losses, layer_weights):
    if len(per_layer_losses) != len(layer_weights):
        raise ValueError(
            f"Expected {len(layer_weights)} local losses, but got {len(per_layer_losses)}."
        )

    weighted_losses = []
    total = None
    for loss, weight in zip(per_layer_losses, layer_weights):
        weighted = loss * weight
        weighted_losses.append(weighted)
        total = weighted if total is None else total + weighted
    return weighted_losses, total


def _feature_std(tensor):
    return float(tensor.detach().float().std().cpu().item())


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _pool_top_features(tensor):
    if tensor.dim() <= 2:
        return tensor
    return tensor.mean(dim=tuple(range(2, tensor.dim())))


def _variance_loss_from_samples(samples):
    if samples.shape[0] < 2:
        zero = samples.sum() * 0.0
        return zero, 0.0

    std_per_dim = torch.sqrt(samples.var(dim=0, unbiased=False) + TOP_VARIANCE_EPS)
    variance_loss = torch.relu(TOP_VARIANCE_TARGET - std_per_dim).mean()
    return variance_loss, float(std_per_dim.detach().mean().cpu().item())


class TemporalFeatureVarianceWindow:
    def __init__(self, window_size):
        if window_size < 2:
            raise ValueError("Temporal variance window must contain at least 2 frames.")
        self.window_size = int(window_size)
        self.history = []

    def reset(self):
        self.history = []

    def compute(self, top_forward_output):
        pooled = _pool_top_features(top_forward_output)
        samples = torch.cat(self.history + [pooled], dim=0) if self.history else pooled
        variance_loss, pooled_std = _variance_loss_from_samples(samples)

        self.history.extend(sample.detach() for sample in pooled.split(1, dim=0))
        max_history = self.window_size - 1
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
        return variance_loss, pooled_std, int(samples.shape[0])


def compute_top_variance_regularizer(top_forward_output, temporal_window=None):
    if temporal_window is not None:
        return temporal_window.compute(top_forward_output)

    pooled = _pool_top_features(top_forward_output)
    variance_loss, pooled_std = _variance_loss_from_samples(pooled)
    return variance_loss, pooled_std, int(pooled.shape[0])


def resolve_top_target(student, teacher, next_frames):
    if TOP_TARGET_SOURCE == "ema_teacher":
        return teacher.extract_top_forward_feature(next_frames, detach=True)
    if TOP_TARGET_SOURCE == "student_self":
        return student.extract_top_forward_feature(next_frames, detach=True)
    raise ValueError(f"Unsupported PREDIFY_TOP_TARGET_SOURCE: {TOP_TARGET_SOURCE}")


def resolve_temporal_targets(student, teacher, future_frames):
    if TOP_TARGET_SOURCE == "ema_teacher":
        return teacher.extract_top_forward_features(future_frames, detach=True)
    if TOP_TARGET_SOURCE == "student_self":
        return student.extract_top_forward_features(future_frames, detach=True)
    raise ValueError(f"Unsupported PREDIFY_TOP_TARGET_SOURCE: {TOP_TARGET_SOURCE}")


def run_epoch(student, teacher, dataloader, optimizer=None, motion_target_stats=None):
    training = optimizer is not None
    if training:
        student.train()
    else:
        student.eval()
    if teacher is not None:
        teacher.eval()

    weighted_loss_history = []
    optimized_loss_history = []
    total_local_loss_history = []
    top_local_loss_history = []
    top_feature_std_history = []
    top_pooled_std_history = []
    top_variance_loss_history = []
    top_variance_sample_count_history = []
    temporal_loss_history = []
    temporal_mae_history = []
    temporal_cosine_history = []
    forward_mae_history = []
    yaw_mae_history = []
    forward_mae_per_horizon_history = [[] for _ in TEMPORAL_HORIZONS]
    yaw_mae_per_horizon_history = [[] for _ in TEMPORAL_HORIZONS]
    per_layer_history = [[] for _ in range(student.number_of_layers)]

    if STREAM_MODE:
        sequence_records = dataloader
    else:
        sequence_records = [{"name": "legacy_pair_batches", "loader": dataloader}]
    total_batches = sum(len(sequence["loader"]) for sequence in sequence_records)

    iterator = tqdm(total=total_batches, desc="train" if training else "val")
    for sequence in sequence_records:
        temporal_variance_window = (
            TemporalFeatureVarianceWindow(TOP_VARIANCE_WINDOW) if STREAM_MODE else None
        )
        if STREAM_MODE:
            student.reset()
            if teacher is not None:
                teacher.reset()

        for batch in sequence["loader"]:
            if STREAM_MODE and RESET_EACH_FRAME:
                student.reset()
                if teacher is not None:
                    teacher.reset()

            if TASK_ALIGNED_TARGET == "ego_motion":
                if motion_target_stats is None:
                    raise ValueError("2-DoF longitudinal-yaw training requires training-set normalization stats.")
                current_frames, future_frames, raw_temporal_targets, current_names, future_names = batch
                raw_temporal_targets = raw_temporal_targets.to(
                    device,
                    non_blocking=device.type == "cuda",
                )
                temporal_targets = normalize_motion_targets(
                    raw_temporal_targets,
                    motion_target_stats,
                )
            else:
                current_frames, future_frames, current_names, future_names = batch
                temporal_targets = None
                raw_temporal_targets = None
            current_frames = current_frames.to(device, non_blocking=device.type == "cuda")
            future_frames = future_frames.to(device, non_blocking=device.type == "cuda")

            next_frames = future_frames[:, 0]
            top_target = resolve_top_target(student, teacher, next_frames)
            temporal_top_targets = None
            temporal_target_override = temporal_targets
            if TASK_ALIGNED_TARGET != "ego_motion":
                temporal_top_targets = resolve_temporal_targets(student, teacher, future_frames)

            model_step = student.step_frame if STREAM_MODE else student.forward
            if training:
                optimizer.zero_grad(set_to_none=True)
                model_step(
                    current_frames,
                    top_target=top_target,
                    temporal_top_targets=temporal_top_targets,
                    temporal_target_override=temporal_target_override,
                )
                per_layer_losses, total_local_loss = student.collect_learn_flow_losses()
                _, weighted_loss = combine_local_losses(per_layer_losses, LAYER_LOSS_WEIGHTS)
                top_variance_loss, top_pooled_std, top_variance_sample_count = (
                    compute_top_variance_regularizer(
                        student.layer_states[-1].forward_output,
                        temporal_window=temporal_variance_window,
                    )
                )
                temporal_loss = student.collect_temporal_prediction_loss()
                if temporal_loss is None:
                    temporal_loss = weighted_loss.new_zeros(())
                optimized_loss = (
                    weighted_loss
                    + TOP_VARIANCE_WEIGHT * top_variance_loss
                    + TEMPORAL_PREDICTION_WEIGHT * temporal_loss
                )
                optimized_loss.backward()
                optimizer.step()
                update_ema(student, teacher, EMA_DECAY)
            else:
                with torch.no_grad():
                    model_step(
                        current_frames,
                        top_target=top_target,
                        temporal_top_targets=temporal_top_targets,
                        temporal_target_override=temporal_target_override,
                    )
                    per_layer_losses, total_local_loss = student.collect_learn_flow_losses()
                    _, weighted_loss = combine_local_losses(per_layer_losses, LAYER_LOSS_WEIGHTS)
                    top_variance_loss, top_pooled_std, top_variance_sample_count = (
                        compute_top_variance_regularizer(
                            student.layer_states[-1].forward_output,
                            temporal_window=temporal_variance_window,
                        )
                    )
                    temporal_loss = student.collect_temporal_prediction_loss()
                    if temporal_loss is None:
                        temporal_loss = weighted_loss.new_zeros(())
                    optimized_loss = (
                        weighted_loss
                        + TOP_VARIANCE_WEIGHT * top_variance_loss
                        + TEMPORAL_PREDICTION_WEIGHT * temporal_loss
                    )

            temporal_prediction = student.temporal_prediction
            temporal_target = student.temporal_target
            if temporal_prediction is not None and temporal_target is not None:
                temporal_mae = torch.mean(torch.abs(temporal_prediction - temporal_target.detach()))
                if TASK_ALIGNED_TARGET == "ego_motion":
                    physical_prediction = denormalize_motion_targets(
                        temporal_prediction,
                        motion_target_stats,
                    )
                    physical_absolute_error = torch.abs(
                        physical_prediction - raw_temporal_targets.detach()
                    )
                    forward_mae = physical_absolute_error[..., 0].mean()
                    yaw_mae = physical_absolute_error[..., 1].mean()
                    temporal_cosine = F.cosine_similarity(
                        physical_prediction.detach().float(),
                        raw_temporal_targets.detach().float(),
                        dim=-1,
                    ).mean()
                    for horizon_index in range(len(TEMPORAL_HORIZONS)):
                        forward_mae_per_horizon_history[horizon_index].append(
                            float(
                                physical_absolute_error[:, horizon_index, 0]
                                .mean()
                                .detach()
                                .cpu()
                                .item()
                            )
                        )
                        yaw_mae_per_horizon_history[horizon_index].append(
                            float(
                                physical_absolute_error[:, horizon_index, 1]
                                .mean()
                                .detach()
                                .cpu()
                                .item()
                            )
                        )
                else:
                    forward_mae = weighted_loss.new_zeros(())
                    yaw_mae = weighted_loss.new_zeros(())
                    temporal_cosine = F.cosine_similarity(
                        temporal_prediction.detach().float(),
                        temporal_target.detach().float(),
                        dim=-1,
                    ).mean()
            else:
                temporal_mae = weighted_loss.new_zeros(())
                temporal_cosine = weighted_loss.new_zeros(())
                forward_mae = weighted_loss.new_zeros(())
                yaw_mae = weighted_loss.new_zeros(())

            weighted_loss_history.append(float(weighted_loss.detach().cpu().item()))
            optimized_loss_history.append(float(optimized_loss.detach().cpu().item()))
            total_local_loss_history.append(float(total_local_loss.detach().cpu().item()))
            top_local_loss_history.append(float(per_layer_losses[-1].detach().cpu().item()))
            top_feature_std_history.append(_feature_std(student.layer_states[-1].forward_output))
            top_pooled_std_history.append(top_pooled_std)
            top_variance_loss_history.append(float(top_variance_loss.detach().cpu().item()))
            top_variance_sample_count_history.append(top_variance_sample_count)
            temporal_loss_history.append(float(temporal_loss.detach().cpu().item()))
            temporal_mae_history.append(float(temporal_mae.detach().cpu().item()))
            temporal_cosine_history.append(float(temporal_cosine.detach().cpu().item()))
            if TASK_ALIGNED_TARGET == "ego_motion":
                forward_mae_history.append(float(forward_mae.detach().cpu().item()))
                yaw_mae_history.append(float(yaw_mae.detach().cpu().item()))

            for layer_idx, loss in enumerate(per_layer_losses):
                per_layer_history[layer_idx].append(float(loss.detach().cpu().item()))

            iterator.update(1)
    iterator.close()

    return {
        "mean_weighted_loss": _mean(weighted_loss_history),
        "mean_optimized_loss": _mean(optimized_loss_history),
        "mean_total_local_loss": _mean(total_local_loss_history),
        "mean_top_local_loss": _mean(top_local_loss_history),
        "mean_top_feature_std": _mean(top_feature_std_history),
        "mean_top_pooled_std": _mean(top_pooled_std_history),
        "mean_top_variance_loss": _mean(top_variance_loss_history),
        "mean_top_variance_sample_count": _mean(top_variance_sample_count_history),
        "max_top_variance_sample_count": max(top_variance_sample_count_history, default=0),
        "mean_temporal_loss": _mean(temporal_loss_history),
        "mean_temporal_mae": _mean(temporal_mae_history),
        "mean_temporal_cosine": _mean(temporal_cosine_history),
        "mean_standardized_motion_mse": (
            _mean(temporal_loss_history) if TASK_ALIGNED_TARGET == "ego_motion" else None
        ),
        "mean_standardized_motion_mae": (
            _mean(temporal_mae_history) if TASK_ALIGNED_TARGET == "ego_motion" else None
        ),
        "mean_forward_displacement_mae_m": _mean(forward_mae_history),
        "mean_yaw_change_mae_rad": _mean(yaw_mae_history),
        "mean_forward_displacement_mae_m_per_horizon": tuple(
            _mean(values) for values in forward_mae_per_horizon_history
        ),
        "mean_yaw_change_mae_rad_per_horizon": tuple(
            _mean(values) for values in yaw_mae_per_horizon_history
        ),
        "mean_per_layer_local_loss": tuple(_mean(layer_values) for layer_values in per_layer_history),
        "num_batches": len(weighted_loss_history),
        "num_sequences": len(sequence_records),
        "stream_mode": STREAM_MODE,
        "reset_each_frame": RESET_EACH_FRAME,
    }


def main():
    seed_everything(RANDOM_SEED)

    if len(LAYER_LOSS_WEIGHTS) != 5:
        raise ValueError(
            f"PREDIFY_LAYER_LOSS_WEIGHTS must contain 5 values for pvgg_tf, got {len(LAYER_LOSS_WEIGHTS)}."
        )
    if TOP_TARGET_SOURCE not in {"ema_teacher", "student_self"}:
        raise ValueError(
            "PREDIFY_TOP_TARGET_SOURCE must be one of: ema_teacher, student_self."
        )
    if TASK_ALIGNED_TARGET not in {"", "ego_motion"}:
        raise ValueError("PREDIFY_TASK_ALIGNED_TARGET must be empty or 'ego_motion'.")
    if TASK_ALIGNED_TARGET == "ego_motion" and TEMPORAL_TARGET_MODE != "ego_motion":
        raise ValueError(
            "When PREDIFY_TASK_ALIGNED_TARGET=ego_motion, set PREDIFY_TEMPORAL_TARGET_MODE=ego_motion."
        )
    if TEMPORAL_PREDICTION_WEIGHT < 0:
        raise ValueError("PREDIFY_TEMPORAL_PREDICTION_WEIGHT must be non-negative.")
    if TEMPORAL_PREDICTION_WEIGHT == 0:
        print(
            "WARNING: temporal_prediction_weight=0 disables training of the temporal predictor.",
            flush=True,
        )
    if RESET_EACH_FRAME and not STREAM_MODE:
        raise ValueError("PREDIFY_RESET_EACH_FRAME=1 requires PREDIFY_STREAM_MODE=1.")
    if TARGET_FLOW_MODE not in {"recursive", "quasi_steady"}:
        raise ValueError("PREDIFY_TARGET_FLOW_MODE must be recursive or quasi_steady.")
    if ERROR_STATE_MODE not in {"instant", "ema", "lag1"}:
        raise ValueError("PREDIFY_ERROR_STATE_MODE must be instant, ema, or lag1.")
    if LOCAL_LOSS_ERROR_SOURCE not in {"instant", "state"}:
        raise ValueError("PREDIFY_LOCAL_LOSS_ERROR_SOURCE must be instant or state.")
    if TOP_VARIANCE_WINDOW < 2:
        raise ValueError("PREDIFY_TOP_VARIANCE_WINDOW must be at least 2.")
    if TOP_VARIANCE_WEIGHT < 0:
        raise ValueError("PREDIFY_TOP_VARIANCE_WEIGHT must be non-negative.")
    if TOP_VARIANCE_TARGET <= 0:
        raise ValueError("PREDIFY_TOP_VARIANCE_TARGET must be positive.")
    if TOP_VARIANCE_EPS <= 0:
        raise ValueError("PREDIFY_TOP_VARIANCE_EPS must be positive.")
    if MOTION_STD_EPS <= 0:
        raise ValueError("PREDIFY_MOTION_STD_EPS must be positive.")
    if TOP_VARIANCE_WEIGHT > 0 and TOP_VARIANCE_TARGET <= math.sqrt(TOP_VARIANCE_EPS):
        raise ValueError(
            "PREDIFY_TOP_VARIANCE_TARGET must be greater than sqrt(PREDIFY_TOP_VARIANCE_EPS) "
            "when temporal variance regularization is enabled."
        )

    train_loader, val_loader, train_drives, val_drives = build_train_val_loaders()
    motion_target_stats = (
        compute_motion_target_stats(train_loader)
        if TASK_ALIGNED_TARGET == "ego_motion"
        else None
    )
    serialized_motion_target_stats = serialize_motion_target_stats(motion_target_stats)
    if STREAM_MODE:
        train_pairs = _count_sequence_samples(train_loader)
        val_pairs = _count_sequence_samples(val_loader) if val_loader is not None else 0
        train_time_filter_stats = _collect_sequence_time_filter_stats(train_loader)
        val_time_filter_stats = _collect_sequence_time_filter_stats(val_loader) if val_loader is not None else {}
    else:
        train_pairs = len(train_loader.dataset)
        val_pairs = len(val_loader.dataset) if val_loader is not None else 0
        train_time_filter_stats = collect_time_filter_stats(train_loader.dataset)
        val_time_filter_stats = collect_time_filter_stats(val_loader.dataset) if val_loader is not None else {}

    print(
        f"Using KITTI temporal samples: train_drives={tuple(train_drives)}, "
        f"val_drives={tuple(val_drives) if val_drives else ('split-from-train',)}, "
        f"train_pairs={train_pairs}, val_pairs={val_pairs}, temporal_horizons={TEMPORAL_HORIZONS}, "
        f"task_aligned_target={TASK_ALIGNED_TARGET or 'none'}, "
        f"fixed_ts_s={FIXED_TS_S}, fixed_ts_tol_s={FIXED_TS_TOL_S}, "
        f"stream_mode={STREAM_MODE}, reset_each_frame={RESET_EACH_FRAME}, seed={RANDOM_SEED}, "
        f"shuffle_train_pairs={SHUFFLE_TRAIN_PAIRS}, shuffle_val_pairs={SHUFFLE_VAL_PAIRS}, "
        f"shuffle_seed={SHUFFLE_SEED}",
        flush=True,
    )
    if train_time_filter_stats:
        print(f"Train time filter stats: {train_time_filter_stats}", flush=True)
    if val_time_filter_stats:
        print(f"Val time filter stats: {val_time_filter_stats}", flush=True)
    if serialized_motion_target_stats is not None:
        print(
            f"Training-only 2-DoF longitudinal-yaw normalization: "
            f"{serialized_motion_target_stats}",
            flush=True,
        )
    print(
        f"Starting target-flow training with pretrained={USE_PRETRAINED}, "
        f"target_flow_mode={TARGET_FLOW_MODE}, epochs={EPOCHS}, batchsize={BATCH_SIZE}, "
        f"lr={LEARNING_RATE}, ema_decay={EMA_DECAY}, top_target_source={TOP_TARGET_SOURCE}, device={device}, "
        f"feedback_decoder_trainable=True, "
        f"layer_loss_weights={LAYER_LOSS_WEIGHTS}, top_variance_weight={TOP_VARIANCE_WEIGHT}, "
        f"top_variance_target={TOP_VARIANCE_TARGET}, top_variance_eps={TOP_VARIANCE_EPS}, "
        f"top_variance_window={TOP_VARIANCE_WINDOW}, "
        f"temporal_prediction_weight={TEMPORAL_PREDICTION_WEIGHT}, "
        f"temporal_target_mode={TEMPORAL_TARGET_MODE}, temporal_horizons={TEMPORAL_HORIZONS}, "
        f"task_aligned_target={TASK_ALIGNED_TARGET or 'none'}, "
        f"error_state_mode={ERROR_STATE_MODE}, local_loss_error_source={LOCAL_LOSS_ERROR_SOURCE}, "
        f"error_sample_time={ERROR_SAMPLE_TIME}, "
        f"error_time_constant={ERROR_TIME_CONSTANT}, error_gain={ERROR_GAIN}",
        flush=True,
    )

    student = build_student_model()
    teacher = build_teacher_model(student)
    optimizer = build_optimizer(student)

    history = {
        "config": {
            "kitti_root": KITTI_ROOT,
            "kitti_drive": KITTI_DRIVE,
            "train_drives": tuple(train_drives),
            "val_drives": tuple(val_drives),
            "kitti_camera": KITTI_CAMERA,
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "fixed_ts_s": FIXED_TS_S,
            "fixed_ts_tol_s": FIXED_TS_TOL_S,
            "stream_mode": STREAM_MODE,
            "reset_each_frame": RESET_EACH_FRAME,
            "seed": RANDOM_SEED,
            "shuffle_train_pairs": SHUFFLE_TRAIN_PAIRS,
            "shuffle_val_pairs": SHUFFLE_VAL_PAIRS,
            "shuffle_seed": SHUFFLE_SEED,
            "train_time_filter_stats": train_time_filter_stats,
            "val_time_filter_stats": val_time_filter_stats,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "ema_decay": EMA_DECAY,
            "top_target_source": TOP_TARGET_SOURCE,
            "feedback_decoder_trainable": True,
            "train_fraction": TRAIN_FRACTION,
            "target_flow_mode": TARGET_FLOW_MODE,
            "temporal_target_mode": TEMPORAL_TARGET_MODE,
            "temporal_horizons": TEMPORAL_HORIZONS,
            "task_aligned_target": TASK_ALIGNED_TARGET or "none",
            "motion_target_name": (
                "longitudinal_yaw_2dof" if TASK_ALIGNED_TARGET == "ego_motion" else "none"
            ),
            "motion_target_stats": serialized_motion_target_stats,
            "dynamic_error": ERROR_STATE_MODE != "instant",
            "error_state_mode": ERROR_STATE_MODE,
            "local_loss_error_source": LOCAL_LOSS_ERROR_SOURCE,
            "error_sample_time": ERROR_SAMPLE_TIME,
            "error_time_constant": ERROR_TIME_CONSTANT,
            "error_gain": ERROR_GAIN,
            "pretrained": USE_PRETRAINED,
            "layer_loss_weights": LAYER_LOSS_WEIGHTS,
            "top_variance_weight": TOP_VARIANCE_WEIGHT,
            "top_variance_target": TOP_VARIANCE_TARGET,
            "top_variance_eps": TOP_VARIANCE_EPS,
            "top_variance_window": TOP_VARIANCE_WINDOW,
            "temporal_prediction_weight": TEMPORAL_PREDICTION_WEIGHT,
        },
        "epochs": [],
        "best_checkpoint": None,
    }

    best_val_temporal_loss = float("inf")
    best_student_checkpoint_path = SAVE_BEST_STUDENT_PATH or derive_checkpoint_path(
        OUTPUT_PATH,
        "_best_student.pt",
    )

    start = datetime.now()
    print(f"STARTING AT : {start}", flush=True)
    temporal_loss_label = (
        "standardized_motion_mse"
        if TASK_ALIGNED_TARGET == "ego_motion"
        else "temporal_loss"
    )
    temporal_cosine_label = (
        "raw_motion_cosine"
        if TASK_ALIGNED_TARGET == "ego_motion"
        else "temporal_cosine"
    )

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(
            student,
            teacher,
            train_loader,
            optimizer=optimizer,
            motion_target_stats=motion_target_stats,
        )
        val_metrics = (
            run_epoch(
                student,
                teacher,
                val_loader,
                optimizer=None,
                motion_target_stats=motion_target_stats,
            )
            if val_loader
            else None
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history["epochs"].append(epoch_record)

        if val_metrics is not None:
            val_temporal_loss = val_metrics["mean_temporal_loss"]
            if val_temporal_loss < best_val_temporal_loss:
                best_val_temporal_loss = val_temporal_loss
                history["best_checkpoint"] = {
                    "metric": (
                        "val_standardized_longitudinal_yaw_mse"
                        if TASK_ALIGNED_TARGET == "ego_motion"
                        else "val_mean_temporal_loss"
                    ),
                    "value": val_temporal_loss,
                    "epoch": epoch,
                    "path": best_student_checkpoint_path,
                }
                save_model_checkpoint(
                    student,
                    best_student_checkpoint_path,
                    history["config"],
                    epoch_record,
                    checkpoint_kind="best_val_temporal_loss",
                )
                print(
                    f"Saved best student checkpoint at epoch {epoch} "
                    f"with val_temporal_loss={val_temporal_loss:.6f} "
                    f"to {best_student_checkpoint_path}",
                    flush=True,
                )

        print(
            f"Epoch {epoch:03d} | "
            f"train_weighted={train_metrics['mean_weighted_loss']:.6f} | "
            f"train_objective={train_metrics['mean_optimized_loss']:.6f} | "
            f"train_top={train_metrics['mean_top_local_loss']:.6f} | "
            f"train_top_std={train_metrics['mean_top_feature_std']:.6f} | "
            f"train_top_poolstd={train_metrics['mean_top_pooled_std']:.6f} | "
            f"train_varloss={train_metrics['mean_top_variance_loss']:.6f} | "
            f"train_varframes={train_metrics['max_top_variance_sample_count']} | "
            f"train_{temporal_loss_label}={train_metrics['mean_temporal_loss']:.6f} | "
            f"train_normmae={train_metrics['mean_temporal_mae']:.6f} | "
            f"train_forward_mae_m={train_metrics['mean_forward_displacement_mae_m']:.6f} | "
            f"train_yaw_mae_rad={train_metrics['mean_yaw_change_mae_rad']:.6f} | "
            f"train_{temporal_cosine_label}={train_metrics['mean_temporal_cosine']:.6f}",
            flush=True,
        )
        if val_metrics is not None:
            print(
                f"Epoch {epoch:03d} | "
                f"val_weighted={val_metrics['mean_weighted_loss']:.6f} | "
                f"val_objective={val_metrics['mean_optimized_loss']:.6f} | "
                f"val_top={val_metrics['mean_top_local_loss']:.6f} | "
                f"val_top_std={val_metrics['mean_top_feature_std']:.6f} | "
                f"val_top_poolstd={val_metrics['mean_top_pooled_std']:.6f} | "
                f"val_varloss={val_metrics['mean_top_variance_loss']:.6f} | "
                f"val_varframes={val_metrics['max_top_variance_sample_count']} | "
                f"val_{temporal_loss_label}={val_metrics['mean_temporal_loss']:.6f} | "
                f"val_normmae={val_metrics['mean_temporal_mae']:.6f} | "
                f"val_forward_mae_m={val_metrics['mean_forward_displacement_mae_m']:.6f} | "
                f"val_yaw_mae_rad={val_metrics['mean_yaw_change_mae_rad']:.6f} | "
                f"val_{temporal_cosine_label}={val_metrics['mean_temporal_cosine']:.6f}",
                flush=True,
            )

    end = datetime.now()
    print(f"TOTAL TIME TAKEN : {end-start}", flush=True)

    with open(OUTPUT_PATH, "wb") as handle:
        pickle.dump(history, handle)
    print(f"Saved training history to {OUTPUT_PATH}", flush=True)

    student_checkpoint_path = SAVE_STUDENT_PATH or derive_checkpoint_path(OUTPUT_PATH, "_student.pt")
    save_model_checkpoint(
        student,
        student_checkpoint_path,
        history["config"],
        history["epochs"][-1] if history["epochs"] else None,
        checkpoint_kind="final_student",
    )
    print(f"Saved student checkpoint to {student_checkpoint_path}", flush=True)

    if teacher is not None:
        teacher_checkpoint_path = SAVE_TEACHER_PATH or derive_checkpoint_path(OUTPUT_PATH, "_teacher.pt")
        save_model_checkpoint(
            teacher,
            teacher_checkpoint_path,
            history["config"],
            history["epochs"][-1] if history["epochs"] else None,
            checkpoint_kind="final_teacher",
        )
        print(f"Saved teacher checkpoint to {teacher_checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
