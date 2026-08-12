import math
from bisect import bisect_right
from datetime import datetime
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.transforms import transforms


def split_contiguous_indices(indices):
    segments = []
    for index in indices:
        if not segments or index != segments[-1][-1] + 1:
            segments.append([index])
        else:
            segments[-1].append(index)
    return tuple(tuple(segment) for segment in segments)


def get_sample_raw_frame_indices(dataset, index):
    """Return the raw frame positions read by one temporal sample."""
    if isinstance(dataset, Subset):
        return get_sample_raw_frame_indices(dataset.dataset, dataset.indices[index])

    if isinstance(dataset, ConcatDataset):
        dataset_index = bisect_right(dataset.cumulative_sizes, index)
        previous_size = (
            dataset.cumulative_sizes[dataset_index - 1] if dataset_index > 0 else 0
        )
        return get_sample_raw_frame_indices(
            dataset.datasets[dataset_index],
            index - previous_size,
        )

    if isinstance(dataset, ShuffledFuturePairDataset):
        return get_sample_raw_frame_indices(
            dataset.dataset,
            dataset.reference_indices[index],
        )

    valid_start_indices = getattr(dataset, "valid_start_indices", None)
    if valid_start_indices is None:
        raise TypeError(
            f"Dataset {type(dataset).__name__} does not expose valid_start_indices."
        )
    start_index = int(valid_start_indices[index])
    horizons = tuple(int(value) for value in getattr(dataset, "horizons", (1,)))
    return frozenset((start_index, *(start_index + horizon for horizon in horizons)))


def collect_raw_frame_indices(dataset):
    """Collect every raw frame position read by a dataset or subset."""
    return frozenset(
        raw_index
        for sample_index in range(len(dataset))
        for raw_index in get_sample_raw_frame_indices(dataset, sample_index)
    )


def build_same_drive_train_val_subsets(
    dataset,
    train_fraction=0.6,
    val_fraction=0.2,
    gap_frames=20,
):
    """Split one drive in raw-frame space with an explicit temporal gap."""
    if isinstance(dataset, (Subset, ConcatDataset)):
        raise TypeError("Same-drive splitting requires one unsliced drive dataset.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1.")
    if gap_frames < 0:
        raise ValueError(f"gap_frames must be non-negative, got {gap_frames}.")

    frame_paths = getattr(dataset, "frame_paths", None)
    if frame_paths is None:
        raise TypeError(
            f"Dataset {type(dataset).__name__} does not expose frame_paths."
        )
    raw_frame_count = len(frame_paths)
    train_stop = int(math.floor(raw_frame_count * train_fraction))
    val_frame_count = int(math.floor(raw_frame_count * val_fraction))
    val_start = train_stop + gap_frames
    val_stop = val_start + val_frame_count
    if val_stop > raw_frame_count:
        raise ValueError(
            "The requested same-drive split does not fit in the drive: "
            f"train_stop={train_stop}, gap_frames={gap_frames}, "
            f"val_frame_count={val_frame_count}, raw_frame_count={raw_frame_count}."
        )

    train_sample_indices = []
    val_sample_indices = []
    excluded_sample_indices = []
    for sample_index in range(len(dataset)):
        raw_indices = get_sample_raw_frame_indices(dataset, sample_index)
        if max(raw_indices) < train_stop:
            train_sample_indices.append(sample_index)
        elif min(raw_indices) >= val_start and max(raw_indices) < val_stop:
            val_sample_indices.append(sample_index)
        else:
            excluded_sample_indices.append(sample_index)

    if not train_sample_indices or not val_sample_indices:
        raise ValueError(
            "Same-drive split produced an empty train or validation subset: "
            f"train_samples={len(train_sample_indices)}, "
            f"val_samples={len(val_sample_indices)}."
        )

    train_dataset = Subset(dataset, train_sample_indices)
    val_dataset = Subset(dataset, val_sample_indices)
    train_raw_frames = collect_raw_frame_indices(train_dataset)
    val_raw_frames = collect_raw_frame_indices(val_dataset)
    shared_raw_frames = train_raw_frames & val_raw_frames
    if shared_raw_frames:
        raise RuntimeError(
            "Same-drive train/validation split shares raw frames: "
            f"{sorted(shared_raw_frames)}."
        )

    metadata = {
        "raw_frame_count": raw_frame_count,
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
        "minimum_gap_frames": int(gap_frames),
        "actual_gap_frames": int(gap_frames),
        "train_raw_frame_range": (0, train_stop - 1),
        "gap_raw_frame_range": (train_stop, val_start - 1),
        "val_raw_frame_range": (val_start, val_stop - 1),
        "unused_tail_raw_frame_range": (
            (val_stop, raw_frame_count - 1) if val_stop < raw_frame_count else None
        ),
        "train_sample_indices": tuple(train_sample_indices),
        "val_sample_indices": tuple(val_sample_indices),
        "excluded_sample_count": len(excluded_sample_indices),
        "train_raw_frame_count": len(train_raw_frames),
        "val_raw_frame_count": len(val_raw_frames),
        "shared_raw_frame_count": 0,
    }
    return train_dataset, val_dataset, metadata


class KITTINextFramePairDataset(Dataset):
    """
    Build adjacent-frame pairs from a single KITTI Raw drive.

    Sample i is:
        (frame_i, frame_{i+1})
    """

    def __init__(self, root, drive, camera="image_02", fixed_dt_s=None, dt_tolerance_s=0.0):
        self.sequence_dir = Path(root) / drive / camera / "data"
        self.frame_timestamps_path = Path(root) / drive / camera / "timestamps.txt"
        if not self.sequence_dir.is_dir():
            raise FileNotFoundError(f"Could not find KITTI image directory: {self.sequence_dir}")
        if not self.frame_timestamps_path.is_file():
            raise FileNotFoundError(f"Could not find KITTI timestamps: {self.frame_timestamps_path}")

        self.frame_paths = sorted(self.sequence_dir.glob("*.png"))
        if len(self.frame_paths) < 2:
            raise FileNotFoundError(
                f"Need at least 2 PNG frames to build adjacent pairs: {self.sequence_dir}"
            )
        with self.frame_timestamps_path.open("r") as handle:
            self.frame_timestamps = [_parse_timestamp(line) for line in handle if line.strip()]
        if len(self.frame_timestamps) != len(self.frame_paths):
            raise ValueError(
                f"Frame/timestamp count mismatch for {camera}: frames={len(self.frame_paths)}, "
                f"timestamps={len(self.frame_timestamps)}"
            )

        self.fixed_dt_s = float(fixed_dt_s) if fixed_dt_s is not None else None
        self.dt_tolerance_s = float(dt_tolerance_s)
        if self.fixed_dt_s is not None and self.fixed_dt_s <= 0:
            raise ValueError(f"fixed_dt_s must be positive, but got {self.fixed_dt_s}.")
        if self.dt_tolerance_s < 0:
            raise ValueError(f"dt_tolerance_s must be non-negative, but got {self.dt_tolerance_s}.")
        self.valid_start_indices = []
        self.valid_sample_segments = ()
        self.time_filter_stats = {}

        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._set_valid_start_indices(max_horizon=1)

    def __len__(self):
        return len(self.valid_start_indices)

    def _load_frame(self, path):
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

    def _get_frame_dt_seconds(self, left_index, right_index):
        return (self.frame_timestamps[right_index] - self.frame_timestamps[left_index]).total_seconds()

    def _step_matches_fixed_dt(self, left_index, right_index):
        dt_seconds = self._get_frame_dt_seconds(left_index, right_index)
        if dt_seconds <= 0:
            return False
        if self.fixed_dt_s is None:
            return True
        return abs(dt_seconds - self.fixed_dt_s) <= self.dt_tolerance_s

    def _set_valid_start_indices(self, max_horizon):
        candidate_count = len(self.frame_paths) - max_horizon
        if candidate_count <= 0:
            raise FileNotFoundError(
                f"Need at least {max_horizon + 1} frames to build max_horizon={max_horizon}: "
                f"{self.sequence_dir}"
            )

        valid_start_indices = []
        dropped_count = 0
        for start_index in range(candidate_count):
            sample_valid = True
            for step in range(1, max_horizon + 1):
                if not self._step_matches_fixed_dt(start_index + step - 1, start_index + step):
                    sample_valid = False
                    break
            if sample_valid:
                valid_start_indices.append(start_index)
            else:
                dropped_count += 1

        if not valid_start_indices:
            if self.fixed_dt_s is None:
                raise RuntimeError(f"No valid KITTI temporal samples found under {self.sequence_dir}.")
            raise RuntimeError(
                f"No valid KITTI temporal samples remain after fixed-dt filtering under {self.sequence_dir}. "
                f"Requested fixed_dt_s={self.fixed_dt_s}, dt_tolerance_s={self.dt_tolerance_s}, "
                f"max_horizon={max_horizon}."
            )

        self.valid_start_indices = valid_start_indices
        raw_index_segments = split_contiguous_indices(valid_start_indices)
        sample_segments = []
        sample_offset = 0
        for segment in raw_index_segments:
            sample_segments.append(tuple(range(sample_offset, sample_offset + len(segment))))
            sample_offset += len(segment)
        self.valid_sample_segments = tuple(sample_segments)
        self.time_filter_stats = {
            "fixed_dt_s": self.fixed_dt_s,
            "dt_tolerance_s": self.dt_tolerance_s,
            "max_horizon": max_horizon,
            "candidate_samples": candidate_count,
            "valid_samples": len(valid_start_indices),
            "dropped_samples": dropped_count,
            "num_contiguous_segments": len(self.valid_sample_segments),
            "contiguous_segment_lengths": tuple(
                len(segment) for segment in self.valid_sample_segments
            ),
        }

    def __getitem__(self, index):
        start_index = self.valid_start_indices[index]
        current_path = self.frame_paths[start_index]
        next_path = self.frame_paths[start_index + 1]
        current_frame = self._load_frame(current_path)
        next_frame = self._load_frame(next_path)
        return current_frame, next_frame, current_path.name, next_path.name


class KITTIMultiHorizonFrameDataset(KITTINextFramePairDataset):
    """
    Build multi-horizon future-frame samples from a single KITTI Raw drive.

    Sample i is:
        (frame_i, [frame_{i+h} for h in horizons])
    """

    def __init__(self, root, drive, camera="image_02", horizons=(1,), fixed_dt_s=None, dt_tolerance_s=0.0):
        super().__init__(root, drive, camera, fixed_dt_s=fixed_dt_s, dt_tolerance_s=dt_tolerance_s)
        parsed_horizons = tuple(int(horizon) for horizon in horizons)
        if not parsed_horizons:
            raise ValueError("At least one temporal horizon is required.")
        if any(horizon <= 0 for horizon in parsed_horizons):
            raise ValueError(f"Temporal horizons must be positive: {parsed_horizons}")

        self.horizons = parsed_horizons
        self.max_horizon = max(self.horizons)
        self._set_valid_start_indices(max_horizon=self.max_horizon)

    def __getitem__(self, index):
        start_index = self.valid_start_indices[index]
        current_path = self.frame_paths[start_index]
        current_frame = self._load_frame(current_path)

        future_frames = []
        future_names = []
        for horizon in self.horizons:
            future_path = self.frame_paths[start_index + horizon]
            future_frames.append(self._load_frame(future_path))
            future_names.append(future_path.name)

        return current_frame, torch.stack(future_frames, dim=0), current_path.name, tuple(future_names)


def _parse_timestamp(timestamp_text):
    timestamp_text = timestamp_text.strip()
    if "." not in timestamp_text:
        return datetime.fromisoformat(timestamp_text)

    prefix, fractional = timestamp_text.split(".", 1)
    microseconds = fractional[:6].ljust(6, "0")
    return datetime.fromisoformat(f"{prefix}.{microseconds}")


def _wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class KITTIEgoMotionPairDataset(KITTINextFramePairDataset):
    """
    Adjacent-frame dataset with a 2-DoF longitudinal-yaw target from OXTS.

    Targets:
        delta_forward_m : forward displacement between t and t+1
        delta_yaw_rad   : yaw change between t and t+1
    """

    def __init__(self, root, drive, camera="image_02", fixed_dt_s=None, dt_tolerance_s=0.0):
        super().__init__(root, drive, camera, fixed_dt_s=fixed_dt_s, dt_tolerance_s=dt_tolerance_s)
        drive_root = Path(root) / drive
        self.oxts_dir = drive_root / "oxts" / "data"
        self.oxts_timestamps_path = drive_root / "oxts" / "timestamps.txt"

        if not self.oxts_dir.is_dir():
            raise FileNotFoundError(f"Could not find OXTS directory: {self.oxts_dir}")
        if not self.oxts_timestamps_path.is_file():
            raise FileNotFoundError(f"Could not find OXTS timestamps: {self.oxts_timestamps_path}")

        self.oxts_paths = sorted(self.oxts_dir.glob("*.txt"))
        if len(self.oxts_paths) != len(self.frame_paths):
            raise ValueError(
                f"Frame/OXTS count mismatch for {drive}: frames={len(self.frame_paths)}, "
                f"oxts={len(self.oxts_paths)}"
            )

        with self.oxts_timestamps_path.open("r") as handle:
            self.oxts_timestamps = [_parse_timestamp(line) for line in handle if line.strip()]
        if len(self.oxts_timestamps) != len(self.frame_paths):
            raise ValueError(
                f"Frame/timestamp count mismatch for {drive}: frames={len(self.frame_paths)}, "
                f"timestamps={len(self.oxts_timestamps)}"
            )

    def _load_oxts_values(self, path):
        with path.open("r") as handle:
            values = [float(value) for value in handle.readline().strip().split()]
        if len(values) < 23:
            raise ValueError(f"Unexpected OXTS format in {path}")
        return values

    def _build_pair_target(self, index):
        current_values = self._load_oxts_values(self.oxts_paths[index])
        next_values = self._load_oxts_values(self.oxts_paths[index + 1])

        delta_time_s = self._get_frame_dt_seconds(index, index + 1)
        if delta_time_s <= 0:
            raise ValueError(f"Non-positive frame dt at pair index {index}: {delta_time_s}")

        current_yaw = current_values[5]
        next_yaw = next_values[5]
        current_vf = current_values[8]
        next_vf = next_values[8]

        mean_forward_speed = 0.5 * (current_vf + next_vf)
        delta_forward_m = mean_forward_speed * delta_time_s
        delta_yaw_rad = _wrap_angle(next_yaw - current_yaw)

        return torch.tensor([delta_forward_m, delta_yaw_rad], dtype=torch.float32)

    def _build_horizon_target(self, index, horizon):
        if horizon <= 0:
            raise ValueError(f"Horizon must be positive, but got {horizon}.")
        if index + horizon >= len(self.frame_paths):
            raise IndexError(
                f"Horizon target out of range: index={index}, horizon={horizon}, frames={len(self.frame_paths)}."
            )

        delta_forward_m = 0.0
        for step in range(horizon):
            delta_forward_m += float(self._build_pair_target(index + step)[0].item())

        current_values = self._load_oxts_values(self.oxts_paths[index])
        future_values = self._load_oxts_values(self.oxts_paths[index + horizon])
        delta_yaw_rad = _wrap_angle(future_values[5] - current_values[5])
        return torch.tensor([delta_forward_m, delta_yaw_rad], dtype=torch.float32)

    def __getitem__(self, index):
        current_frame, next_frame, current_name, next_name = super().__getitem__(index)
        target = self.get_motion_target(index)
        return current_frame, next_frame, target, current_name, next_name

    def get_motion_target(self, index):
        start_index = self.valid_start_indices[index]
        return self._build_pair_target(start_index)


class KITTIEgoMotionMultiHorizonDataset(KITTIEgoMotionPairDataset):
    """
    Multi-horizon KITTI dataset with 2-DoF longitudinal-yaw targets.

    Sample i is:
        (frame_i, [frame_{i+h}], [motion_{i->i+h}])
    """

    def __init__(self, root, drive, camera="image_02", horizons=(1,), fixed_dt_s=None, dt_tolerance_s=0.0):
        super().__init__(root, drive, camera, fixed_dt_s=fixed_dt_s, dt_tolerance_s=dt_tolerance_s)
        parsed_horizons = tuple(int(horizon) for horizon in horizons)
        if not parsed_horizons:
            raise ValueError("At least one temporal horizon is required.")
        if any(horizon <= 0 for horizon in parsed_horizons):
            raise ValueError(f"Temporal horizons must be positive: {parsed_horizons}")

        self.horizons = parsed_horizons
        self.max_horizon = max(self.horizons)
        self._set_valid_start_indices(max_horizon=self.max_horizon)

    def __getitem__(self, index):
        start_index = self.valid_start_indices[index]
        current_path = self.frame_paths[start_index]
        current_frame = self._load_frame(current_path)

        future_frames = []
        future_names = []
        for horizon in self.horizons:
            future_path = self.frame_paths[start_index + horizon]
            future_frames.append(self._load_frame(future_path))
            future_names.append(future_path.name)

        return (
            current_frame,
            torch.stack(future_frames, dim=0),
            self.get_motion_target(index),
            current_path.name,
            tuple(future_names),
        )

    def get_motion_target(self, index):
        start_index = self.valid_start_indices[index]
        return torch.stack(
            [self._build_horizon_target(start_index, horizon) for horizon in self.horizons],
            dim=0,
        )


def build_kitti_pair_dataloader(
    root,
    drive,
    camera="image_02",
    fixed_dt_s=None,
    dt_tolerance_s=0.0,
    max_pairs=0,
    batch_size=1,
    num_workers=0,
    pin_memory=False,
):
    dataset = KITTINextFramePairDataset(
        root,
        drive,
        camera,
        fixed_dt_s=fixed_dt_s,
        dt_tolerance_s=dt_tolerance_s,
    )
    if max_pairs > 0 and max_pairs < len(dataset):
        dataset = Subset(dataset, list(range(max_pairs)))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_kitti_multi_horizon_dataloader(
    root,
    drive,
    camera="image_02",
    horizons=(1,),
    fixed_dt_s=None,
    dt_tolerance_s=0.0,
    max_pairs=0,
    batch_size=1,
    num_workers=0,
    pin_memory=False,
):
    dataset = KITTIMultiHorizonFrameDataset(
        root,
        drive,
        camera,
        horizons=horizons,
        fixed_dt_s=fixed_dt_s,
        dt_tolerance_s=dt_tolerance_s,
    )
    if max_pairs > 0 and max_pairs < len(dataset):
        dataset = Subset(dataset, list(range(max_pairs)))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_kitti_pair_dataset(root, drives, camera="image_02", fixed_dt_s=None, dt_tolerance_s=0.0, max_pairs=0):
    if isinstance(drives, str):
        drives = [drives]

    datasets = [
        KITTINextFramePairDataset(
            root,
            drive,
            camera,
            fixed_dt_s=fixed_dt_s,
            dt_tolerance_s=dt_tolerance_s,
        )
        for drive in drives
    ]
    if not datasets:
        raise ValueError("At least one KITTI drive is required.")

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    if max_pairs > 0 and max_pairs < len(dataset):
        dataset = Subset(dataset, list(range(max_pairs)))
    return dataset


def build_kitti_multi_horizon_dataset(
    root,
    drives,
    camera="image_02",
    horizons=(1,),
    fixed_dt_s=None,
    dt_tolerance_s=0.0,
    max_pairs=0,
):
    if isinstance(drives, str):
        drives = [drives]

    datasets = [
        KITTIMultiHorizonFrameDataset(
            root,
            drive,
            camera,
            horizons=horizons,
            fixed_dt_s=fixed_dt_s,
            dt_tolerance_s=dt_tolerance_s,
        )
        for drive in drives
    ]
    if not datasets:
        raise ValueError("At least one KITTI drive is required.")

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    if max_pairs > 0 and max_pairs < len(dataset):
        dataset = Subset(dataset, list(range(max_pairs)))
    return dataset


def build_kitti_ego_motion_pair_dataset(
    root,
    drives,
    camera="image_02",
    fixed_dt_s=None,
    dt_tolerance_s=0.0,
    max_pairs=0,
):
    if isinstance(drives, str):
        drives = [drives]

    datasets = [
        KITTIEgoMotionPairDataset(
            root,
            drive,
            camera,
            fixed_dt_s=fixed_dt_s,
            dt_tolerance_s=dt_tolerance_s,
        )
        for drive in drives
    ]
    if not datasets:
        raise ValueError("At least one KITTI drive is required.")

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    if max_pairs > 0 and max_pairs < len(dataset):
        dataset = Subset(dataset, list(range(max_pairs)))
    return dataset


def build_kitti_ego_motion_multi_horizon_dataset(
    root,
    drives,
    camera="image_02",
    horizons=(1,),
    fixed_dt_s=None,
    dt_tolerance_s=0.0,
    max_pairs=0,
):
    if isinstance(drives, str):
        drives = [drives]

    datasets = [
        KITTIEgoMotionMultiHorizonDataset(
            root,
            drive,
            camera,
            horizons=horizons,
            fixed_dt_s=fixed_dt_s,
            dt_tolerance_s=dt_tolerance_s,
        )
        for drive in drives
    ]
    if not datasets:
        raise ValueError("At least one KITTI drive is required.")

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    if max_pairs > 0 and max_pairs < len(dataset):
        dataset = Subset(dataset, list(range(max_pairs)))
    return dataset


def collect_time_filter_stats(dataset):
    if isinstance(dataset, Subset):
        stats = collect_time_filter_stats(dataset.dataset)
        if stats:
            stats = dict(stats)
            stats["subset_samples"] = len(dataset)
        return stats

    if isinstance(dataset, ConcatDataset):
        child_stats = [collect_time_filter_stats(child) for child in dataset.datasets]
        child_stats = [stats for stats in child_stats if stats]
        if not child_stats:
            return {}
        fixed_dt_values = {stats.get("fixed_dt_s") for stats in child_stats}
        tolerance_values = {stats.get("dt_tolerance_s") for stats in child_stats}
        return {
            "fixed_dt_s": fixed_dt_values.pop() if len(fixed_dt_values) == 1 else None,
            "dt_tolerance_s": tolerance_values.pop() if len(tolerance_values) == 1 else None,
            "max_horizon": max(stats.get("max_horizon", 1) for stats in child_stats),
            "candidate_samples": sum(stats.get("candidate_samples", 0) for stats in child_stats),
            "valid_samples": sum(stats.get("valid_samples", 0) for stats in child_stats),
            "dropped_samples": sum(stats.get("dropped_samples", 0) for stats in child_stats),
            "num_contiguous_segments": sum(
                stats.get("num_contiguous_segments", 0) for stats in child_stats
            ),
            "contiguous_segment_lengths": tuple(
                length
                for stats in child_stats
                for length in stats.get("contiguous_segment_lengths", ())
            ),
        }

    return dict(getattr(dataset, "time_filter_stats", {}))


def get_motion_target(dataset, index):
    if isinstance(dataset, Subset):
        return get_motion_target(dataset.dataset, dataset.indices[index])

    if isinstance(dataset, ConcatDataset):
        dataset_index = bisect_right(dataset.cumulative_sizes, index)
        previous_size = dataset.cumulative_sizes[dataset_index - 1] if dataset_index > 0 else 0
        return get_motion_target(dataset.datasets[dataset_index], index - previous_size)

    if isinstance(dataset, ShuffledFuturePairDataset):
        return get_motion_target(dataset.dataset, dataset.reference_indices[index])

    target_getter = getattr(dataset, "get_motion_target", None)
    if target_getter is None:
        raise TypeError(f"Dataset does not expose motion targets without loading images: {type(dataset)}")
    return target_getter(index)


def build_derangement(length, seed=0):
    if length <= 1:
        raise ValueError(f"Need at least 2 samples to build a derangement, but got length={length}.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    base_indices = torch.arange(length)
    for _ in range(128):
        permutation = torch.randperm(length, generator=generator)
        if not torch.any(permutation == base_indices):
            return permutation.tolist()

    permutation = torch.roll(base_indices, shifts=1)
    return permutation.tolist()


class ShuffledFuturePairDataset(Dataset):
    """
    Keep current-frame inputs fixed, but replace their future frame / target branch
    with a deterministic shuffled sample from the same dataset.

    This is used as a negative control to break temporal alignment while preserving
    tensor shapes and marginal target statistics.
    """

    def __init__(self, dataset, seed=0):
        if len(dataset) <= 1:
            raise ValueError("ShuffledFuturePairDataset requires at least 2 samples.")
        self.dataset = dataset
        self.shuffle_seed = int(seed)
        self.reference_indices = build_derangement(len(dataset), seed=self.shuffle_seed)
        base_stats = collect_time_filter_stats(dataset)
        self.time_filter_stats = dict(base_stats) if base_stats else {}
        self.time_filter_stats.update(
            {
                "shuffled_future_pairs": True,
                "shuffle_seed": self.shuffle_seed,
                "wrapped_samples": len(dataset),
            }
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        current_sample = self.dataset[index]
        reference_sample = self.dataset[self.reference_indices[index]]

        if not isinstance(current_sample, tuple) or not isinstance(reference_sample, tuple):
            raise TypeError("ShuffledFuturePairDataset expects tuple samples.")

        if len(current_sample) == 4:
            current_frame, _, current_name, _ = current_sample
            _, shuffled_future, _, shuffled_future_name = reference_sample
            return current_frame, shuffled_future, current_name, shuffled_future_name

        if len(current_sample) == 5:
            current_frame, _, _, current_name, _ = current_sample
            _, shuffled_future, shuffled_target, _, shuffled_future_name = reference_sample
            return current_frame, shuffled_future, shuffled_target, current_name, shuffled_future_name

        raise ValueError(
            f"Unsupported sample format for ShuffledFuturePairDataset: len={len(current_sample)}."
        )
