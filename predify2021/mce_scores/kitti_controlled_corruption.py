import hashlib
import math
from dataclasses import asdict, dataclass

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .kitti_pairs import KITTIMultiHorizonFrameDataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ControlledCorruptionSchedule:
    baseline_frames: int = 6
    step_frames: int = 3
    ramp_frames: int = 5
    persistent_frames: int = 7
    recovery_frames: int = 9
    step_level: float = 0.5

    def __post_init__(self):
        frame_counts = (
            self.baseline_frames,
            self.step_frames,
            self.ramp_frames,
            self.persistent_frames,
            self.recovery_frames,
        )
        if any(value < 0 for value in frame_counts):
            raise ValueError(f"Schedule frame counts must be non-negative: {frame_counts}.")
        if self.step_frames + self.ramp_frames + self.persistent_frames <= 0:
            raise ValueError("The schedule must contain at least one disturbed frame.")
        if self.recovery_frames <= 0:
            raise ValueError("The schedule must contain a recovery phase.")
        if not 0.0 < self.step_level <= 1.0:
            raise ValueError(f"step_level must be in (0, 1], got {self.step_level}.")

    @property
    def total_frames(self):
        return (
            self.baseline_frames
            + self.step_frames
            + self.ramp_frames
            + self.persistent_frames
            + self.recovery_frames
        )

    @property
    def disturbance_onset_offset(self):
        return self.baseline_frames

    @property
    def recovery_onset_offset(self):
        return (
            self.baseline_frames
            + self.step_frames
            + self.ramp_frames
            + self.persistent_frames
        )

    def phase_and_severity(self, frame_offset):
        frame_offset = int(frame_offset)
        if frame_offset < self.baseline_frames:
            return "baseline", 0.0

        step_stop = self.baseline_frames + self.step_frames
        if frame_offset < step_stop:
            return "step_change", self.step_level

        ramp_stop = step_stop + self.ramp_frames
        if frame_offset < ramp_stop:
            ramp_index = frame_offset - step_stop
            progress = (ramp_index + 1) / max(1, self.ramp_frames)
            severity = self.step_level + (1.0 - self.step_level) * progress
            return "ramp_change", min(1.0, severity)

        persistent_stop = ramp_stop + self.persistent_frames
        if frame_offset < persistent_stop:
            return "persistent_bias", 1.0

        return "recovery", 0.0

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ControlledCorruptionConfig:
    bias_rgb: tuple = (0.15, -0.08, 0.05)
    noise_std: float = 0.03
    seed: int = 0

    def __post_init__(self):
        if len(self.bias_rgb) != 3:
            raise ValueError("bias_rgb must contain exactly three channel values.")
        if any(not math.isfinite(float(value)) for value in self.bias_rgb):
            raise ValueError(f"bias_rgb must be finite, got {self.bias_rgb}.")
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError(f"noise_std must be finite and non-negative, got {self.noise_std}.")

    def to_dict(self):
        return asdict(self)


def _absolute_frame_seed(experiment_seed, drive, camera, frame_name):
    identity = f"{int(experiment_seed)}|{drive}|{camera}|{frame_name}".encode("utf-8")
    digest = hashlib.sha256(identity).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def apply_controlled_corruption(
    image_tensor,
    severity,
    config,
    drive,
    camera,
    frame_name,
):
    """Apply deterministic corruption in [0, 1] before normalization."""
    severity = float(severity)
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}.")
    if severity == 0.0:
        return image_tensor

    bias = image_tensor.new_tensor(config.bias_rgb).view(3, 1, 1)
    corrupted = image_tensor + severity * bias
    if config.noise_std > 0:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _absolute_frame_seed(config.seed, drive, camera, frame_name)
        )
        noise = torch.randn(
            image_tensor.shape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).to(image_tensor.dtype)
        corrupted = corrupted + severity * config.noise_std * noise
    return corrupted.clamp(0.0, 1.0)


class ControlledCorruptionKITTIDataset(KITTIMultiHorizonFrameDataset):
    """KITTI stream with absolute-frame deterministic pre-normalization corruption."""

    def __init__(
        self,
        root,
        drive,
        camera="image_02",
        horizons=(1,),
        fixed_dt_s=None,
        dt_tolerance_s=0.0,
        schedule=None,
        corruption_config=None,
        schedule_start_raw_index=0,
        corruption_enabled=True,
    ):
        super().__init__(
            root,
            drive,
            camera=camera,
            horizons=horizons,
            fixed_dt_s=fixed_dt_s,
            dt_tolerance_s=dt_tolerance_s,
        )
        self.drive = drive
        self.camera = camera
        self.schedule = schedule or ControlledCorruptionSchedule()
        self.corruption_config = corruption_config or ControlledCorruptionConfig()
        self.schedule_start_raw_index = int(schedule_start_raw_index)
        self.corruption_enabled = bool(corruption_enabled)
        self._raw_position_by_path = {
            path: raw_position for raw_position, path in enumerate(self.frame_paths)
        }

    def get_frame_corruption_metadata(self, raw_frame_index):
        frame_offset = int(raw_frame_index) - self.schedule_start_raw_index
        phase, severity = self.schedule.phase_and_severity(frame_offset)
        return {
            "raw_frame_index": int(raw_frame_index),
            "schedule_frame_offset": frame_offset,
            "phase": phase,
            "severity": severity if self.corruption_enabled else 0.0,
            "scheduled_severity": severity,
        }

    def get_sample_corruption_metadata(self, sample_index):
        start_index = int(self.valid_start_indices[sample_index])
        return {
            "current": self.get_frame_corruption_metadata(start_index),
            "future": tuple(
                self.get_frame_corruption_metadata(start_index + horizon)
                for horizon in self.horizons
            ),
        }

    def _load_frame(self, path):
        raw_frame_index = self._raw_position_by_path[path]
        metadata = self.get_frame_corruption_metadata(raw_frame_index)
        with Image.open(path) as image:
            resized = TF.resize(image.convert("RGB"), 224)
            cropped = TF.center_crop(resized, (224, 224))
            image_tensor = TF.pil_to_tensor(cropped).float().div(255.0)

        if self.corruption_enabled:
            image_tensor = apply_controlled_corruption(
                image_tensor,
                metadata["scheduled_severity"],
                self.corruption_config,
                self.drive,
                self.camera,
                path.name,
            )
        return TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def compute_controlled_recovery_metrics(
    frame_records,
    sample_time_s,
    recovery_fraction=0.1,
    recovery_consecutive_frames=3,
):
    if not frame_records:
        raise ValueError("At least one frame record is required.")
    if sample_time_s <= 0:
        raise ValueError(f"sample_time_s must be positive, got {sample_time_s}.")
    if not 0.0 <= recovery_fraction <= 1.0:
        raise ValueError(
            f"recovery_fraction must be in [0, 1], got {recovery_fraction}."
        )
    if recovery_consecutive_frames <= 0:
        raise ValueError("recovery_consecutive_frames must be positive.")

    for record in frame_records:
        record["excess_feature_mse"] = max(
            0.0,
            record["corrupted_feature_mse"] - record["clean_feature_mse"],
        )

    disturbed_phases = {"step_change", "ramp_change", "persistent_bias"}
    disturbed = [
        record for record in frame_records if record["future_phase"] in disturbed_phases
    ]
    recovery = [
        record for record in frame_records if record["future_phase"] == "recovery"
    ]
    baseline = [
        record for record in frame_records if record["future_phase"] == "baseline"
    ]
    if not disturbed or not recovery or not baseline:
        raise ValueError(
            "Frame records must contain baseline, disturbed, and recovery phases."
        )

    analysis_records = disturbed + recovery
    peak_error_record = max(
        analysis_records,
        key=lambda record: record["corrupted_feature_mse"],
    )
    peak_excess_record = max(
        analysis_records,
        key=lambda record: record["excess_feature_mse"],
    )
    baseline_excess = sum(record["excess_feature_mse"] for record in baseline) / len(
        baseline
    )
    peak_excess = peak_excess_record["excess_feature_mse"]
    recovery_threshold = baseline_excess + recovery_fraction * max(
        0.0,
        peak_excess - baseline_excess,
    )

    recovered_record = None
    for index in range(len(recovery)):
        window = recovery[index : index + recovery_consecutive_frames]
        if len(window) < recovery_consecutive_frames:
            break
        if all(record["excess_feature_mse"] <= recovery_threshold for record in window):
            recovered_record = window[0]
            break

    recovery_start_raw_index = recovery[0]["future_raw_frame_index"]
    recovery_time_frames = (
        recovered_record["future_raw_frame_index"] - recovery_start_raw_index
        if recovered_record is not None
        else None
    )
    return {
        "peak_error_mse": peak_error_record["corrupted_feature_mse"],
        "peak_error_rms": math.sqrt(peak_error_record["corrupted_feature_mse"]),
        "peak_error_raw_frame_index": peak_error_record["future_raw_frame_index"],
        "peak_excess_mse": peak_excess,
        "peak_excess_raw_frame_index": peak_excess_record["future_raw_frame_index"],
        "baseline_excess_mse": baseline_excess,
        "recovery_threshold_excess_mse": recovery_threshold,
        "recovery_fraction": recovery_fraction,
        "recovery_consecutive_frames": recovery_consecutive_frames,
        "recovery_time_frames": recovery_time_frames,
        "recovery_time_s": (
            recovery_time_frames * sample_time_s
            if recovery_time_frames is not None
            else None
        ),
        "recovery_censored": recovered_record is None,
        "recovered_raw_frame_index": (
            recovered_record["future_raw_frame_index"]
            if recovered_record is not None
            else None
        ),
        "auec_mse_seconds": sum(
            record["corrupted_feature_mse"] * sample_time_s
            for record in analysis_records
        ),
        "excess_auec_mse_seconds": sum(
            record["excess_feature_mse"] * sample_time_s
            for record in analysis_records
        ),
        "recovery_excess_auec_mse_seconds": sum(
            record["excess_feature_mse"] * sample_time_s for record in recovery
        ),
        "analysis_frame_count": len(analysis_records),
        "baseline_frame_count": len(baseline),
        "disturbed_frame_count": len(disturbed),
        "recovery_frame_count": len(recovery),
    }
