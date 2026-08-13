import hashlib
import io
import math
from collections import Counter
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from .kitti_pairs import KITTIMultiHorizonFrameDataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ControlledCorruptionSchedule:
    trajectory: str = "step_hold_recovery"
    baseline_frames: int = 10
    transition_frames: int = 1
    hold_frames: int = 17
    recovery_frames: int = 18

    def __post_init__(self):
        allowed_trajectories = {
            "step_hold_recovery",
            "ramp_hold_recovery",
            "iid_noise_recovery",
        }
        if self.trajectory not in allowed_trajectories:
            raise ValueError(
                f"Unsupported corruption trajectory {self.trajectory!r}; "
                f"expected one of {sorted(allowed_trajectories)}."
            )
        frame_counts = (
            self.baseline_frames,
            self.transition_frames,
            self.hold_frames,
            self.recovery_frames,
        )
        if any(value < 0 for value in frame_counts):
            raise ValueError(f"Schedule frame counts must be non-negative: {frame_counts}.")
        if self.transition_frames + self.hold_frames <= 0:
            raise ValueError("The schedule must contain at least one disturbed frame.")
        if self.trajectory == "step_hold_recovery" and self.transition_frames != 1:
            raise ValueError("A step trajectory requires exactly one transition frame.")
        if self.trajectory == "ramp_hold_recovery" and self.transition_frames <= 1:
            raise ValueError("A ramp trajectory requires at least two transition frames.")
        if self.trajectory == "iid_noise_recovery" and self.transition_frames != 0:
            raise ValueError("An i.i.d. noise trajectory does not have a transition phase.")
        if self.recovery_frames <= 0:
            raise ValueError("The schedule must contain a recovery phase.")

    @property
    def total_frames(self):
        return (
            self.baseline_frames
            + self.transition_frames
            + self.hold_frames
            + self.recovery_frames
        )

    @property
    def disturbance_onset_offset(self):
        return self.baseline_frames

    @property
    def recovery_onset_offset(self):
        return (
            self.baseline_frames
            + self.transition_frames
            + self.hold_frames
        )

    def phase_and_severity(self, frame_offset):
        frame_offset = int(frame_offset)
        if frame_offset < self.baseline_frames:
            return "baseline", 0.0

        transition_stop = self.baseline_frames + self.transition_frames
        disturbed_stop = transition_stop + self.hold_frames
        if self.trajectory == "step_hold_recovery":
            if frame_offset < transition_stop:
                return "step_change", 1.0
            if frame_offset < disturbed_stop:
                return "persistent_bias", 1.0
        elif self.trajectory == "ramp_hold_recovery":
            if frame_offset < transition_stop:
                ramp_index = frame_offset - self.baseline_frames
                return "ramp_change", (ramp_index + 1) / self.transition_frames
            if frame_offset < disturbed_stop:
                return "persistent_bias", 1.0
        elif frame_offset < disturbed_stop:
            return "iid_noise", 1.0

        return "recovery", 0.0

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ExplicitSeveritySchedule:
    """Absolute-frame schedule for matched-marginal temporal controls."""

    baseline_frames: int
    disturbance_severities: tuple
    recovery_frames: int

    def __post_init__(self):
        if self.baseline_frames < 0:
            raise ValueError("baseline_frames must be non-negative.")
        if self.recovery_frames <= 0:
            raise ValueError("recovery_frames must be positive.")
        if not self.disturbance_severities:
            raise ValueError("disturbance_severities must not be empty.")
        for severity in self.disturbance_severities:
            if not math.isfinite(float(severity)) or not 0.0 < float(severity) <= 1.0:
                raise ValueError(
                    "Every disturbance severity must be finite and in (0, 1], "
                    f"got {severity!r}."
                )

    @property
    def total_frames(self):
        return (
            self.baseline_frames
            + len(self.disturbance_severities)
            + self.recovery_frames
        )

    @property
    def disturbance_onset_offset(self):
        return self.baseline_frames

    @property
    def recovery_onset_offset(self):
        return self.baseline_frames + len(self.disturbance_severities)

    def phase_and_severity(self, frame_offset):
        frame_offset = int(frame_offset)
        if frame_offset < self.baseline_frames:
            return "baseline", 0.0
        disturbance_index = frame_offset - self.baseline_frames
        if disturbance_index < len(self.disturbance_severities):
            return "disturbance", float(
                self.disturbance_severities[disturbance_index]
            )
        return "recovery", 0.0

    def severity_counts(self):
        return dict(Counter(float(value) for value in self.disturbance_severities))

    def to_dict(self):
        return {
            "baseline_frames": self.baseline_frames,
            "disturbance_severities": [
                float(value) for value in self.disturbance_severities
            ],
            "disturbance_severity_counts": self.severity_counts(),
            "recovery_frames": self.recovery_frames,
        }


@dataclass(frozen=True)
class ControlledCorruptionConfig:
    corruption_type: str = "bias"
    bias_rgb: tuple = (0.15, -0.08, 0.05)
    noise_std: float = 0.03
    blur_kernel_size: int = 11
    blur_sigma: float = 3.0
    motion_blur_kernel_size: int = 9
    contrast_factor: float = 0.6
    fog_alpha: float = 0.2
    jpeg_quality: int = 50
    seed: int = 0

    def __post_init__(self):
        if self.corruption_type not in {
            "bias",
            "iid_gaussian",
            "gaussian_blur",
            "motion_blur",
            "contrast",
            "fog",
            "jpeg_compression",
        }:
            raise ValueError(
                "Unsupported corruption_type: "
                f"{self.corruption_type!r}."
            )
        if len(self.bias_rgb) != 3:
            raise ValueError("bias_rgb must contain exactly three channel values.")
        if any(not math.isfinite(float(value)) for value in self.bias_rgb):
            raise ValueError(f"bias_rgb must be finite, got {self.bias_rgb}.")
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError(f"noise_std must be finite and non-negative, got {self.noise_std}.")
        if self.corruption_type == "iid_gaussian" and self.noise_std <= 0:
            raise ValueError("iid_gaussian corruption requires noise_std > 0.")
        if self.blur_kernel_size <= 0 or self.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be a positive odd integer.")
        if not math.isfinite(self.blur_sigma) or self.blur_sigma <= 0:
            raise ValueError("blur_sigma must be finite and positive.")
        if self.motion_blur_kernel_size <= 0 or self.motion_blur_kernel_size % 2 == 0:
            raise ValueError("motion_blur_kernel_size must be a positive odd integer.")
        if not 0.0 < self.contrast_factor <= 1.0:
            raise ValueError("contrast_factor must be in (0, 1].")
        if not 0.0 < self.fog_alpha < 1.0:
            raise ValueError("fog_alpha must be in (0, 1).")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100].")

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

    corrupted = image_tensor
    if config.corruption_type == "bias":
        bias = image_tensor.new_tensor(config.bias_rgb).view(3, 1, 1)
        corrupted = corrupted + severity * bias
    elif config.corruption_type == "iid_gaussian":
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
    elif config.corruption_type == "gaussian_blur":
        corrupted = TF.gaussian_blur(
            image_tensor,
            kernel_size=[config.blur_kernel_size, config.blur_kernel_size],
            sigma=[severity * config.blur_sigma, severity * config.blur_sigma],
        )
    elif config.corruption_type == "motion_blur":
        kernel_size = config.motion_blur_kernel_size
        kernel = image_tensor.new_zeros((3, 1, kernel_size, kernel_size))
        kernel[:, 0, kernel_size // 2, :] = 1.0 / kernel_size
        padded = F.pad(
            image_tensor.unsqueeze(0),
            (kernel_size // 2,) * 4,
            mode="reflect",
        )
        blurred = F.conv2d(padded, kernel, groups=3).squeeze(0)
        corrupted = (1.0 - severity) * image_tensor + severity * blurred
    elif config.corruption_type == "contrast":
        mean = image_tensor.mean(dim=(1, 2), keepdim=True)
        factor = 1.0 - severity * (1.0 - config.contrast_factor)
        corrupted = mean + factor * (image_tensor - mean)
    elif config.corruption_type == "fog":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _absolute_frame_seed(config.seed, drive, camera, frame_name)
        )
        veil = torch.rand((1, 1, 28, 28), generator=generator)
        veil = F.interpolate(
            veil,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        veil = TF.gaussian_blur(veil, kernel_size=[31, 31], sigma=[10.0, 10.0])
        veil = 0.7 + 0.3 * veil
        alpha = severity * config.fog_alpha
        corrupted = (1.0 - alpha) * image_tensor + alpha * veil
    elif config.corruption_type == "jpeg_compression":
        quality = round(100.0 - severity * (100 - config.jpeg_quality))
        buffer = io.BytesIO()
        TF.to_pil_image(image_tensor).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            corrupted = TF.pil_to_tensor(decoded.convert("RGB")).float().div(255.0)
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
        signed_excess = record["corrupted_feature_mse"] - record["clean_feature_mse"]
        record["signed_excess_feature_mse"] = signed_excess
        record["absolute_excess_feature_mse"] = abs(signed_excess)
        record["positive_excess_feature_mse"] = max(0.0, signed_excess)

    disturbed_phases = {
        "step_change",
        "ramp_change",
        "persistent_bias",
        "iid_noise",
    }
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
    peak_signed_excess_record = max(
        analysis_records,
        key=lambda record: record["signed_excess_feature_mse"],
    )
    minimum_signed_excess_record = min(
        analysis_records,
        key=lambda record: record["signed_excess_feature_mse"],
    )
    peak_absolute_excess_record = max(
        analysis_records,
        key=lambda record: record["absolute_excess_feature_mse"],
    )
    baseline_signed_excess = sum(
        record["signed_excess_feature_mse"] for record in baseline
    ) / len(baseline)
    peak_absolute_excursion = max(
        abs(record["signed_excess_feature_mse"] - baseline_signed_excess)
        for record in analysis_records
    )
    recovery_threshold_radius = recovery_fraction * peak_absolute_excursion

    recovered_record = None
    for index in range(len(recovery)):
        window = recovery[index : index + recovery_consecutive_frames]
        if len(window) < recovery_consecutive_frames:
            break
        if all(
            abs(record["signed_excess_feature_mse"] - baseline_signed_excess)
            <= recovery_threshold_radius
            for record in window
        ):
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
        "peak_signed_excess_mse": peak_signed_excess_record[
            "signed_excess_feature_mse"
        ],
        "peak_signed_excess_raw_frame_index": peak_signed_excess_record[
            "future_raw_frame_index"
        ],
        "minimum_signed_excess_mse": minimum_signed_excess_record[
            "signed_excess_feature_mse"
        ],
        "minimum_signed_excess_raw_frame_index": minimum_signed_excess_record[
            "future_raw_frame_index"
        ],
        "peak_absolute_excess_mse": peak_absolute_excess_record[
            "absolute_excess_feature_mse"
        ],
        "peak_absolute_excess_raw_frame_index": peak_absolute_excess_record[
            "future_raw_frame_index"
        ],
        "baseline_signed_excess_mse": baseline_signed_excess,
        "recovery_threshold_absolute_deviation_mse": recovery_threshold_radius,
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
        "signed_excess_auec_mse_seconds": sum(
            record["signed_excess_feature_mse"] * sample_time_s
            for record in analysis_records
        ),
        "absolute_excess_auec_mse_seconds": sum(
            record["absolute_excess_feature_mse"] * sample_time_s
            for record in analysis_records
        ),
        "positive_excess_auec_mse_seconds": sum(
            record["positive_excess_feature_mse"] * sample_time_s
            for record in analysis_records
        ),
        "recovery_signed_excess_auec_mse_seconds": sum(
            record["signed_excess_feature_mse"] * sample_time_s
            for record in recovery
        ),
        "analysis_frame_count": len(analysis_records),
        "baseline_frame_count": len(baseline),
        "disturbed_frame_count": len(disturbed),
        "recovery_frame_count": len(recovery),
    }
