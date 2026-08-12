import csv
import json
import math
import os
import re
import subprocess
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predify2021.mce_scores.kitti_controlled_corruption import (
    ControlledCorruptionConfig,
    ControlledCorruptionKITTIDataset,
    ControlledCorruptionSchedule,
    compute_controlled_recovery_metrics,
)
from predify2021.mce_scores.kitti_pairs import build_same_drive_train_val_subsets
from predify2021.model_factory.get_model import get_model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _parse_checkpoint_specs(raw_value):
    specs = []
    labels = set()
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "PREDIFY_CONTROLLED_CHECKPOINTS entries must be label=/path/to/checkpoint.pt."
            )
        label, path = (part.strip() for part in item.split("=", 1))
        if not label or not path:
            raise ValueError(f"Invalid checkpoint specification: {item!r}.")
        if re.fullmatch(r"[A-Za-z0-9_.-]+", label) is None:
            raise ValueError(
                f"Checkpoint label may contain only letters, numbers, '.', '_', and '-': {label!r}."
            )
        if label in labels:
            raise ValueError(f"Duplicate checkpoint label: {label!r}.")
        labels.add(label)
        specs.append((label, Path(path)))
    if not specs:
        raise ValueError("PREDIFY_CONTROLLED_CHECKPOINTS must contain at least one checkpoint.")
    return specs


def _parse_float_tuple(raw_value, expected_length, name):
    values = tuple(float(value.strip()) for value in raw_value.split(",") if value.strip())
    if len(values) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} values, got {values}.")
    return values


def _resolve_git_revision():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def validate_controlled_checkpoint(checkpoint, drive, split_metadata):
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint must contain a config dictionary.")
    if config.get("prediction_task") != "future_feature":
        raise ValueError(
            "Controlled corruption evaluation requires prediction_task='future_feature'."
        )
    history_mode = config.get("future_feature_history_mode")
    if history_mode not in {"none", "temporal_error", "copy_current"}:
        raise ValueError(
            "Controlled corruption evaluation accepts only none, temporal_error, or "
            f"copy_current history, got {history_mode!r}."
        )
    if not config.get("same_drive_split"):
        raise ValueError(
            "Controlled corruption claims require a checkpoint trained with "
            "PREDIFY_SAME_DRIVE_SPLIT=1."
        )
    if config.get("same_drive_drive") != drive:
        raise ValueError(
            f"Checkpoint same_drive_drive={config.get('same_drive_drive')!r} does not "
            f"match evaluation drive={drive!r}."
        )

    checkpoint_split = config.get("same_drive_split_metadata")
    if not isinstance(checkpoint_split, dict):
        raise ValueError("Checkpoint is missing same_drive_split_metadata.")
    fields_to_match = (
        "raw_frame_count",
        "train_raw_frame_range",
        "val_raw_frame_range",
        "minimum_gap_frames",
    )
    mismatches = {}
    for field in fields_to_match:
        checkpoint_value = checkpoint_split.get(field)
        evaluation_value = split_metadata.get(field)
        if field.endswith("_range"):
            matches = tuple(checkpoint_value or ()) == tuple(evaluation_value or ())
        else:
            matches = checkpoint_value == evaluation_value
        if not matches:
            mismatches[field] = (checkpoint_value, evaluation_value)
    if mismatches:
        raise ValueError(f"Checkpoint/evaluation same-drive split mismatch: {mismatches}.")
    expected_train_pairs = len(checkpoint_split.get("train_sample_indices", ()))
    expected_val_pairs = len(checkpoint_split.get("val_sample_indices", ()))
    if config.get("train_pairs") != expected_train_pairs or config.get(
        "val_pairs"
    ) != expected_val_pairs:
        raise ValueError(
            "Controlled corruption evaluation requires the complete same-drive "
            "train and validation sample ranges; checkpoint pair counts are "
            f"train={config.get('train_pairs')}, val={config.get('val_pairs')}, "
            f"expected train={expected_train_pairs}, val={expected_val_pairs}."
        )

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must contain a state_dict dictionary.")
    predictor_weight = state_dict.get("future_feature_predictor.0.weight")
    if not torch.is_tensor(predictor_weight) or predictor_weight.ndim != 4:
        raise ValueError("Checkpoint is missing future_feature_predictor.0.weight.")
    kernel_shape = tuple(int(value) for value in predictor_weight.shape[-2:])
    if kernel_shape[0] != kernel_shape[1] or kernel_shape[0] not in {1, 3}:
        raise ValueError(f"Unsupported predictor kernel shape: {kernel_shape}.")
    configured_kernel = int(config.get("future_feature_predictor_kernel_size", 0))
    if configured_kernel != kernel_shape[0]:
        raise ValueError(
            f"Checkpoint kernel config={configured_kernel} disagrees with weight={kernel_shape}."
        )
    prediction_form = config.get("future_feature_prediction_form", "current_residual")
    if prediction_form == "residual_Fhat_next=F_current+delta_hat":
        prediction_form = "current_residual"
    if prediction_form not in {
        "current_residual",
        "historical_warp",
        "historical_warp_residual",
    }:
        raise ValueError(f"Unsupported future feature prediction form: {prediction_form!r}.")
    return {
        "history_mode": history_mode,
        "predictor_kernel_size": configured_kernel,
        "prediction_form": prediction_form,
        "future_motion_radius": int(config.get("future_motion_radius", 1)),
        "future_motion_patch_size": int(config.get("future_motion_patch_size", 3)),
        "checkpoint_kind": checkpoint.get("checkpoint_kind"),
        "checkpoint_git_revision": config.get("git_revision", "unknown"),
        "selected_epoch": checkpoint.get("selected_epoch"),
    }


def _build_model(checkpoint, validation):
    config = checkpoint["config"]
    model = get_model(
        "pvgg_tf",
        pretrained=False,
        target_flow_mode=config.get("target_flow_mode", "recursive"),
        temporal_target_mode=config.get("temporal_target_mode", "next_top"),
        temporal_horizons=tuple(config.get("temporal_horizons", (1,))),
        dynamic_error=config.get("dynamic_error", True),
        error_state_mode=config.get("error_state_mode", "ema"),
        local_loss_error_source=config.get("local_loss_error_source", "instant"),
        error_sample_time=config.get("error_sample_time", 0.1035),
        error_time_constant=config.get("error_time_constant", 0.5),
        error_gain=config.get("error_gain", 1.0),
        temporal_error_sample_time=config.get("temporal_error_sample_time", 0.1035),
        temporal_error_time_constant=config.get("temporal_error_time_constant", 0.5),
        temporal_error_gain=config.get("temporal_error_gain", 1.0),
        task="future_feature",
        future_feature_history_mode=validation["history_mode"],
        future_feature_predictor_kernel_size=validation["predictor_kernel_size"],
        future_feature_prediction_form=validation["prediction_form"],
        future_motion_radius=validation["future_motion_radius"],
        future_motion_patch_size=validation["future_motion_patch_size"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model.to(DEVICE)


def _tensor_rms(tensor):
    return math.sqrt(float(tensor.detach().float().square().mean().cpu().item()))


def summarize_predictor_input_weights(model):
    weight = model.future_feature_predictor[0].weight.detach().float()
    input_channels = int(weight.shape[1])
    if input_channels % 2 != 0:
        raise ValueError(
            "Future-feature predictor input channels must split evenly into F and E, "
            f"got {input_channels}."
        )
    split = input_channels // 2
    feature_weight_rms = _tensor_rms(weight[:, :split])
    history_weight_rms = _tensor_rms(weight[:, split:])
    return {
        "feature_input_weight_rms": feature_weight_rms,
        "history_input_weight_rms": history_weight_rms,
        "history_to_feature_weight_rms_ratio": history_weight_rms
        / max(feature_weight_rms, 1e-12),
    }


def compute_history_utilization(model, outputs):
    current_top = outputs["current_top"].detach()
    history_top = outputs["history_top"].detach()
    predicted_delta = outputs["predicted_delta_top"].detach()
    future_target = outputs["future_top_target"].detach()
    history_mode = model.future_feature_history_mode

    if history_mode == "copy_current":
        zero_history_delta = torch.zeros_like(predicted_delta)
        history_contribution = torch.zeros_like(predicted_delta)
        utilization_applicable = False
    else:
        zero_history_delta = model.future_feature_predictor(
            torch.cat([current_top, torch.zeros_like(history_top)], dim=1)
        ).detach()
        history_contribution = predicted_delta - zero_history_delta
        utilization_applicable = history_mode == "temporal_error"

    zero_history_error = future_target - (current_top + zero_history_delta)
    actual_error = future_target - (current_top + predicted_delta)
    current_rms = _tensor_rms(current_top)
    history_rms = _tensor_rms(history_top)
    delta_rms = _tensor_rms(predicted_delta)
    contribution_rms = _tensor_rms(history_contribution)
    return {
        "history_utilization_applicable": utilization_applicable,
        "current_feature_rms": current_rms,
        "history_input_rms": history_rms,
        "history_to_feature_rms_ratio": history_rms / max(current_rms, 1e-12),
        "predicted_delta_rms": delta_rms,
        "zero_history_predicted_delta_rms": _tensor_rms(zero_history_delta),
        "history_contribution_rms": contribution_rms,
        "history_contribution_to_delta_rms_ratio": contribution_rms
        / max(delta_rms, 1e-12),
        "zero_history_feature_mse": float(
            zero_history_error.float().square().mean().cpu().item()
        ),
        "history_feature_mse_change": float(
            (
                actual_error.float().square().mean()
                - zero_history_error.float().square().mean()
            )
            .cpu()
            .item()
        ),
    }


def summarize_temporal_response(records):
    phase_order = (
        "baseline",
        "step_change",
        "ramp_change",
        "persistent_bias",
        "iid_noise",
        "recovery",
    )
    metric_names = (
        "corrupted_prediction_error_rms",
        "corrupted_temporal_error_input_rms",
        "corrupted_temporal_error_state_rms",
        "corrupted_feature_mse",
        "corrupted_current_feature_rms",
        "corrupted_history_input_rms",
        "corrupted_history_to_feature_rms_ratio",
        "corrupted_history_contribution_rms",
        "corrupted_history_contribution_to_delta_rms_ratio",
        "corrupted_history_feature_mse_change",
    )
    phase_means = {}
    for phase in phase_order:
        phase_records = [record for record in records if record["future_phase"] == phase]
        if not phase_records:
            continue
        phase_means[phase] = {
            "frame_count": len(phase_records),
            **{
                metric_name: sum(record[metric_name] for record in phase_records)
                / len(phase_records)
                for metric_name in metric_names
            },
        }

    peak_e = max(records, key=lambda record: record["corrupted_prediction_error_rms"])
    peak_state = max(
        records,
        key=lambda record: record["corrupted_temporal_error_state_rms"],
    )
    return {
        "phase_means": phase_means,
        "peak_prediction_error_rms": peak_e["corrupted_prediction_error_rms"],
        "peak_prediction_error_raw_frame_index": peak_e["future_raw_frame_index"],
        "peak_temporal_error_state_rms": peak_state[
            "corrupted_temporal_error_state_rms"
        ],
        "peak_temporal_error_state_raw_frame_index": peak_state[
            "future_raw_frame_index"
        ],
        "e_to_E_peak_lag_frames": (
            peak_state["future_raw_frame_index"] - peak_e["future_raw_frame_index"]
        ),
    }


def _evaluate_stream(model, dataset, val_sample_indices):
    model.reset()
    records = []
    with torch.inference_mode():
        for stream_index, sample_index in enumerate(val_sample_indices):
            current_frame, future_frames, current_name, future_names = dataset[
                sample_index
            ]
            current_frame = current_frame.unsqueeze(0).to(DEVICE)
            next_frame = future_frames[0].unsqueeze(0).to(DEVICE)
            metadata = dataset.get_sample_corruption_metadata(sample_index)
            current_metadata = metadata["current"]
            future_metadata = metadata["future"][0]

            model.step_frame(
                current_frame,
                top_target_provider=lambda: model.extract_top_forward_feature(
                    next_frame,
                    detach=True,
                ),
            )
            outputs = model.future_prediction_outputs
            prediction_error = outputs["prediction_error_top"].detach().float()
            prediction_mse = float(prediction_error.square().mean().cpu().item())
            copy_error = (
                outputs["future_top_target"].detach().float()
                - outputs["current_top"].detach().float()
            )
            temporal_state = model.temporal_error_state_memory
            if temporal_state is None:
                raise RuntimeError("Temporal prediction error state was not updated.")
            utilization = compute_history_utilization(model, outputs)
            records.append(
                {
                    "stream_index": stream_index,
                    "sample_index": int(sample_index),
                    "current_raw_frame_index": current_metadata["raw_frame_index"],
                    "future_raw_frame_index": future_metadata["raw_frame_index"],
                    "current_frame_name": current_name,
                    "future_frame_name": future_names[0],
                    "current_phase": current_metadata["phase"],
                    "future_phase": future_metadata["phase"],
                    "current_severity": current_metadata["severity"],
                    "future_severity": future_metadata["severity"],
                    "prediction_error_mean": float(prediction_error.mean().cpu().item()),
                    "prediction_error_mse": prediction_mse,
                    "prediction_error_rms": math.sqrt(prediction_mse),
                    "temporal_error_input_mean": float(
                        outputs["history_top"].detach().float().mean().cpu().item()
                    ),
                    "temporal_error_input_rms": _tensor_rms(outputs["history_top"]),
                    "temporal_error_state_mean": float(
                        temporal_state.detach().float().mean().cpu().item()
                    ),
                    "temporal_error_state_rms": _tensor_rms(temporal_state),
                    "feature_mse": prediction_mse,
                    "copy_current_feature_mse": float(
                        copy_error.square().mean().cpu().item()
                    ),
                    **utilization,
                }
            )
    return records


def _merge_clean_and_corrupted(clean_records, corrupted_records):
    if len(clean_records) != len(corrupted_records):
        raise RuntimeError("Clean/corrupted streams have different lengths.")
    merged = []
    metric_names = (
        "prediction_error_mean",
        "prediction_error_mse",
        "prediction_error_rms",
        "temporal_error_input_mean",
        "temporal_error_input_rms",
        "temporal_error_state_mean",
        "temporal_error_state_rms",
        "feature_mse",
        "copy_current_feature_mse",
        "current_feature_rms",
        "history_input_rms",
        "history_to_feature_rms_ratio",
        "predicted_delta_rms",
        "zero_history_predicted_delta_rms",
        "history_contribution_rms",
        "history_contribution_to_delta_rms_ratio",
        "zero_history_feature_mse",
        "history_feature_mse_change",
    )
    for clean, corrupted in zip(clean_records, corrupted_records):
        if clean["future_raw_frame_index"] != corrupted["future_raw_frame_index"]:
            raise RuntimeError("Clean/corrupted streams are not raw-frame aligned.")
        record = {
            key: value
            for key, value in corrupted.items()
            if key not in metric_names
        }
        record["current_severity"] = corrupted["current_severity"]
        record["future_severity"] = corrupted["future_severity"]
        record["history_utilization_applicable"] = corrupted[
            "history_utilization_applicable"
        ]
        for metric_name in metric_names:
            record[f"clean_{metric_name}"] = clean[metric_name]
            record[f"corrupted_{metric_name}"] = corrupted[metric_name]
        merged.append(record)
    return merged


def _write_frame_artifacts(output_dir, label, records):
    jsonl_path = output_dir / f"{label}_frames.jsonl"
    csv_path = output_dir / f"{label}_recovery_curve.csv"
    with jsonl_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    fieldnames = list(records[0])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return jsonl_path, csv_path


def _plot_recovery_curves(output_path, records_by_label, trajectory_name):
    figure, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    phase_colors = {
        "step_change": "#f5c2c7",
        "ramp_change": "#ffe69c",
        "persistent_bias": "#badbcc",
        "iid_noise": "#d8c7e8",
        "recovery": "#cfe2ff",
    }
    first_records = next(iter(records_by_label.values()))
    x_values = [record["future_raw_frame_index"] for record in first_records]
    for record in first_records:
        color = phase_colors.get(record["future_phase"])
        if color:
            x = record["future_raw_frame_index"]
            for axis in axes:
                axis.axvspan(x - 0.5, x + 0.5, color=color, alpha=0.22, linewidth=0)

    for label, records in records_by_label.items():
        x_values = [record["future_raw_frame_index"] for record in records]
        axes[0].plot(
            x_values,
            [record["signed_excess_feature_mse"] for record in records],
            label=label,
            linewidth=1.8,
        )
        axes[1].plot(
            x_values,
            [record["corrupted_feature_mse"] for record in records],
            label=f"{label} corrupted",
            linewidth=1.8,
        )
        axes[1].plot(
            x_values,
            [record["clean_feature_mse"] for record in records],
            label=f"{label} clean",
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
        )
        axes[2].plot(
            x_values,
            [record["corrupted_prediction_error_rms"] for record in records],
            label=label,
            linewidth=1.8,
        )
        axes[3].plot(
            x_values,
            [record["corrupted_temporal_error_state_rms"] for record in records],
            label=label,
            linewidth=1.8,
        )

    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_ylabel("signed delta L_t")
    axes[1].set_ylabel("L_t feature MSE")
    axes[2].set_ylabel("e_t RMS")
    axes[3].set_ylabel("E_t RMS")
    axes[3].set_xlabel("Absolute future raw frame index")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle(f"Same-drive controlled corruption: {trajectory_name}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_independent_trajectory_specs(
    validation_raw_frame_count,
    baseline_frames,
    recovery_frames,
    ramp_frames,
    bias_rgb,
    noise_std,
    seed,
):
    disturbed_frames = (
        int(validation_raw_frame_count) - int(baseline_frames) - int(recovery_frames)
    )
    if disturbed_frames <= 1:
        raise ValueError(
            "Validation segment is too short for baseline, disturbance, and recovery: "
            f"raw_frames={validation_raw_frame_count}, baseline={baseline_frames}, "
            f"recovery={recovery_frames}."
        )
    if not 1 < ramp_frames <= disturbed_frames:
        raise ValueError(
            f"ramp_frames must be in [2, {disturbed_frames}], got {ramp_frames}."
        )

    bias_config = ControlledCorruptionConfig(
        corruption_type="bias",
        bias_rgb=bias_rgb,
        noise_std=0.0,
        seed=seed,
    )
    noise_config = ControlledCorruptionConfig(
        corruption_type="iid_gaussian",
        bias_rgb=(0.0, 0.0, 0.0),
        noise_std=noise_std,
        seed=seed,
    )
    return {
        "step_bias": {
            "schedule": ControlledCorruptionSchedule(
                trajectory="step_hold_recovery",
                baseline_frames=baseline_frames,
                transition_frames=1,
                hold_frames=disturbed_frames - 1,
                recovery_frames=recovery_frames,
            ),
            "corruption": bias_config,
            "interpretation": "abrupt fixed RGB bias, hold, then clean recovery",
        },
        "ramp_bias": {
            "schedule": ControlledCorruptionSchedule(
                trajectory="ramp_hold_recovery",
                baseline_frames=baseline_frames,
                transition_frames=ramp_frames,
                hold_frames=disturbed_frames - ramp_frames,
                recovery_frames=recovery_frames,
            ),
            "corruption": bias_config,
            "interpretation": "gradual fixed RGB bias, hold, then clean recovery",
        },
        "iid_noise": {
            "schedule": ControlledCorruptionSchedule(
                trajectory="iid_noise_recovery",
                baseline_frames=baseline_frames,
                transition_frames=0,
                hold_frames=disturbed_frames,
                recovery_frames=recovery_frames,
            ),
            "corruption": noise_config,
            "interpretation": "absolute-frame deterministic i.i.d. Gaussian-noise negative control",
        },
    }


def main():
    checkpoint_specs = _parse_checkpoint_specs(
        os.environ.get("PREDIFY_CONTROLLED_CHECKPOINTS", "")
    )
    output_dir = Path(
        os.environ.get(
            "PREDIFY_CONTROLLED_OUTPUT_DIR",
            "/tmp/predify-storage/experiments/same_drive_controlled_corruption",
        )
    )
    summary_path = output_dir / "summary.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to write into non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    root = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
    drive = os.environ.get(
        "PREDIFY_KITTI_DRIVE",
        "2011_09_26/2011_09_26_drive_0005_sync",
    )
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    dt_tolerance_s = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
    train_fraction = float(os.environ.get("PREDIFY_TRAIN_FRACTION", "0.6"))
    val_fraction = float(os.environ.get("PREDIFY_VAL_FRACTION", "0.2"))
    gap_frames = int(os.environ.get("PREDIFY_SAME_DRIVE_GAP_FRAMES", "20"))
    if not math.isclose(train_fraction, 0.6) or not math.isclose(val_fraction, 0.2):
        raise ValueError(
            "Controlled evaluation requires 60% train, a raw-frame gap, then 20% val."
        )
    if gap_frames != 20:
        raise ValueError("Controlled evaluation requires exactly 20 raw gap frames.")

    baseline_frames = int(
        os.environ.get("PREDIFY_CORRUPTION_BASELINE_FRAMES", "10")
    )
    recovery_frames = int(
        os.environ.get("PREDIFY_CORRUPTION_RECOVERY_FRAMES", "18")
    )
    ramp_frames = int(os.environ.get("PREDIFY_CORRUPTION_RAMP_FRAMES", "8"))
    bias_rgb = _parse_float_tuple(
        os.environ.get("PREDIFY_CORRUPTION_BIAS_RGB", "0.15,-0.08,0.05"),
        3,
        "PREDIFY_CORRUPTION_BIAS_RGB",
    )
    noise_std = float(os.environ.get("PREDIFY_CORRUPTION_NOISE_STD", "0.03"))
    corruption_seed = int(os.environ.get("PREDIFY_CORRUPTION_SEED", "0"))
    recovery_fraction = float(
        os.environ.get("PREDIFY_RECOVERY_THRESHOLD_FRACTION", "0.1")
    )
    recovery_consecutive_frames = int(
        os.environ.get("PREDIFY_RECOVERY_CONSECUTIVE_FRAMES", "3")
    )

    split_probe_dataset = ControlledCorruptionKITTIDataset(
        root,
        drive,
        camera=camera,
        horizons=(1,),
        fixed_dt_s=fixed_dt_s,
        dt_tolerance_s=dt_tolerance_s,
        corruption_enabled=False,
    )
    _, validation_subset, split_metadata = build_same_drive_train_val_subsets(
        split_probe_dataset,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        gap_frames=gap_frames,
    )
    schedule_start = split_metadata["val_raw_frame_range"][0]
    val_sample_indices = tuple(int(index) for index in validation_subset.indices)
    if val_sample_indices != tuple(
        range(val_sample_indices[0], val_sample_indices[-1] + 1)
    ):
        raise ValueError(
            "Controlled corruption evaluation requires one contiguous fixed-dt "
            "validation stream."
        )

    validation_raw_frame_count = (
        split_metadata["val_raw_frame_range"][1] - schedule_start + 1
    )
    trajectory_specs = build_independent_trajectory_specs(
        validation_raw_frame_count=validation_raw_frame_count,
        baseline_frames=baseline_frames,
        recovery_frames=recovery_frames,
        ramp_frames=ramp_frames,
        bias_rgb=bias_rgb,
        noise_std=noise_std,
        seed=corruption_seed,
    )

    records_by_trajectory = {name: {} for name in trajectory_specs}
    checkpoint_summaries = {}
    for label, checkpoint_path in checkpoint_specs:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        print(f"Evaluating {label}: {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validation = validate_controlled_checkpoint(
            checkpoint,
            drive=drive,
            split_metadata=split_metadata,
        )
        model = _build_model(checkpoint, validation)
        checkpoint_summaries[label] = {
            "checkpoint_path": str(checkpoint_path.resolve()),
            **validation,
            "predictor_input_weights": summarize_predictor_input_weights(model),
            "trajectories": {},
        }
        for trajectory_name, trajectory_spec in trajectory_specs.items():
            print(f"  trajectory {trajectory_name}", flush=True)
            schedule = trajectory_spec["schedule"]
            corruption_config = trajectory_spec["corruption"]
            clean_dataset = ControlledCorruptionKITTIDataset(
                root,
                drive,
                camera=camera,
                horizons=(1,),
                fixed_dt_s=fixed_dt_s,
                dt_tolerance_s=dt_tolerance_s,
                schedule=schedule,
                corruption_config=corruption_config,
                schedule_start_raw_index=schedule_start,
                corruption_enabled=False,
            )
            corrupted_dataset = ControlledCorruptionKITTIDataset(
                root,
                drive,
                camera=camera,
                horizons=(1,),
                fixed_dt_s=fixed_dt_s,
                dt_tolerance_s=dt_tolerance_s,
                schedule=schedule,
                corruption_config=corruption_config,
                schedule_start_raw_index=schedule_start,
                corruption_enabled=True,
            )
            clean_records = _evaluate_stream(model, clean_dataset, val_sample_indices)
            corrupted_records = _evaluate_stream(
                model,
                corrupted_dataset,
                val_sample_indices,
            )
            merged_records = _merge_clean_and_corrupted(
                clean_records,
                corrupted_records,
            )
            recovery_metrics = compute_controlled_recovery_metrics(
                merged_records,
                sample_time_s=fixed_dt_s,
                recovery_fraction=recovery_fraction,
                recovery_consecutive_frames=recovery_consecutive_frames,
            )
            artifact_label = f"{label}__{trajectory_name}"
            jsonl_path, csv_path = _write_frame_artifacts(
                output_dir,
                artifact_label,
                merged_records,
            )
            records_by_trajectory[trajectory_name][label] = merged_records
            checkpoint_summaries[label]["trajectories"][trajectory_name] = {
                "frame_jsonl": str(jsonl_path.resolve()),
                "recovery_curve_csv": str(csv_path.resolve()),
                "metrics": recovery_metrics,
                "temporal_response": summarize_temporal_response(merged_records),
            }
        del model
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    curve_paths = {}
    for trajectory_name, records_by_label in records_by_trajectory.items():
        curve_path = output_dir / f"{trajectory_name}_recovery.png"
        _plot_recovery_curves(curve_path, records_by_label, trajectory_name)
        curve_paths[trajectory_name] = str(curve_path.resolve())
    summary = {
        "experiment": "kitti_same_drive_independent_controlled_corruption",
        "git_revision": _resolve_git_revision(),
        "device": str(DEVICE),
        "kitti_root": str(Path(root).resolve()),
        "drive": drive,
        "camera": camera,
        "fixed_dt_s": fixed_dt_s,
        "fixed_dt_tolerance_s": dt_tolerance_s,
        "split": split_metadata,
        "trajectory_reset_policy": "reset before every clean and corrupted trajectory",
        "trajectories": {
            name: {
                "schedule": spec["schedule"].to_dict(),
                "corruption": spec["corruption"].to_dict(),
                "interpretation": spec["interpretation"],
            }
            for name, spec in trajectory_specs.items()
        },
        "injection_point": "after_resize_center_crop_before_imagenet_normalize",
        "randomness_key": "sha256(seed,drive,camera,absolute_frame_name)",
        "paper_claim_ready": False,
        "limitation": (
            "same-drive mechanistic diagnostic on one short continuous validation segment; "
            "not evidence of broad robustness or cross-drive generalization"
        ),
        "metric_definitions": {
            "signed_excess": "delta L_t = corrupted feature MSE - paired clean feature MSE",
            "peak_error_secondary": "maximum raw corrupted L_t from disturbance through recovery",
            "recovery_time": (
                "frames after recovery onset until signed excess remains within "
                f"{recovery_fraction:.3f} of the peak absolute excursion from baseline for "
                f"{recovery_consecutive_frames} consecutive frames"
            ),
            "signed_excess_auec": "sum signed delta L_t times fixed_dt_s",
            "absolute_excess_auec": "sum abs(delta L_t) times fixed_dt_s",
            "positive_excess_auec": "sum max(delta L_t, 0) times fixed_dt_s",
            "state_scale": "per-frame RMS(F_t) and RMS(previous completed E_t)",
            "state_utilization": "P(F_t,E_t)-P(F_t,0) on the same frame and checkpoint",
        },
        "recovery_curve_pngs": curve_paths,
        "checkpoints": checkpoint_summaries,
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Saved controlled corruption summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
