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
    return {
        "history_mode": history_mode,
        "predictor_kernel_size": configured_kernel,
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
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model.to(DEVICE)


def _tensor_rms(tensor):
    return math.sqrt(float(tensor.detach().float().square().mean().cpu().item()))


def summarize_temporal_response(records):
    phase_order = (
        "baseline",
        "step_change",
        "ramp_change",
        "persistent_bias",
        "recovery",
    )
    metric_names = (
        "corrupted_prediction_error_rms",
        "corrupted_temporal_error_input_rms",
        "corrupted_temporal_error_state_rms",
        "corrupted_feature_mse",
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


def _plot_recovery_curves(output_path, records_by_label):
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    phase_colors = {
        "step_change": "#f5c2c7",
        "ramp_change": "#ffe69c",
        "persistent_bias": "#badbcc",
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
            [record["corrupted_feature_mse"] for record in records],
            label=f"{label} corrupted",
            linewidth=1.8,
        )
        axes[0].plot(
            x_values,
            [record["clean_feature_mse"] for record in records],
            label=f"{label} clean",
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
        )
        axes[1].plot(
            x_values,
            [record["corrupted_prediction_error_rms"] for record in records],
            label=label,
            linewidth=1.8,
        )
        axes[2].plot(
            x_values,
            [record["corrupted_temporal_error_state_rms"] for record in records],
            label=label,
            linewidth=1.8,
        )

    axes[0].set_ylabel("L_t feature MSE")
    axes[1].set_ylabel("e_(t+1) RMS")
    axes[2].set_ylabel("E_(t+1) RMS")
    axes[2].set_xlabel("Absolute future raw frame index")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle("Same-drive controlled corruption and recovery")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


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

    schedule = ControlledCorruptionSchedule(
        baseline_frames=int(os.environ.get("PREDIFY_CORRUPTION_BASELINE_FRAMES", "6")),
        step_frames=int(os.environ.get("PREDIFY_CORRUPTION_STEP_FRAMES", "3")),
        ramp_frames=int(os.environ.get("PREDIFY_CORRUPTION_RAMP_FRAMES", "5")),
        persistent_frames=int(
            os.environ.get("PREDIFY_CORRUPTION_PERSISTENT_FRAMES", "7")
        ),
        recovery_frames=int(
            os.environ.get("PREDIFY_CORRUPTION_RECOVERY_FRAMES", "9")
        ),
        step_level=float(os.environ.get("PREDIFY_CORRUPTION_STEP_LEVEL", "0.5")),
    )
    corruption_config = ControlledCorruptionConfig(
        bias_rgb=_parse_float_tuple(
            os.environ.get("PREDIFY_CORRUPTION_BIAS_RGB", "0.15,-0.08,0.05"),
            3,
            "PREDIFY_CORRUPTION_BIAS_RGB",
        ),
        noise_std=float(os.environ.get("PREDIFY_CORRUPTION_NOISE_STD", "0.03")),
        seed=int(os.environ.get("PREDIFY_CORRUPTION_SEED", "0")),
    )
    recovery_fraction = float(
        os.environ.get("PREDIFY_RECOVERY_THRESHOLD_FRACTION", "0.1")
    )
    recovery_consecutive_frames = int(
        os.environ.get("PREDIFY_RECOVERY_CONSECUTIVE_FRAMES", "3")
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
        corruption_enabled=True,
    )
    _, corrupted_val, split_metadata = build_same_drive_train_val_subsets(
        corrupted_dataset,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        gap_frames=gap_frames,
    )
    schedule_start = split_metadata["val_raw_frame_range"][0]
    corrupted_dataset.schedule_start_raw_index = schedule_start
    val_sample_indices = tuple(int(index) for index in corrupted_val.indices)
    if val_sample_indices != tuple(
        range(val_sample_indices[0], val_sample_indices[-1] + 1)
    ):
        raise ValueError(
            "Controlled corruption evaluation requires one contiguous fixed-dt "
            "validation stream."
        )

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
    validation_raw_frame_count = (
        split_metadata["val_raw_frame_range"][1] - schedule_start + 1
    )
    if schedule.total_frames != validation_raw_frame_count:
        raise ValueError(
            f"Corruption schedule has {schedule.total_frames} raw frames, but the "
            f"20% validation range has {validation_raw_frame_count}; they must match."
        )

    records_by_label = {}
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
        clean_records = _evaluate_stream(model, clean_dataset, val_sample_indices)
        corrupted_records = _evaluate_stream(
            model,
            corrupted_dataset,
            val_sample_indices,
        )
        merged_records = _merge_clean_and_corrupted(clean_records, corrupted_records)
        recovery_metrics = compute_controlled_recovery_metrics(
            merged_records,
            sample_time_s=fixed_dt_s,
            recovery_fraction=recovery_fraction,
            recovery_consecutive_frames=recovery_consecutive_frames,
        )
        jsonl_path, csv_path = _write_frame_artifacts(
            output_dir,
            label,
            merged_records,
        )
        records_by_label[label] = merged_records
        checkpoint_summaries[label] = {
            "checkpoint_path": str(checkpoint_path.resolve()),
            **validation,
            "frame_jsonl": str(jsonl_path.resolve()),
            "recovery_curve_csv": str(csv_path.resolve()),
            "metrics": recovery_metrics,
            "temporal_response": summarize_temporal_response(merged_records),
        }
        del model
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    curve_path = output_dir / "controlled_corruption_recovery.png"
    _plot_recovery_curves(curve_path, records_by_label)
    summary = {
        "experiment": "kitti_same_drive_controlled_corruption",
        "git_revision": _resolve_git_revision(),
        "device": str(DEVICE),
        "kitti_root": str(Path(root).resolve()),
        "drive": drive,
        "camera": camera,
        "fixed_dt_s": fixed_dt_s,
        "fixed_dt_tolerance_s": dt_tolerance_s,
        "split": split_metadata,
        "schedule": schedule.to_dict(),
        "corruption": {
            **corruption_config.to_dict(),
            "injection_point": "after_resize_center_crop_before_imagenet_normalize",
            "randomness_key": "sha256(seed,drive,camera,absolute_frame_name)",
        },
        "metric_definitions": {
            "peak_error": "maximum corrupted L_t feature MSE from disturbance onset through recovery",
            "recovery_time": (
                "frames after recovery onset until excess MSE remains below baseline plus "
                f"{recovery_fraction:.3f} of peak excursion for "
                f"{recovery_consecutive_frames} consecutive frames"
            ),
            "auec": "sum of per-frame feature MSE times fixed_dt_s",
            "excess_auec": "sum max(corrupted MSE - paired clean MSE, 0) times fixed_dt_s",
        },
        "recovery_curve_png": str(curve_path.resolve()),
        "checkpoints": checkpoint_summaries,
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Saved controlled corruption summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
