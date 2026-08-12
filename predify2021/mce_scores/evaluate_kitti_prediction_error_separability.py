import csv
import hashlib
import json
import math
import os
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn.functional as F

from predify2021.mce_scores.kitti_controlled_corruption import (
    ControlledCorruptionConfig,
    ControlledCorruptionKITTIDataset,
    ControlledCorruptionSchedule,
)
from predify2021.model_factory.get_model import get_model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
METRICS = (
    "prediction_error_l2",
    "prediction_error_l2_ema",
    "prediction_error_cosine_previous",
    "prediction_error_temporal_variance",
)


def binary_auroc(scores, labels):
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires at least one positive and one negative.")
    favorable_pairs = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                favorable_pairs += 1.0
            elif positive == negative:
                favorable_pairs += 0.5
    return favorable_pairs / (len(positives) * len(negatives))


def causal_error_statistics(errors, previous_ema, ema_alpha, rolling_window):
    if not errors:
        raise ValueError("At least one prediction error is required.")
    current = errors[-1].detach().float()
    current_l2 = float(torch.linalg.vector_norm(current).item())
    ema = (
        current_l2
        if previous_ema is None
        else ema_alpha * current_l2 + (1.0 - ema_alpha) * previous_ema
    )
    cosine = None
    if len(errors) > 1:
        cosine = float(
            F.cosine_similarity(
                current.flatten().unsqueeze(0),
                errors[-2].detach().float().flatten().unsqueeze(0),
                dim=1,
            ).item()
        )
    window = tuple(errors)[-rolling_window:]
    stacked = torch.stack(window, dim=0)
    temporal_variance = float(stacked.var(dim=0, unbiased=False).mean().item())
    return {
        "prediction_error_l2": current_l2,
        "prediction_error_rms": math.sqrt(float(current.square().mean().item())),
        "prediction_error_l2_ema": ema,
        "prediction_error_cosine_previous": cosine,
        "prediction_error_temporal_variance": temporal_variance,
        "temporal_variance_window_count": len(window),
    }


def validate_checkpoint(checkpoint, expected_revision):
    config = checkpoint.get("config")
    if not isinstance(config, dict) or config.get("prediction_task") != "future_feature":
        raise ValueError("A future-feature checkpoint is required.")
    if config.get("future_feature_history_mode") != "temporal_error":
        raise ValueError("The go/no-go diagnostic requires temporal_error history mode.")
    prediction_form = config.get("future_feature_prediction_form", "current_residual")
    if prediction_form == "residual_Fhat_next=F_current+delta_hat":
        prediction_form = "current_residual"
    if prediction_form != "current_residual":
        raise ValueError("The diagnostic requires the original current_residual form.")
    if config.get("git_revision") != expected_revision:
        raise ValueError(
            f"Checkpoint revision {config.get('git_revision')!r} != {expected_revision!r}."
        )
    if checkpoint.get("checkpoint_kind") != "best_val_future_feature_mse":
        raise ValueError("The diagnostic requires a best-validation checkpoint.")
    return {
        "prediction_form": prediction_form,
        "history_mode": "temporal_error",
        "predictor_kernel_size": int(config["future_feature_predictor_kernel_size"]),
        "selected_epoch": checkpoint.get("selected_epoch"),
    }


def build_model(checkpoint, validation):
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
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model.to(DEVICE)


def _phase(stream_index, baseline_frames, shift_frames):
    if stream_index < baseline_frames:
        return "baseline"
    if stream_index < baseline_frames + shift_frames:
        return "disturbance"
    return "recovery"


def _parameter_versions(model):
    return tuple(parameter._version for parameter in model.parameters())


def evaluate_trajectory(
    model,
    dataset,
    sample_indices,
    condition,
    drive,
    baseline_frames,
    shift_frames,
    ema_alpha,
    rolling_window,
):
    model.reset()
    before_versions = _parameter_versions(model)
    errors = deque(maxlen=rolling_window)
    previous_ema = None
    rows = []
    with torch.inference_mode():
        for stream_index, sample_index in enumerate(sample_indices):
            current, futures, current_name, future_names = dataset[sample_index]
            current = current.unsqueeze(0).to(DEVICE)
            future = futures[0].unsqueeze(0).to(DEVICE)
            model.step_frame(
                current,
                top_target_provider=lambda: model.extract_top_forward_feature(
                    future,
                    detach=True,
                ),
            )
            output = model.future_prediction_outputs
            error = output["prediction_error_top"].detach().float().squeeze(0)
            errors.append(error)
            stats = causal_error_statistics(
                errors,
                previous_ema=previous_ema,
                ema_alpha=ema_alpha,
                rolling_window=rolling_window,
            )
            previous_ema = stats["prediction_error_l2_ema"]
            current_raw_index = int(dataset.valid_start_indices[sample_index])
            future_raw_index = current_raw_index + 1
            phase = _phase(stream_index, baseline_frames, shift_frames)
            rows.append(
                {
                    "drive": drive,
                    "condition": condition,
                    "stream_index": stream_index,
                    "sample_index": int(sample_index),
                    "current_raw_frame_index": current_raw_index,
                    "future_raw_frame_index": future_raw_index,
                    "current_frame_name": current_name,
                    "future_frame_name": future_names[0],
                    "phase": phase,
                    "classification_included": phase == "disturbance",
                    "classification_label": (
                        1 if phase == "disturbance" and condition == "persistent_blur" else 0
                    ),
                    **stats,
                }
            )
    if before_versions != _parameter_versions(model):
        raise RuntimeError("A model parameter changed during inference-only evaluation.")
    return rows


def _distribution(values):
    if not values:
        return {
            "mean": None,
            "std": None,
            "minimum": None,
            "p50": None,
            "p90": None,
            "maximum": None,
        }
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "minimum": tensor.min().item(),
        "p50": torch.quantile(tensor, 0.5).item(),
        "p90": torch.quantile(tensor, 0.9).item(),
        "maximum": tensor.max().item(),
    }


def summarize(rows, calibration_drive, validation_drive):
    phase_groups = defaultdict(list)
    for row in rows:
        phase_groups[(row["drive"], row["condition"], row["phase"])].append(row)
    phase_summary = {}
    for key, group in sorted(phase_groups.items()):
        drive, condition, phase = key
        phase_summary[f"{drive}/{condition}/{phase}"] = {
            "frame_count": len(group),
            "metrics": {
                metric: _distribution(
                    [float(row[metric]) for row in group if row[metric] is not None]
                )
                for metric in METRICS
            },
        }

    included = [row for row in rows if row["classification_included"]]
    directions = {}
    calibration_aurocs = {}
    for metric in METRICS:
        drive_rows = [row for row in included if row["drive"] == calibration_drive]
        raw_auc = binary_auroc(
            [float(row[metric]) for row in drive_rows],
            [int(row["classification_label"]) for row in drive_rows],
        )
        direction = 1 if raw_auc >= 0.5 else -1
        directions[metric] = direction
        calibration_aurocs[metric] = max(raw_auc, 1.0 - raw_auc)

    selected_metric = max(METRICS, key=lambda metric: calibration_aurocs[metric])
    aurocs = {}
    for drive in (calibration_drive, validation_drive):
        drive_rows = [row for row in included if row["drive"] == drive]
        aurocs[drive] = {}
        for metric in METRICS:
            direction = directions[metric]
            positives = [row for row in drive_rows if row["condition"] == "persistent_blur"]
            control_sets = {
                "joint_negative": [
                    row for row in drive_rows if row["condition"] != "persistent_blur"
                ],
                "clean_only": [row for row in drive_rows if row["condition"] == "clean"],
                "iid_noise_only": [
                    row for row in drive_rows if row["condition"] == "iid_noise"
                ],
            }
            aurocs[drive][metric] = {}
            for control_name, negatives in control_sets.items():
                examples = positives + negatives
                aurocs[drive][metric][control_name] = binary_auroc(
                    [direction * float(row[metric]) for row in examples],
                    [1 if row["condition"] == "persistent_blur" else 0 for row in examples],
                )

    heldout_auc = aurocs[validation_drive][selected_metric]["joint_negative"]
    if heldout_auc >= 0.8:
        decision = "go_promising"
    elif heldout_auc >= 0.7:
        decision = "continue_feature_design"
    elif heldout_auc >= 0.6:
        decision = "cautious"
    else:
        decision = "no_go"
    return {
        "phase_distributions": phase_summary,
        "score_directions_selected_on_calibration_drive": directions,
        "calibration_joint_negative_aurocs": calibration_aurocs,
        "selected_metric": selected_metric,
        "aurocs": aurocs,
        "heldout_selected_metric_joint_negative_auroc": heldout_auc,
        "decision": decision,
    }


def _checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    output_dir = Path(os.environ["PREDIFY_ERROR_SEPARABILITY_OUTPUT_DIR"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(os.environ["PREDIFY_ERROR_SEPARABILITY_CHECKPOINT"])
    expected_checkpoint_revision = os.environ["PREDIFY_CHECKPOINT_GIT_REVISION"]
    evaluation_revision = os.environ["PREDIFY_GIT_REVISION"]
    root = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    drives = tuple(
        item.strip()
        for item in os.environ["PREDIFY_ERROR_SEPARABILITY_DRIVES"].split(",")
        if item.strip()
    )
    if len(drives) != 2:
        raise ValueError("Exactly two drives are required: calibration then validation.")
    baseline_frames = int(os.environ.get("PREDIFY_ERROR_BASELINE_FRAMES", "40"))
    shift_frames = int(os.environ.get("PREDIFY_ERROR_SHIFT_FRAMES", "80"))
    recovery_frames = int(os.environ.get("PREDIFY_ERROR_RECOVERY_FRAMES", "30"))
    transition_count = baseline_frames + shift_frames + recovery_frames
    ema_alpha = float(os.environ.get("PREDIFY_ERROR_NORM_EMA_ALPHA", "0.207"))
    rolling_window = int(os.environ.get("PREDIFY_ERROR_ROLLING_WINDOW", "8"))
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
    noise_std = float(os.environ.get("PREDIFY_ERROR_IID_NOISE_STD", "0.08"))
    blur_kernel = int(os.environ.get("PREDIFY_ERROR_BLUR_KERNEL_SIZE", "11"))
    blur_sigma = float(os.environ.get("PREDIFY_ERROR_BLUR_SIGMA", "3.0"))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validation = validate_checkpoint(checkpoint, expected_checkpoint_revision)
    model = build_model(checkpoint, validation)
    schedule = ControlledCorruptionSchedule(
        trajectory="step_hold_recovery",
        # Transition 40 observes the first corrupted target at raw frame 41.
        # This yields exactly 40/80/30 prediction-error records by phase.
        baseline_frames=baseline_frames + 1,
        transition_frames=1,
        hold_frames=shift_frames - 1,
        recovery_frames=recovery_frames,
    )
    conditions = {
        "clean": (False, ControlledCorruptionConfig(corruption_type="gaussian_blur")),
        "persistent_blur": (
            True,
            ControlledCorruptionConfig(
                corruption_type="gaussian_blur",
                blur_kernel_size=blur_kernel,
                blur_sigma=blur_sigma,
            ),
        ),
        "iid_noise": (
            True,
            ControlledCorruptionConfig(
                corruption_type="iid_gaussian",
                noise_std=noise_std,
                seed=0,
            ),
        ),
    }
    rows = []
    dataset_records = {}
    for drive in drives:
        for condition, (enabled, corruption) in conditions.items():
            dataset = ControlledCorruptionKITTIDataset(
                root,
                drive,
                camera=camera,
                horizons=(1,),
                fixed_dt_s=fixed_dt_s,
                dt_tolerance_s=tolerance,
                schedule=schedule,
                corruption_config=corruption,
                schedule_start_raw_index=0,
                corruption_enabled=enabled,
            )
            sample_indices = [
                index
                for index, raw_start in enumerate(dataset.valid_start_indices)
                if int(raw_start) < transition_count
            ]
            if len(sample_indices) != transition_count:
                raise ValueError(
                    f"{drive} supplies {len(sample_indices)} valid transitions, "
                    f"expected {transition_count}."
                )
            expected_starts = tuple(range(transition_count))
            actual_starts = tuple(
                int(dataset.valid_start_indices[index]) for index in sample_indices
            )
            if actual_starts != expected_starts:
                raise ValueError(f"{drive} does not provide the required contiguous prefix.")
            dataset_records[f"{drive}/{condition}"] = {
                "transition_count": len(sample_indices),
                "raw_frame_range": [0, transition_count],
                "time_filter_stats": dataset.time_filter_stats,
            }
            rows.extend(
                evaluate_trajectory(
                    model,
                    dataset,
                    sample_indices,
                    condition=condition,
                    drive=drive,
                    baseline_frames=baseline_frames,
                    shift_frames=shift_frames,
                    ema_alpha=ema_alpha,
                    rolling_window=rolling_window,
                )
            )

    csv_path = output_dir / "per_frame.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    analysis = summarize(rows, calibration_drive=drives[0], validation_drive=drives[1])
    result = {
        "experiment": "prediction_error_persistent_shift_separability",
        "evaluation_git_revision": evaluation_revision,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
        "checkpoint_git_revision": expected_checkpoint_revision,
        "checkpoint_validation": validation,
        "device": str(DEVICE),
        "network_update": False,
        "optimizer_created": False,
        "protocol": {
            "trajectory": "clean -> corruption -> clean",
            "baseline_transitions": baseline_frames,
            "disturbance_transitions": shift_frames,
            "recovery_transitions": recovery_frames,
            "classification_frames": "matched disturbance-phase future frames only",
            "positive": "persistent_blur",
            "negative": ["clean", "iid_noise"],
            "ema_alpha": ema_alpha,
            "rolling_window": rolling_window,
            "blur_kernel_size": blur_kernel,
            "blur_sigma": blur_sigma,
            "iid_noise_std": noise_std,
            "direction_selection": "calibration drive only; frozen for validation drive",
            "decision_thresholds_are_user_defined_not_domain_standards": {
                "go_promising": ">=0.8",
                "continue_feature_design": ">=0.7 and <0.8",
                "cautious": ">=0.6 and <0.7",
                "no_go": "<0.6",
            },
        },
        "drives": {
            "calibration": drives[0],
            "validation": drives[1],
        },
        "datasets": dataset_records,
        "analysis": analysis,
        "row_count": len(rows),
        "per_frame_csv": "per_frame.csv",
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "row_count": len(rows),
                "selected_metric": analysis["selected_metric"],
                "heldout_auroc": analysis[
                    "heldout_selected_metric_joint_negative_auroc"
                ],
                "decision": analysis["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
