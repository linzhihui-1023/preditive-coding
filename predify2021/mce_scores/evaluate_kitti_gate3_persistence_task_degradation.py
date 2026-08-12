import csv
import hashlib
import json
import math
import os
from collections import defaultdict, deque
from pathlib import Path

import torch

from predify2021.mce_scores.evaluate_kitti_matched_blur_persistence import (
    CONDITIONS as CORRUPTED_CONDITIONS,
    build_counterbalanced_sigma_sequences,
)
from predify2021.mce_scores.evaluate_kitti_prediction_error_separability import (
    METRICS,
    _parameter_versions,
    binary_auroc,
    build_model as build_detector_model,
    causal_error_statistics,
    validate_checkpoint as validate_detector_checkpoint,
)
from predify2021.mce_scores.kitti_controlled_corruption import (
    ControlledCorruptionConfig,
    ControlledCorruptionKITTIDataset,
    ExplicitSeveritySchedule,
)
from predify2021.mce_scores.kitti_pairs import KITTIEgoMotionMultiHorizonDataset
from predify2021.model_factory.get_model import get_model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DETECTOR_SCORE = "prediction_error_cosine_previous"
CONDITIONS = ("clean",) + CORRUPTED_CONDITIONS


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gate2_reference(summary, expected_revision):
    if summary.get("experiment") != "prediction_error_matched_blur_persistence":
        raise ValueError("Gate 3 requires the Gate 2 matched-blur result.")
    if summary.get("evaluation_git_revision") != expected_revision:
        raise ValueError("Gate 2 evaluation revision does not match the protocol lock.")
    analysis = summary.get("analysis", {})
    if analysis.get("selected_metric") != DETECTOR_SCORE:
        raise ValueError("Gate 2 did not freeze error cosine as its selected detector.")
    directions = analysis.get("score_directions_selected_on_calibration_drive", {})
    if directions.get(DETECTOR_SCORE) != 1:
        raise ValueError("Gate 2 detector direction must be higher cosine.")
    protocol = summary.get("protocol", {})
    if protocol.get("primary_analysis_unit") != "nonoverlapping 8-frame disturbance window":
        raise ValueError("Gate 3 requires Gate 2's frozen eight-frame detector window.")
    return {
        "detector_score": DETECTOR_SCORE,
        "detector_direction": "higher_is_more_persistent",
        "window_size": 8,
        "baseline_transitions": int(protocol["baseline_transitions"]),
        "disturbance_transitions": int(protocol["disturbance_transitions"]),
        "recovery_transitions": int(protocol["recovery_transitions"]),
        "blur_kernel_size": int(protocol["blur_kernel_size"]),
        "sigma_levels": tuple(float(value) for value in protocol["sigma_levels"]),
        "frames_per_sigma": int(protocol["frames_per_sigma_per_trajectory"]),
        "replicate_count": int(protocol["replicate_count"]),
        "shuffle_seed": int(protocol["shuffle_seed"]),
        "drives": (
            summary["drives"]["calibration"],
            summary["drives"]["validation"],
        ),
    }


def validate_motion_checkpoint(checkpoint, expected_revision):
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("The motion checkpoint is missing its configuration.")
    expected = {
        "git_revision": expected_revision,
        "task_aligned_target": "ego_motion",
        "motion_target_name": "longitudinal_yaw_2dof",
        "temporal_target_mode": "ego_motion",
        "target_flow_mode": "recursive",
        "error_state_mode": "ema",
        "reset_each_frame": False,
        "current_top_duplicate": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"Motion checkpoint contract mismatch at {key}: "
                f"{config.get(key)!r} != {value!r}."
            )
    if tuple(config.get("temporal_horizons", ())) != (1,):
        raise ValueError("Gate 3 requires the horizon-1 2-DoF motion checkpoint.")
    if checkpoint.get("checkpoint_kind") != "best_val_temporal_loss":
        raise ValueError("Gate 3 requires a best-validation motion checkpoint.")
    stats = config.get("motion_target_stats")
    if not isinstance(stats, dict) or stats.get("name") != "longitudinal_yaw_2dof":
        raise ValueError("The motion checkpoint lacks physical-unit target statistics.")
    return {
        "selected_epoch": checkpoint.get("selected_epoch", {}).get("epoch"),
        "target_mean": torch.tensor(stats["mean"], dtype=torch.float32),
        "target_std": torch.tensor(stats["std"], dtype=torch.float32),
        "config": config,
    }


def build_motion_model(checkpoint, validation):
    config = validation["config"]
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
        task="motion",
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(DEVICE)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _phase(stream_index, baseline_transitions, disturbance_transitions):
    if stream_index < baseline_transitions:
        return "baseline"
    if stream_index < baseline_transitions + disturbance_transitions:
        return "disturbance"
    return "recovery"


def evaluate_trajectory(
    detector,
    motion_model,
    image_dataset,
    motion_dataset,
    sample_indices,
    drive,
    condition,
    replicate,
    protocol,
    maximum_sigma,
    target_mean,
    target_std,
    ema_alpha,
    rolling_window,
):
    detector.reset()
    motion_model.reset()
    detector_versions = _parameter_versions(detector)
    motion_versions = _parameter_versions(motion_model)
    errors = deque(maxlen=rolling_window)
    previous_ema = None
    target_mean_device = target_mean.to(DEVICE)
    target_std_device = target_std.to(DEVICE)
    rows = []
    with torch.inference_mode():
        for stream_index, sample_index in enumerate(sample_indices):
            current, futures, current_name, future_names = image_dataset[sample_index]
            current = current.unsqueeze(0).to(DEVICE)
            future = futures[0].unsqueeze(0).to(DEVICE)
            raw_target = motion_dataset.get_motion_target(sample_index).unsqueeze(0).to(DEVICE)
            standardized_target = (raw_target - target_mean_device) / target_std_device

            detector.step_frame(
                current,
                top_target_provider=lambda: detector.extract_top_forward_feature(
                    future,
                    detach=True,
                ),
            )
            error = (
                detector.future_prediction_outputs["prediction_error_top"]
                .detach()
                .float()
                .squeeze(0)
            )
            errors.append(error)
            stats = causal_error_statistics(
                errors,
                previous_ema=previous_ema,
                ema_alpha=ema_alpha,
                rolling_window=rolling_window,
            )
            previous_ema = stats["prediction_error_l2_ema"]

            motion_model.step_frame(
                current,
                top_target_provider=lambda: motion_model.extract_top_forward_feature(
                    future,
                    detach=True,
                ),
                temporal_target_override=standardized_target,
            )
            physical_prediction = (
                motion_model.temporal_prediction * target_std_device
                + target_mean_device
            )
            absolute_error = (physical_prediction - raw_target).abs()[0, 0]
            prediction = physical_prediction[0, 0]
            target = raw_target[0, 0]

            metadata = image_dataset.get_sample_corruption_metadata(sample_index)
            phase = _phase(
                stream_index,
                protocol["baseline_transitions"],
                protocol["disturbance_transitions"],
            )
            disturbance_index = (
                stream_index - protocol["baseline_transitions"]
                if phase == "disturbance"
                else None
            )
            current_raw_index = int(image_dataset.valid_start_indices[sample_index])
            rows.append(
                {
                    "drive": drive,
                    "condition": condition,
                    "replicate": replicate,
                    "stream_index": stream_index,
                    "sample_index": int(sample_index),
                    "current_raw_frame_index": current_raw_index,
                    "future_raw_frame_index": current_raw_index + 1,
                    "current_frame_name": current_name,
                    "future_frame_name": future_names[0],
                    "phase": phase,
                    "disturbance_index": disturbance_index,
                    "current_blur_sigma": maximum_sigma
                    * metadata["current"]["severity"],
                    "future_blur_sigma": maximum_sigma
                    * metadata["future"][0]["severity"],
                    "forward_target_m": float(target[0].item()),
                    "yaw_target_rad": float(target[1].item()),
                    "forward_prediction_m": float(prediction[0].item()),
                    "yaw_prediction_rad": float(prediction[1].item()),
                    "forward_absolute_error_m": float(absolute_error[0].item()),
                    "yaw_absolute_error_rad": float(absolute_error[1].item()),
                    **stats,
                }
            )
    if detector_versions != _parameter_versions(detector):
        raise RuntimeError("A detector parameter changed during Gate 3 inference.")
    if motion_versions != _parameter_versions(motion_model):
        raise RuntimeError("A motion-model parameter changed during Gate 3 inference.")
    return rows


def verify_gate2_trace(rows, reference_rows, tolerance=1e-7):
    gate3 = {
        (row["drive"], row["condition"], str(row["replicate"]), str(row["stream_index"])): row
        for row in rows
        if row["condition"] in CORRUPTED_CONDITIONS
    }
    gate2 = {
        (row["drive"], row["condition"], row["replicate"], row["stream_index"]): row
        for row in reference_rows
    }
    if set(gate3) != set(gate2):
        raise ValueError("Gate 3 corrupted trajectories do not match Gate 2 frame keys.")
    maximum_delta = 0.0
    for key in gate2:
        current = gate3[key]
        reference = gate2[key]
        for field in ("current_blur_sigma", "future_blur_sigma") + METRICS:
            if current[field] is None or reference[field] == "":
                if current[field] is not None or reference[field] != "":
                    raise ValueError(f"Gate 2 null mismatch at {key}/{field}.")
                continue
            delta = abs(float(current[field]) - float(reference[field]))
            maximum_delta = max(maximum_delta, delta)
            if delta > tolerance:
                raise ValueError(
                    f"Gate 2 detector trace mismatch at {key}/{field}: {delta}."
                )
    return {
        "matched_row_count": len(gate2),
        "compared_fields": ["current_blur_sigma", "future_blur_sigma", *METRICS],
        "absolute_tolerance": tolerance,
        "maximum_absolute_delta": maximum_delta,
    }


def build_windows_and_degradations(rows, window_size, disturbance_transitions):
    grouped = defaultdict(list)
    for row in rows:
        if row["phase"] == "disturbance":
            grouped[(row["drive"], row["condition"], row["replicate"])].append(row)

    windows = []
    for (drive, condition, replicate), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: row["disturbance_index"])
        if [row["disturbance_index"] for row in ordered] != list(
            range(disturbance_transitions)
        ):
            raise ValueError("A Gate 3 trajectory has a non-contiguous disturbance.")
        for start in range(0, disturbance_transitions, window_size):
            chunk = ordered[start : start + window_size]
            windows.append(
                {
                    "drive": drive,
                    "condition": condition,
                    "replicate": replicate,
                    "window_index": start // window_size,
                    "disturbance_start_index": start,
                    "disturbance_stop_index_exclusive": start + window_size,
                    "persistence_score": sum(
                        float(row[DETECTOR_SCORE]) for row in chunk
                    )
                    / window_size,
                    "forward_mae_m": sum(
                        row["forward_absolute_error_m"] for row in chunk
                    )
                    / window_size,
                    "yaw_mae_rad": sum(
                        row["yaw_absolute_error_rad"] for row in chunk
                    )
                    / window_size,
                    "mean_future_blur_sigma": sum(
                        row["future_blur_sigma"] for row in chunk
                    )
                    / window_size,
                }
            )

    clean = {
        (row["drive"], row["window_index"]): row
        for row in windows
        if row["condition"] == "clean"
    }
    degradations = []
    for row in windows:
        if row["condition"] == "clean":
            continue
        clean_row = clean[(row["drive"], row["window_index"])]
        degradations.append(
            {
                **row,
                "clean_persistence_score": clean_row["persistence_score"],
                "persistence_score_excess_over_clean": row["persistence_score"]
                - clean_row["persistence_score"],
                "clean_forward_mae_m": clean_row["forward_mae_m"],
                "clean_yaw_mae_rad": clean_row["yaw_mae_rad"],
                "forward_mae_degradation_m": row["forward_mae_m"]
                - clean_row["forward_mae_m"],
                "yaw_mae_degradation_rad": row["yaw_mae_rad"]
                - clean_row["yaw_mae_rad"],
            }
        )
    return windows, degradations


def _pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Correlation requires equally sized inputs with at least 2 rows.")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denominator


def _average_ranks(values):
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0 + 1.0
        for position in range(start, stop):
            ranks[ordered[position]] = rank
        start = stop
    return ranks


def _association(rows, degradation_field):
    scores = [float(row["persistence_score"]) for row in rows]
    score_excess = [float(row["persistence_score_excess_over_clean"]) for row in rows]
    degradation = [float(row[degradation_field]) for row in rows]
    positive_labels = [int(value > 0.0) for value in degradation]
    auroc = None
    if len(set(positive_labels)) == 2:
        auroc = binary_auroc(scores, positive_labels)
    return {
        "window_count": len(rows),
        "mean_persistence_score": sum(scores) / len(scores),
        "mean_signed_degradation": sum(degradation) / len(degradation),
        "positive_degradation_fraction": sum(positive_labels) / len(positive_labels),
        "pearson_score_vs_signed_degradation": _pearson(scores, degradation),
        "spearman_score_vs_signed_degradation": _pearson(
            _average_ranks(scores), _average_ranks(degradation)
        ),
        "pearson_score_excess_vs_signed_degradation": _pearson(
            score_excess, degradation
        ),
        "score_auroc_for_positive_degradation": auroc,
    }


def summarize(degradations, drives):
    scopes = {"all_drives_all_corruptions": degradations}
    for drive in drives:
        scopes[f"{drive}/all_corruptions"] = [
            row for row in degradations if row["drive"] == drive
        ]
        for condition in CORRUPTED_CONDITIONS:
            scopes[f"{drive}/{condition}"] = [
                row
                for row in degradations
                if row["drive"] == drive and row["condition"] == condition
            ]
    return {
        scope: {
            "forward": _association(rows, "forward_mae_degradation_m"),
            "yaw": _association(rows, "yaw_mae_degradation_rad"),
        }
        for scope, rows in scopes.items()
    }


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    output_dir = Path(os.environ["PREDIFY_GATE3_OUTPUT_DIR"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    gate2_dir = Path(os.environ["PREDIFY_GATE2_RESULT_DIR"])
    gate2_summary_path = gate2_dir / "summary.json"
    gate2_frames_path = gate2_dir / "per_frame.csv"
    expected_gate2_revision = os.environ["PREDIFY_GATE2_GIT_REVISION"]
    expected_gate2_summary_sha = os.environ["PREDIFY_GATE2_SUMMARY_SHA256"]
    expected_gate2_frames_sha = os.environ["PREDIFY_GATE2_PER_FRAME_SHA256"]
    if _sha256(gate2_summary_path) != expected_gate2_summary_sha:
        raise ValueError("Gate 2 summary hash does not match the protocol lock.")
    if _sha256(gate2_frames_path) != expected_gate2_frames_sha:
        raise ValueError("Gate 2 per-frame hash does not match the protocol lock.")
    with gate2_summary_path.open() as handle:
        gate2_summary = json.load(handle)
    with gate2_frames_path.open() as handle:
        gate2_rows = list(csv.DictReader(handle))
    protocol = validate_gate2_reference(gate2_summary, expected_gate2_revision)

    detector_path = Path(os.environ["PREDIFY_GATE3_DETECTOR_CHECKPOINT"])
    detector_revision = os.environ["PREDIFY_DETECTOR_CHECKPOINT_GIT_REVISION"]
    detector_checkpoint = torch.load(detector_path, map_location="cpu", weights_only=False)
    detector_validation = validate_detector_checkpoint(
        detector_checkpoint, detector_revision
    )
    detector = build_detector_model(detector_checkpoint, detector_validation)

    motion_path = Path(os.environ["PREDIFY_GATE3_MOTION_CHECKPOINT"])
    motion_revision = os.environ["PREDIFY_MOTION_CHECKPOINT_GIT_REVISION"]
    motion_checkpoint = torch.load(motion_path, map_location="cpu", weights_only=False)
    motion_validation = validate_motion_checkpoint(motion_checkpoint, motion_revision)
    motion_model = build_motion_model(motion_checkpoint, motion_validation)

    sequences = build_counterbalanced_sigma_sequences(
        protocol["sigma_levels"],
        protocol["frames_per_sigma"],
        protocol["replicate_count"],
        protocol["shuffle_seed"],
    )
    maximum_sigma = max(protocol["sigma_levels"])
    transition_count = (
        protocol["baseline_transitions"]
        + protocol["disturbance_transitions"]
        + protocol["recovery_transitions"]
    )
    root = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
    ema_alpha = float(os.environ.get("PREDIFY_ERROR_NORM_EMA_ALPHA", "0.207"))
    rolling_window = int(os.environ.get("PREDIFY_ERROR_ROLLING_WINDOW", "8"))

    rows = []
    dataset_records = {}
    for drive in protocol["drives"]:
        motion_dataset = KITTIEgoMotionMultiHorizonDataset(
            root,
            drive,
            camera=camera,
            horizons=(1,),
            fixed_dt_s=fixed_dt_s,
            dt_tolerance_s=tolerance,
        )
        trajectory_specs = [("clean", -1, sequences[0]["persistent_blur"], False)]
        trajectory_specs.extend(
            (condition, replicate, condition_sequences[condition], True)
            for replicate, condition_sequences in sorted(sequences.items())
            for condition in CORRUPTED_CONDITIONS
        )
        for condition, replicate, sigma_sequence, enabled in trajectory_specs:
            schedule = ExplicitSeveritySchedule(
                baseline_frames=protocol["baseline_transitions"] + 1,
                disturbance_severities=tuple(
                    sigma / maximum_sigma for sigma in sigma_sequence
                ),
                recovery_frames=protocol["recovery_transitions"],
            )
            image_dataset = ControlledCorruptionKITTIDataset(
                root,
                drive,
                camera=camera,
                horizons=(1,),
                fixed_dt_s=fixed_dt_s,
                dt_tolerance_s=tolerance,
                schedule=schedule,
                corruption_config=ControlledCorruptionConfig(
                    corruption_type="gaussian_blur",
                    blur_kernel_size=protocol["blur_kernel_size"],
                    blur_sigma=maximum_sigma,
                ),
                schedule_start_raw_index=0,
                corruption_enabled=enabled,
            )
            if image_dataset.valid_start_indices != motion_dataset.valid_start_indices:
                raise ValueError("Image and OXTS datasets do not expose the same transitions.")
            sample_indices = [
                index
                for index, raw_start in enumerate(image_dataset.valid_start_indices)
                if int(raw_start) < transition_count
            ]
            starts = tuple(
                int(image_dataset.valid_start_indices[index]) for index in sample_indices
            )
            if starts != tuple(range(transition_count)):
                raise ValueError(f"{drive} does not supply the Gate 2 contiguous prefix.")
            dataset_records[f"{drive}/{condition}/replicate_{replicate}"] = {
                "transition_count": len(sample_indices),
                "raw_frame_range": [0, transition_count],
                "corruption_enabled": enabled,
                "time_filter_stats": image_dataset.time_filter_stats,
            }
            rows.extend(
                evaluate_trajectory(
                    detector,
                    motion_model,
                    image_dataset,
                    motion_dataset,
                    sample_indices,
                    drive,
                    condition,
                    replicate,
                    protocol,
                    maximum_sigma,
                    motion_validation["target_mean"],
                    motion_validation["target_std"],
                    ema_alpha,
                    rolling_window,
                )
            )

    trace_audit = verify_gate2_trace(rows, gate2_rows)
    windows, degradations = build_windows_and_degradations(
        rows,
        protocol["window_size"],
        protocol["disturbance_transitions"],
    )
    analysis = summarize(degradations, protocol["drives"])
    _write_csv(output_dir / "per_frame.csv", rows)
    _write_csv(output_dir / "windows.csv", windows)
    _write_csv(output_dir / "degradations.csv", degradations)

    result = {
        "experiment": "gate3_frozen_persistence_score_vs_2dof_degradation",
        "evaluation_git_revision": os.environ["PREDIFY_GIT_REVISION"],
        "device": str(DEVICE),
        "network_update": False,
        "optimizer_created": False,
        "protocol": {
            "question": (
                "Does Gate 2's frozen persistence score correspond to worse "
                "frozen 2-DoF task performance?"
            ),
            "conditions": list(CONDITIONS),
            "detector": (
                "S = mean cos(e_t,e_(t-1)) over the same nonoverlapping "
                "8-frame Gate 2 windows"
            ),
            "detector_metric_selection_frozen_from_gate2": True,
            "detector_direction_frozen_from_gate2": "higher_is_more_persistent",
            "degradation": "D = corrupted window MAE - same-frame clean window MAE",
            "degradation_is_signed": True,
            "motion_components": ["forward_displacement_m", "yaw_change_rad"],
            "clean_counterfactual": (
                "independently reset clean trajectory on identical raw frames "
                "and OXTS targets"
            ),
            "window_size": protocol["window_size"],
            "baseline_transitions": protocol["baseline_transitions"],
            "disturbance_transitions": protocol["disturbance_transitions"],
            "recovery_transitions": protocol["recovery_transitions"],
            "corrupted_trajectory_protocol_identical_to_gate2": True,
            "correlated_windows_are_diagnostic_units_not_independent_drives": True,
            "no_new_decision_threshold_was_selected": True,
        },
        "gate2_lock": {
            "result_dir": str(gate2_dir),
            "evaluation_git_revision": expected_gate2_revision,
            "summary_sha256": expected_gate2_summary_sha,
            "per_frame_sha256": expected_gate2_frames_sha,
            "trace_audit": trace_audit,
        },
        "detector_checkpoint": {
            "path": str(detector_path),
            "sha256": _sha256(detector_path),
            "git_revision": detector_revision,
            "validation": detector_validation,
        },
        "motion_checkpoint": {
            "path": str(motion_path),
            "sha256": _sha256(motion_path),
            "git_revision": motion_revision,
            "validation": {
                "selected_epoch": motion_validation["selected_epoch"],
                "target_mean": motion_validation["target_mean"].tolist(),
                "target_std": motion_validation["target_std"].tolist(),
                "checkpoint_kind": motion_checkpoint["checkpoint_kind"],
                "matrix_group": "A_inherit_recursive_ema",
            },
        },
        "drives": {
            "calibration": protocol["drives"][0],
            "validation": protocol["drives"][1],
        },
        "datasets": dataset_records,
        "analysis": analysis,
        "row_count": len(rows),
        "window_row_count": len(windows),
        "degradation_row_count": len(degradations),
        "per_frame_csv": "per_frame.csv",
        "windows_csv": "windows.csv",
        "degradations_csv": "degradations.csv",
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    heldout_scope = f"{protocol['drives'][1]}/all_corruptions"
    print(
        json.dumps(
            {
                "summary_path": str(output_dir / "summary.json"),
                "row_count": len(rows),
                "window_row_count": len(windows),
                "degradation_row_count": len(degradations),
                "gate2_maximum_trace_delta": trace_audit["maximum_absolute_delta"],
                "heldout": analysis[heldout_scope],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
