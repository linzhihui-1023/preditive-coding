import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict, deque
from itertools import groupby
from pathlib import Path

import torch

from predify2021.mce_scores.evaluate_kitti_prediction_error_separability import (
    METRICS,
    _parameter_versions,
    binary_auroc,
    build_model,
    causal_error_statistics,
    validate_checkpoint,
)
from predify2021.mce_scores.kitti_controlled_corruption import (
    ControlledCorruptionConfig,
    ControlledCorruptionKITTIDataset,
    ExplicitSeveritySchedule,
)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONDITIONS = ("persistent_blur", "shuffled_blur")


def build_counterbalanced_sigma_sequences(
    sigma_levels,
    frames_per_level,
    replicate_count,
    shuffle_seed,
):
    levels = tuple(float(value) for value in sigma_levels)
    if len(levels) < 2 or len(set(levels)) != len(levels):
        raise ValueError("sigma_levels must contain at least two unique values.")
    if any(value <= 0.0 for value in levels):
        raise ValueError("All sigma levels must be positive.")
    if frames_per_level <= 0:
        raise ValueError("frames_per_level must be positive.")
    if replicate_count != len(levels):
        raise ValueError(
            "replicate_count must equal the number of sigma levels for exact "
            "per-frame counterbalancing."
        )

    shuffled_labels = [
        level_index
        for level_index in range(len(levels))
        for _ in range(frames_per_level)
    ]
    random.Random(int(shuffle_seed)).shuffle(shuffled_labels)
    sequences = {}
    for replicate in range(replicate_count):
        persistent = tuple(
            levels[(block_index + replicate) % len(levels)]
            for block_index in range(len(levels))
            for _ in range(frames_per_level)
        )
        shuffled = tuple(
            levels[(label + replicate) % len(levels)] for label in shuffled_labels
        )
        if Counter(persistent) != Counter(shuffled):
            raise RuntimeError("Matched trajectories do not have the same sigma multiset.")
        sequences[replicate] = {
            "persistent_blur": persistent,
            "shuffled_blur": shuffled,
        }

    expected_per_frame = Counter(levels)
    frame_count = len(levels) * frames_per_level
    for condition in CONDITIONS:
        for frame_index in range(frame_count):
            exposure = Counter(
                sequences[replicate][condition][frame_index]
                for replicate in range(replicate_count)
            )
            if exposure != expected_per_frame:
                raise RuntimeError(
                    f"{condition} is not counterbalanced at frame {frame_index}."
                )
    return sequences


def temporal_organization(sequence):
    run_lengths = tuple(len(tuple(group)) for _, group in groupby(sequence))
    adjacent_equal = sum(
        left == right for left, right in zip(sequence[:-1], sequence[1:])
    )
    adjacent_deltas = [
        abs(right - left) for left, right in zip(sequence[:-1], sequence[1:])
    ]
    return {
        "frame_count": len(sequence),
        "run_count": len(run_lengths),
        "mean_run_length": sum(run_lengths) / len(run_lengths),
        "maximum_run_length": max(run_lengths),
        "adjacent_equal_fraction": adjacent_equal / (len(sequence) - 1),
        "mean_absolute_adjacent_sigma_change": (
            sum(adjacent_deltas) / len(adjacent_deltas)
        ),
    }


def _phase(stream_index, baseline_transitions, disturbance_transitions):
    if stream_index < baseline_transitions:
        return "baseline"
    if stream_index < baseline_transitions + disturbance_transitions:
        return "disturbance"
    return "recovery"


def evaluate_trajectory(
    model,
    dataset,
    sample_indices,
    drive,
    condition,
    replicate,
    baseline_transitions,
    disturbance_transitions,
    maximum_sigma,
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
            error = (
                model.future_prediction_outputs["prediction_error_top"]
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
            metadata = dataset.get_sample_corruption_metadata(sample_index)
            current_raw_index = int(dataset.valid_start_indices[sample_index])
            phase = _phase(
                stream_index,
                baseline_transitions,
                disturbance_transitions,
            )
            disturbance_index = (
                stream_index - baseline_transitions
                if phase == "disturbance"
                else None
            )
            current_sigma = maximum_sigma * metadata["current"]["scheduled_severity"]
            future_sigma = maximum_sigma * metadata["future"][0]["scheduled_severity"]
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
                    "classification_included": phase == "disturbance",
                    "classification_label": int(
                        phase == "disturbance" and condition == "persistent_blur"
                    ),
                    "current_blur_sigma": current_sigma,
                    "future_blur_sigma": future_sigma,
                    "absolute_sigma_change": abs(future_sigma - current_sigma),
                    **stats,
                }
            )
    if before_versions != _parameter_versions(model):
        raise RuntimeError("A model parameter changed during inference-only evaluation.")
    return rows


def build_nonoverlapping_windows(rows, window_size, disturbance_transitions):
    if window_size <= 0 or disturbance_transitions % window_size != 0:
        raise ValueError(
            "window_size must be positive and divide disturbance_transitions exactly."
        )
    groups = defaultdict(list)
    for row in rows:
        if row["classification_included"]:
            groups[(row["drive"], row["condition"], row["replicate"])].append(row)

    windows = []
    for (drive, condition, replicate), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: row["disturbance_index"])
        if [row["disturbance_index"] for row in ordered] != list(
            range(disturbance_transitions)
        ):
            raise ValueError("A trajectory does not contain a contiguous disturbance.")
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
                    "classification_label": int(condition == "persistent_blur"),
                    "mean_future_blur_sigma": sum(
                        row["future_blur_sigma"] for row in chunk
                    )
                    / window_size,
                    "mean_absolute_sigma_change": sum(
                        row["absolute_sigma_change"] for row in chunk
                    )
                    / window_size,
                    **{
                        metric: sum(float(row[metric]) for row in chunk) / window_size
                        for metric in METRICS
                    },
                }
            )
    return windows


def _oriented_auroc(examples, metric, direction):
    return binary_auroc(
        [direction * float(row[metric]) for row in examples],
        [int(row["classification_label"]) for row in examples],
    )


def summarize(rows, windows, calibration_drive, validation_drive):
    directions = {}
    calibration_aurocs = {}
    calibration_windows = [
        row for row in windows if row["drive"] == calibration_drive
    ]
    for metric in METRICS:
        raw_auc = _oriented_auroc(calibration_windows, metric, 1)
        directions[metric] = 1 if raw_auc >= 0.5 else -1
        calibration_aurocs[metric] = max(raw_auc, 1.0 - raw_auc)
    selected_metric = max(METRICS, key=lambda metric: calibration_aurocs[metric])

    window_aurocs = {}
    frame_aurocs = {}
    onset_excluded_window_aurocs = {}
    trajectory_mean_aurocs = {}
    for drive in (calibration_drive, validation_drive):
        drive_windows = [row for row in windows if row["drive"] == drive]
        drive_frames = [
            row
            for row in rows
            if row["drive"] == drive and row["classification_included"]
        ]
        trajectory_means = []
        trajectory_groups = defaultdict(list)
        for row in drive_frames:
            trajectory_groups[(row["condition"], row["replicate"])].append(row)
        for (condition, replicate), group in sorted(trajectory_groups.items()):
            trajectory_means.append(
                {
                    "condition": condition,
                    "replicate": replicate,
                    "classification_label": int(condition == "persistent_blur"),
                    **{
                        metric: sum(float(row[metric]) for row in group) / len(group)
                        for metric in METRICS
                    },
                }
            )
        window_aurocs[drive] = {}
        frame_aurocs[drive] = {}
        onset_excluded_window_aurocs[drive] = {}
        trajectory_mean_aurocs[drive] = {}
        for metric in METRICS:
            direction = directions[metric]
            window_aurocs[drive][metric] = _oriented_auroc(
                drive_windows, metric, direction
            )
            frame_aurocs[drive][metric] = _oriented_auroc(
                drive_frames, metric, direction
            )
            onset_excluded_window_aurocs[drive][metric] = _oriented_auroc(
                [row for row in drive_windows if row["window_index"] > 0],
                metric,
                direction,
            )
            trajectory_mean_aurocs[drive][metric] = _oriented_auroc(
                trajectory_means, metric, direction
            )

    heldout_auc = window_aurocs[validation_drive][selected_metric]
    if heldout_auc >= 0.8:
        decision = "go_promising"
    elif heldout_auc >= 0.7:
        decision = "continue_feature_design"
    elif heldout_auc >= 0.6:
        decision = "cautious"
    else:
        decision = "no_go"
    return {
        "primary_unit": "nonoverlapping disturbance window",
        "score_directions_selected_on_calibration_drive": directions,
        "calibration_primary_window_aurocs": calibration_aurocs,
        "selected_metric": selected_metric,
        "primary_window_aurocs": window_aurocs,
        "secondary_per_frame_aurocs": frame_aurocs,
        "onset_excluded_primary_window_aurocs": onset_excluded_window_aurocs,
        "trajectory_mean_aurocs": trajectory_mean_aurocs,
        "heldout_selected_metric_primary_window_auroc": heldout_auc,
        "decision": decision,
    }


def verify_sigma_invariants(sequences, levels):
    expected = Counter(levels)
    pairwise = {}
    for replicate, conditions in sorted(sequences.items()):
        persistent = conditions["persistent_blur"]
        shuffled = conditions["shuffled_blur"]
        persistent_counts = Counter(persistent)
        shuffled_counts = Counter(shuffled)
        if persistent_counts != shuffled_counts:
            raise RuntimeError(f"Sigma marginal mismatch in replicate {replicate}.")
        pairwise[str(replicate)] = {
            "exact_multiset_match": True,
            "sigma_counts": {
                str(sigma): persistent_counts[sigma] for sigma in sorted(expected)
            },
            "persistent_temporal_organization": temporal_organization(persistent),
            "shuffled_temporal_organization": temporal_organization(shuffled),
        }
    return {
        "pairwise_replicate_multisets_match": True,
        "per_absolute_frame_counterbalanced_across_replicates": True,
        "replicates": pairwise,
    }


def _checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    output_dir = Path(os.environ["PREDIFY_MATCHED_PERSISTENCE_OUTPUT_DIR"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(os.environ["PREDIFY_MATCHED_PERSISTENCE_CHECKPOINT"])
    checkpoint_revision = os.environ["PREDIFY_CHECKPOINT_GIT_REVISION"]
    evaluation_revision = os.environ["PREDIFY_GIT_REVISION"]
    root = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    drives = tuple(
        value.strip()
        for value in os.environ["PREDIFY_MATCHED_PERSISTENCE_DRIVES"].split(",")
        if value.strip()
    )
    if len(drives) != 2:
        raise ValueError("Exactly two drives are required: calibration then validation.")

    baseline_transitions = int(os.environ.get("PREDIFY_MATCHED_BASELINE_FRAMES", "40"))
    recovery_transitions = int(os.environ.get("PREDIFY_MATCHED_RECOVERY_FRAMES", "30"))
    sigma_levels = tuple(
        float(value)
        for value in os.environ.get("PREDIFY_MATCHED_BLUR_SIGMAS", "0.75,1.5,2.25,3.0").split(",")
    )
    frames_per_level = int(os.environ.get("PREDIFY_MATCHED_FRAMES_PER_SIGMA", "20"))
    replicate_count = int(
        os.environ.get("PREDIFY_MATCHED_REPLICATES", str(len(sigma_levels)))
    )
    shuffle_seed = int(os.environ.get("PREDIFY_MATCHED_SHUFFLE_SEED", "20260812"))
    window_size = int(os.environ.get("PREDIFY_MATCHED_WINDOW_SIZE", "8"))
    ema_alpha = float(os.environ.get("PREDIFY_ERROR_NORM_EMA_ALPHA", "0.207"))
    rolling_window = int(os.environ.get("PREDIFY_ERROR_ROLLING_WINDOW", "8"))
    blur_kernel = int(os.environ.get("PREDIFY_ERROR_BLUR_KERNEL_SIZE", "11"))
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))

    sequences = build_counterbalanced_sigma_sequences(
        sigma_levels,
        frames_per_level,
        replicate_count,
        shuffle_seed,
    )
    disturbance_transitions = len(sigma_levels) * frames_per_level
    transition_count = (
        baseline_transitions + disturbance_transitions + recovery_transitions
    )
    maximum_sigma = max(sigma_levels)
    sigma_invariants = verify_sigma_invariants(sequences, sigma_levels)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validation = validate_checkpoint(checkpoint, checkpoint_revision)
    model = build_model(checkpoint, validation)
    rows = []
    dataset_records = {}
    for drive in drives:
        for replicate, condition_sequences in sorted(sequences.items()):
            for condition in CONDITIONS:
                sigma_sequence = condition_sequences[condition]
                schedule = ExplicitSeveritySchedule(
                    # Transition 40 observes disturbed raw frame 41 as its future.
                    baseline_frames=baseline_transitions + 1,
                    disturbance_severities=tuple(
                        sigma / maximum_sigma for sigma in sigma_sequence
                    ),
                    recovery_frames=recovery_transitions,
                )
                dataset = ControlledCorruptionKITTIDataset(
                    root,
                    drive,
                    camera=camera,
                    horizons=(1,),
                    fixed_dt_s=fixed_dt_s,
                    dt_tolerance_s=tolerance,
                    schedule=schedule,
                    corruption_config=ControlledCorruptionConfig(
                        corruption_type="gaussian_blur",
                        blur_kernel_size=blur_kernel,
                        blur_sigma=maximum_sigma,
                    ),
                    schedule_start_raw_index=0,
                    corruption_enabled=True,
                )
                sample_indices = [
                    index
                    for index, raw_start in enumerate(dataset.valid_start_indices)
                    if int(raw_start) < transition_count
                ]
                starts = tuple(
                    int(dataset.valid_start_indices[index]) for index in sample_indices
                )
                if starts != tuple(range(transition_count)):
                    raise ValueError(
                        f"{drive} does not provide the required contiguous prefix."
                    )
                dataset_records[f"{drive}/{condition}/replicate_{replicate}"] = {
                    "transition_count": len(sample_indices),
                    "raw_frame_range": [0, transition_count],
                    "time_filter_stats": dataset.time_filter_stats,
                }
                rows.extend(
                    evaluate_trajectory(
                        model,
                        dataset,
                        sample_indices,
                        drive,
                        condition,
                        replicate,
                        baseline_transitions,
                        disturbance_transitions,
                        maximum_sigma,
                        ema_alpha,
                        rolling_window,
                    )
                )

    windows = build_nonoverlapping_windows(
        rows,
        window_size=window_size,
        disturbance_transitions=disturbance_transitions,
    )
    analysis = summarize(rows, windows, drives[0], drives[1])
    _write_csv(output_dir / "per_frame.csv", rows)
    _write_csv(output_dir / "windows.csv", windows)
    result = {
        "experiment": "prediction_error_matched_blur_persistence",
        "evaluation_git_revision": evaluation_revision,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
        "checkpoint_git_revision": checkpoint_revision,
        "checkpoint_validation": validation,
        "device": str(DEVICE),
        "network_update": False,
        "optimizer_created": False,
        "protocol": {
            "question": "Can prediction-error statistics distinguish temporal persistence when blur marginals are exactly matched?",
            "conditions": list(CONDITIONS),
            "only_primary_manipulation": "temporal ordering of the same sigma multiset",
            "baseline_transitions": baseline_transitions,
            "disturbance_transitions": disturbance_transitions,
            "recovery_transitions": recovery_transitions,
            "blur_kernel_size": blur_kernel,
            "sigma_levels": list(sigma_levels),
            "frames_per_sigma_per_trajectory": frames_per_level,
            "blur_occupancy_during_disturbance": 1.0,
            "replicate_count": replicate_count,
            "shuffle_seed": shuffle_seed,
            "counterbalancing": "Each absolute frame sees every sigma once per condition across replicates.",
            "primary_analysis_unit": f"nonoverlapping {window_size}-frame disturbance window",
            "primary_units_are_correlated_diagnostic_windows_not_independent_video_drives": True,
            "ema_alpha": ema_alpha,
            "rolling_window": rolling_window,
            "direction_and_metric_selection": "calibration drive only; frozen for validation drive",
            "decision_thresholds_are_user_defined_not_domain_standards": {
                "go_promising": ">=0.8",
                "continue_feature_design": ">=0.7 and <0.8",
                "cautious": ">=0.6 and <0.7",
                "no_go": "<0.6",
            },
        },
        "sigma_invariants": sigma_invariants,
        "drives": {"calibration": drives[0], "validation": drives[1]},
        "datasets": dataset_records,
        "analysis": analysis,
        "row_count": len(rows),
        "window_row_count": len(windows),
        "per_frame_csv": "per_frame.csv",
        "windows_csv": "windows.csv",
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "summary_path": str(output_dir / "summary.json"),
                "row_count": len(rows),
                "window_row_count": len(windows),
                "selected_metric": analysis["selected_metric"],
                "heldout_primary_window_auroc": analysis[
                    "heldout_selected_metric_primary_window_auroc"
                ],
                "decision": analysis["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
