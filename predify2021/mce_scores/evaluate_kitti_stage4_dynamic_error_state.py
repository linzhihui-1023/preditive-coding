import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import torch

from predify2021.mce_scores.evaluate_kitti_stage4_multidrive import (
    FROZEN_TEST_DRIVES,
    VAL_DRIVES,
    build_model,
    validate_multidrive_checkpoint,
)
from predify2021.mce_scores.kitti_controlled_corruption import (
    ControlledCorruptionConfig,
    ControlledCorruptionKITTIDataset,
    ExplicitSeveritySchedule,
)
from predify2021.model_factory.targetflow.core import (
    build_temporal_prediction_error_state,
)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONDITIONS = ("persistent", "shuffled")
SCORE_FIELDS = (
    "instant_error_rms",
    "simple_scalar_ema_error_rms",
    "dynamic_error_state_rms",
)
ERROR_DEFINITION = "e_t=Fhat_t_stage4-F_t_stage4"
DYNAMIC_DEFINITION = (
    "epsilon_t=(Ts_e/tau_e)*e_t+"
    "(1-K_e*Ts_e/tau_e)*epsilon_(t-1)"
)


def binary_auroc(scores, labels):
    positives = [float(score) for score, label in zip(scores, labels) if label == 1]
    negatives = [float(score) for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires positive and negative examples.")
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            favorable += float(positive > negative) + 0.5 * float(positive == negative)
    return favorable / (len(positives) * len(negatives))


def build_counterbalanced_severity_sequences(
    severity_levels,
    frames_per_level,
    replicate_count,
    shuffle_seed,
):
    levels = tuple(float(value) for value in severity_levels)
    if len(levels) < 2 or len(set(levels)) != len(levels):
        raise ValueError("severity_levels must contain at least two unique values.")
    if any(not 0.0 < value <= 1.0 for value in levels):
        raise ValueError("Every severity level must be in (0, 1].")
    if frames_per_level <= 0:
        raise ValueError("frames_per_level must be positive.")
    if replicate_count != len(levels):
        raise ValueError(
            "replicate_count must equal the number of severity levels for exact "
            "absolute-frame counterbalancing."
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
            raise RuntimeError("Persistent and shuffled severity multisets differ.")
        sequences[replicate] = {
            "persistent": persistent,
            "shuffled": shuffled,
        }

    expected_exposure = Counter(levels)
    frame_count = len(levels) * frames_per_level
    for condition in CONDITIONS:
        for frame_index in range(frame_count):
            exposure = Counter(
                sequences[replicate][condition][frame_index]
                for replicate in range(replicate_count)
            )
            if exposure != expected_exposure:
                raise RuntimeError(
                    f"{condition} is not counterbalanced at frame {frame_index}."
                )
    return sequences


def temporal_organization(sequence):
    run_lengths = tuple(len(tuple(group)) for _, group in groupby(sequence))
    return {
        "frame_count": len(sequence),
        "run_count": len(run_lengths),
        "mean_run_length": sum(run_lengths) / len(run_lengths),
        "maximum_run_length": max(run_lengths),
        "adjacent_equal_fraction": sum(
            left == right for left, right in zip(sequence[:-1], sequence[1:])
        )
        / (len(sequence) - 1),
    }


def update_error_score_states(
    instantaneous_error,
    dynamic_state,
    previous_scalar_ema,
    previous_tensor_ema,
    ema_alpha,
):
    error = instantaneous_error.detach().float()
    dynamic = dynamic_state.detach().float()
    instant_rms = math.sqrt(float(error.square().mean().item()))
    scalar_ema = (
        float(ema_alpha) * instant_rms
        + (1.0 - float(ema_alpha)) * float(previous_scalar_ema)
    )
    tensor_previous = (
        torch.zeros_like(error)
        if previous_tensor_ema is None
        else previous_tensor_ema.to(error.device, error.dtype)
    )
    tensor_ema = (
        float(ema_alpha) * error
        + (1.0 - float(ema_alpha)) * tensor_previous
    ).detach()
    return (
        {
            "instant_error_rms": instant_rms,
            "simple_scalar_ema_error_rms": scalar_ema,
            "matched_tensor_ema_error_rms": math.sqrt(
                float(tensor_ema.square().mean().item())
            ),
            "dynamic_error_state_rms": math.sqrt(
                float(dynamic.square().mean().item())
            ),
            "dynamic_minus_matched_tensor_ema_max_abs": float(
                (dynamic - tensor_ema).abs().max().item()
            ),
        },
        scalar_ema,
        tensor_ema,
    )


def _phase(stream_index, baseline_transitions, disturbance_transitions):
    if stream_index < baseline_transitions:
        return "baseline"
    if stream_index < baseline_transitions + disturbance_transitions:
        return "disturbance"
    return "recovery"


def _parameter_versions(model):
    return tuple(parameter._version for parameter in model.parameters())


def evaluate_trajectory(
    model,
    dataset,
    sample_indices,
    drive,
    corruption,
    condition,
    replicate,
    baseline_transitions,
    disturbance_transitions,
    sample_time,
    time_constant,
    error_gain,
):
    model.reset()
    before_versions = _parameter_versions(model)
    integration_factor = float(sample_time) / float(time_constant)
    previous_scalar_ema = 0.0
    previous_tensor_ema = None
    previous_dynamic_state = None
    rows = []
    with torch.inference_mode():
        for stream_index, sample_index in enumerate(sample_indices):
            current, futures, current_name, future_names = dataset[sample_index]
            current = current.unsqueeze(0).to(DEVICE)
            future = futures[0].unsqueeze(0).to(DEVICE)
            target_cache = {}

            def resolve_future_stage(stage):
                if not target_cache:
                    target_cache.update(
                        model.extract_forward_features_at_stages(
                            future,
                            stages=(model.number_of_layers, model.future_feature_stage),
                            detach=True,
                        )
                    )
                return target_cache[int(stage)]

            model.step_frame(
                current,
                top_target_provider=lambda: resolve_future_stage(model.number_of_layers),
                future_feature_target_provider=lambda: resolve_future_stage(
                    model.future_feature_stage
                ),
            )
            outputs = model.future_prediction_outputs
            error = outputs["instantaneous_prediction_error_feature"]
            target = outputs["future_prediction_target"]
            predicted = outputs["predicted_future_feature"]
            if not torch.equal(error, (predicted - target).detach()):
                raise RuntimeError("Runtime prediction-error sign is not Fhat-F.")
            dynamic_state = outputs["dynamic_prediction_error_state_feature"]
            expected_dynamic = build_temporal_prediction_error_state(
                error,
                previous_dynamic_state,
                sample_time=sample_time,
                time_constant=time_constant,
                error_gain=error_gain,
            ).detach()
            dynamic_formula_max_abs = float(
                (dynamic_state - expected_dynamic).abs().max().item()
            )
            if dynamic_formula_max_abs > 1e-6:
                raise RuntimeError("Dynamic prediction-error state formula mismatch.")
            if outputs["temporal_error_state_update_index"] != stream_index + 1:
                raise RuntimeError("Temporal error state did not update exactly once.")

            stats, previous_scalar_ema, previous_tensor_ema = update_error_score_states(
                error,
                dynamic_state,
                previous_scalar_ema,
                previous_tensor_ema,
                integration_factor,
            )
            metadata = dataset.get_sample_corruption_metadata(sample_index)
            current_raw_index = int(dataset.valid_start_indices[sample_index])
            phase = _phase(
                stream_index,
                baseline_transitions,
                disturbance_transitions,
            )
            rows.append(
                {
                    "drive": drive,
                    "corruption": corruption,
                    "condition": condition,
                    "replicate": replicate,
                    "stream_index": stream_index,
                    "sample_index": int(sample_index),
                    "current_raw_frame_index": current_raw_index,
                    "future_raw_frame_index": current_raw_index + 1,
                    "current_frame_name": current_name,
                    "future_frame_name": future_names[0],
                    "phase": phase,
                    "disturbance_index": (
                        stream_index - baseline_transitions
                        if phase == "disturbance"
                        else ""
                    ),
                    "classification_included": phase == "disturbance",
                    "classification_label": int(
                        phase == "disturbance" and condition == "persistent"
                    ),
                    "current_severity": metadata["current"]["scheduled_severity"],
                    "future_severity": metadata["future"][0]["scheduled_severity"],
                    "future_feature_stage": model.future_feature_stage,
                    "error_definition": ERROR_DEFINITION,
                    "dynamic_error_definition": DYNAMIC_DEFINITION,
                    "temporal_error_state_update_index": outputs[
                        "temporal_error_state_update_index"
                    ],
                    "dynamic_formula_max_abs": dynamic_formula_max_abs,
                    **stats,
                }
            )
            previous_dynamic_state = dynamic_state.detach()
    if before_versions != _parameter_versions(model):
        raise RuntimeError("A model parameter changed during inference-only evaluation.")
    return rows


def build_nonoverlapping_windows(rows, window_size, disturbance_transitions):
    if window_size <= 0 or disturbance_transitions % window_size != 0:
        raise ValueError(
            "window_size must be positive and divide disturbance_transitions."
        )
    groups = defaultdict(list)
    for row in rows:
        if row["classification_included"]:
            groups[
                (
                    row["drive"],
                    row["corruption"],
                    row["condition"],
                    row["replicate"],
                )
            ].append(row)

    windows = []
    for (drive, corruption, condition, replicate), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: int(row["disturbance_index"]))
        if [int(row["disturbance_index"]) for row in ordered] != list(
            range(disturbance_transitions)
        ):
            raise ValueError("A trajectory lacks a complete disturbance sequence.")
        for start in range(0, disturbance_transitions, window_size):
            chunk = ordered[start : start + window_size]
            windows.append(
                {
                    "drive": drive,
                    "corruption": corruption,
                    "condition": condition,
                    "replicate": replicate,
                    "window_index": start // window_size,
                    "disturbance_start_index": start,
                    "disturbance_stop_index_exclusive": start + window_size,
                    "classification_label": int(condition == "persistent"),
                    "mean_future_severity": sum(
                        float(row["future_severity"]) for row in chunk
                    )
                    / window_size,
                    **{
                        field: sum(float(row[field]) for row in chunk) / window_size
                        for field in (
                            *SCORE_FIELDS,
                            "matched_tensor_ema_error_rms",
                        )
                    },
                }
            )
    return windows


def _distribution(values):
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    return {
        "count": tensor.numel(),
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "minimum": tensor.min().item(),
        "maximum": tensor.max().item(),
    }


def summarize(rows, windows, drives, corruptions):
    scopes = {"all_drives_all_corruptions": list(windows)}
    for drive in drives:
        scopes[f"{drive}/all_corruptions"] = [
            row for row in windows if row["drive"] == drive
        ]
        for corruption in corruptions:
            scopes[f"{drive}/{corruption}"] = [
                row
                for row in windows
                if row["drive"] == drive and row["corruption"] == corruption
            ]
    for corruption in corruptions:
        scopes[f"all_drives/{corruption}"] = [
            row for row in windows if row["corruption"] == corruption
        ]

    directional_aurocs = {}
    separability_aurocs = {}
    score_contrasts = {}
    for scope, examples in scopes.items():
        labels = [int(row["classification_label"]) for row in examples]
        directional_aurocs[scope] = {
            field: binary_auroc([row[field] for row in examples], labels)
            for field in SCORE_FIELDS
        }
        separability_aurocs[scope] = {
            field: max(auc, 1.0 - auc)
            for field, auc in directional_aurocs[scope].items()
        }
        positives = [
            row for row in examples if int(row["classification_label"]) == 1
        ]
        negatives = [
            row for row in examples if int(row["classification_label"]) == 0
        ]
        score_contrasts[scope] = {
            field: (
                sum(float(row[field]) for row in positives) / len(positives)
                - sum(float(row[field]) for row in negatives) / len(negatives)
            )
            for field in SCORE_FIELDS
        }

    phase_groups = defaultdict(list)
    for row in rows:
        phase_groups[
            (row["drive"], row["corruption"], row["condition"], row["phase"])
        ].append(row)
    phase_distributions = {}
    for key, group in sorted(phase_groups.items()):
        phase_distributions["/".join(key)] = {
            field: _distribution(float(row[field]) for row in group)
            for field in SCORE_FIELDS
        }

    primary_directional = directional_aurocs["all_drives_all_corruptions"]
    primary_separability = separability_aurocs["all_drives_all_corruptions"]
    dynamic_separability = primary_separability["dynamic_error_state_rms"]
    dynamic_better = (
        dynamic_separability > primary_separability["instant_error_rms"]
        and dynamic_separability
        > primary_separability["simple_scalar_ema_error_rms"]
    )
    return {
        "primary_unit": "nonoverlapping disturbance window",
        "score_direction": "higher_means_more_persistent",
        "no_score_or_direction_selection": True,
        "higher_is_persistent_aurocs": directional_aurocs,
        "direction_independent_separability_aurocs": separability_aurocs,
        "persistent_minus_shuffled_score_contrasts": score_contrasts,
        "phase_distributions": phase_distributions,
        "primary_aggregate": {
            "higher_is_persistent_aurocs": primary_directional,
            "direction_independent_separability_aurocs": primary_separability,
            "dynamic_minus_instant_separability_auroc": (
                dynamic_separability - primary_separability["instant_error_rms"]
            ),
            "dynamic_minus_simple_scalar_ema_separability_auroc": (
                dynamic_separability
                - primary_separability["simple_scalar_ema_error_rms"]
            ),
            "dynamic_strictly_better_than_both_primary_controls": dynamic_better,
            "phase1_decision": (
                "go_dynamic_state_better_than_controls"
                if dynamic_better
                else "no_go_dynamic_state_not_better_than_controls"
            ),
        },
        "maximum_dynamic_formula_error": max(
            float(row["dynamic_formula_max_abs"]) for row in rows
        ),
        "maximum_dynamic_minus_matched_tensor_ema_abs": max(
            float(row["dynamic_minus_matched_tensor_ema_max_abs"])
            for row in rows
        ),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    output_dir = Path(os.environ["PREDIFY_STAGE4_ERROR_STATE_OUTPUT_DIR"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(os.environ["PREDIFY_STAGE4_ERROR_STATE_CHECKPOINT"])
    checkpoint_revision = os.environ["PREDIFY_CHECKPOINT_GIT_REVISION"]
    evaluation_revision = os.environ["PREDIFY_GIT_REVISION"]
    root = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    drives = tuple(
        value.strip()
        for value in os.environ["PREDIFY_STAGE4_ERROR_STATE_DRIVES"].split(",")
        if value.strip()
    )
    if drives != VAL_DRIVES:
        raise ValueError("This diagnostic is locked to Val drives 0011 and 0039.")
    if set(drives) & set(FROZEN_TEST_DRIVES):
        raise ValueError("Frozen Test drives must not enter the error-state diagnostic.")

    baseline_transitions = int(os.environ.get("PREDIFY_ERROR_BASELINE_FRAMES", "40"))
    recovery_transitions = int(os.environ.get("PREDIFY_ERROR_RECOVERY_FRAMES", "40"))
    severity_levels = tuple(
        float(value)
        for value in os.environ.get(
            "PREDIFY_ERROR_SEVERITY_LEVELS", "0.25,0.5,0.75,1.0"
        ).split(",")
    )
    frames_per_level = int(os.environ.get("PREDIFY_ERROR_FRAMES_PER_LEVEL", "20"))
    replicate_count = int(os.environ.get("PREDIFY_ERROR_REPLICATES", "4"))
    shuffle_seed = int(os.environ.get("PREDIFY_ERROR_SHUFFLE_SEED", "20260813"))
    window_size = int(os.environ.get("PREDIFY_ERROR_WINDOW_SIZE", "8"))
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validation = validate_multidrive_checkpoint(checkpoint, checkpoint_revision)
    config = checkpoint["config"]
    if validation["history_mode"] != "aligned_difference":
        raise ValueError("The dynamic state must not feed the Stage-4 predictor.")
    sample_time = float(config["temporal_error_sample_time"])
    time_constant = float(config["temporal_error_time_constant"])
    error_gain = float(config["temporal_error_gain"])
    integration_factor = sample_time / time_constant

    sequences = build_counterbalanced_severity_sequences(
        severity_levels,
        frames_per_level,
        replicate_count,
        shuffle_seed,
    )
    disturbance_transitions = len(severity_levels) * frames_per_level
    transition_count = (
        baseline_transitions + disturbance_transitions + recovery_transitions
    )
    corruption_configs = {
        "gaussian_blur": ControlledCorruptionConfig(
            corruption_type="gaussian_blur",
            blur_kernel_size=11,
            blur_sigma=3.0,
        ),
        "iid_gaussian_noise": ControlledCorruptionConfig(
            corruption_type="iid_gaussian",
            noise_std=0.08,
            seed=0,
        ),
        "rgb_bias_domain_proxy": ControlledCorruptionConfig(
            corruption_type="bias",
            bias_rgb=(0.15, -0.08, 0.05),
        ),
    }

    model = build_model(checkpoint, validation)
    rows = []
    dataset_records = {}
    for drive in drives:
        for corruption, corruption_config in corruption_configs.items():
            for replicate, condition_sequences in sorted(sequences.items()):
                for condition in CONDITIONS:
                    schedule = ExplicitSeveritySchedule(
                        baseline_frames=baseline_transitions + 1,
                        disturbance_severities=condition_sequences[condition],
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
                        corruption_config=corruption_config,
                        schedule_start_raw_index=0,
                        corruption_enabled=True,
                    )
                    sample_indices = [
                        index
                        for index, raw_start in enumerate(dataset.valid_start_indices)
                        if int(raw_start) < transition_count
                    ]
                    starts = tuple(
                        int(dataset.valid_start_indices[index])
                        for index in sample_indices
                    )
                    if starts != tuple(range(transition_count)):
                        raise ValueError(
                            f"{drive} does not provide the required contiguous prefix."
                        )
                    record_key = (
                        f"{drive}/{corruption}/{condition}/replicate_{replicate}"
                    )
                    dataset_records[record_key] = {
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
                            corruption,
                            condition,
                            replicate,
                            baseline_transitions,
                            disturbance_transitions,
                            sample_time,
                            time_constant,
                            error_gain,
                        )
                    )

    windows = build_nonoverlapping_windows(
        rows,
        window_size=window_size,
        disturbance_transitions=disturbance_transitions,
    )
    analysis = summarize(rows, windows, drives, tuple(corruption_configs))
    organization = {
        str(replicate): {
            condition: temporal_organization(sequence)
            for condition, sequence in conditions.items()
        }
        for replicate, conditions in sequences.items()
    }
    _write_csv(output_dir / "per_frame.csv", rows)
    _write_csv(output_dir / "windows.csv", windows)
    result = {
        "experiment": "stage4_dynamic_prediction_error_state_phase1",
        "evaluation_git_revision": evaluation_revision,
        "checkpoint_git_revision": checkpoint_revision,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_validation": validation,
        "device": str(DEVICE),
        "network_update": False,
        "optimizer_created": False,
        "predictor_uses_dynamic_error_state": False,
        "frozen_test_read": False,
        "error_definition": ERROR_DEFINITION,
        "dynamic_error_definition": DYNAMIC_DEFINITION,
        "protocol": {
            "question": (
                "Does the dynamic prediction-error state distinguish persistent "
                "failure better than instantaneous error and a scalar EMA envelope?"
            ),
            "drives": list(drives),
            "drive_role": "existing_validation_drives_mechanistic_diagnostic_only",
            "corruptions": list(corruption_configs),
            "rgb_bias_is_domain_shift_proxy_not_a_full_domain_dataset": True,
            "conditions": list(CONDITIONS),
            "only_condition_difference": "temporal_order_of_same_severity_multiset",
            "baseline_transitions": baseline_transitions,
            "disturbance_transitions": disturbance_transitions,
            "recovery_transitions": recovery_transitions,
            "severity_levels": list(severity_levels),
            "frames_per_level": frames_per_level,
            "replicate_count": replicate_count,
            "shuffle_seed": shuffle_seed,
            "window_size": window_size,
            "score_fields": list(SCORE_FIELDS),
            "simple_ema_definition": (
                "s_t=alpha*RMS(e_t)+(1-alpha)*s_(t-1); s_reset=0"
            ),
            "matched_tensor_ema_is_equivalence_audit_not_primary_control": True,
            "sample_time": sample_time,
            "time_constant": time_constant,
            "error_gain": error_gain,
            "integration_factor": integration_factor,
            "memory_factor": 1.0 - error_gain * integration_factor,
            "one_state_update_per_real_video_transition": True,
            "state_detached_between_transitions": True,
            "same_frame_internal_iterations": 0,
        },
        "temporal_organization": organization,
        "datasets": dataset_records,
        "analysis": analysis,
        "row_count": len(rows),
        "window_row_count": len(windows),
        "per_frame_csv": "per_frame.csv",
        "windows_csv": "windows.csv",
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "row_count": len(rows),
                "window_row_count": len(windows),
                "primary_aggregate": analysis["primary_aggregate"],
                "maximum_dynamic_minus_matched_tensor_ema_abs": analysis[
                    "maximum_dynamic_minus_matched_tensor_ema_abs"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
