import copy
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch

from predify2021.mce_scores.kitti_controlled_corruption import (
    ControlledCorruptionConfig,
    ControlledCorruptionKITTIDataset,
    ExplicitSeveritySchedule,
)
from predify2021.model_factory import get_model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONDITIONS = ("feedforward", "pc_no_error", "pc_dynamic_error")
ALLOWED_DRIVES = (
    "2011_09_26/2011_09_26_drive_0011_sync",
    "2011_09_26/2011_09_26_drive_0039_sync",
)
FORBIDDEN_TEST_DRIVE_IDS = ("drive_0051_sync", "drive_0056_sync")
BASELINE_FRAMES = 40
DISTURBANCE_FRAMES = 80
RECOVERY_FRAMES = 40
TOTAL_FRAMES = BASELINE_FRAMES + DISTURBANCE_FRAMES + RECOVERY_FRAMES
BLUR_KERNEL_SIZE = 11
BLUR_SIGMA = 3.0
PRACTICAL_IMPROVEMENT_THRESHOLD = 0.05
RECOVERY_WINDOW_FRAMES = 10


def select_protocol_raw_frames(dataset, total_frames=TOTAL_FRAMES):
    """Select the first contiguous accepted segment with enough real frames."""
    for sample_segment in dataset.valid_sample_segments:
        starts = tuple(
            int(dataset.valid_start_indices[sample_index])
            for sample_index in sample_segment
        )
        if not starts:
            continue
        if any(right != left + 1 for left, right in zip(starts[:-1], starts[1:])):
            raise RuntimeError("A declared KITTI segment is not contiguous.")
        raw_frames = (*starts, starts[-1] + 1)
        if len(raw_frames) >= total_frames:
            return tuple(raw_frames[:total_frames])
    raise ValueError(
        f"No contiguous KITTI segment contains {total_frames} real frames."
    )


def normalized_representation_distance(corrupted, clean):
    corrupted_flat = corrupted.detach().double().flatten(1)
    clean_flat = clean.detach().double().flatten(1)
    difference_norm = torch.linalg.vector_norm(
        corrupted_flat - clean_flat,
        dim=1,
    )
    clean_norm = torch.linalg.vector_norm(clean_flat, dim=1).clamp_min(1e-12)
    return float(torch.mean(difference_norm / clean_norm).item())


def tensor_rms(tensor):
    value = tensor.detach().double()
    return float(torch.sqrt(torch.mean(value.square())).item())


def parameter_versions(model):
    return tuple(parameter._version for parameter in model.parameters())


def assert_frozen_and_detached(model):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("A Phase-1 model parameter requires gradients.")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("A Phase-1 model parameter received a gradient.")
    memories = (
        model.representation_state_memory
        + model.prediction_state_memory
        + model.instant_error_state_memory
        + model.error_state_memory
    )
    if any(memory is None for memory in memories):
        raise RuntimeError("A cross-frame state memory was not populated.")
    if any(memory.requires_grad or memory.grad_fn is not None for memory in memories):
        raise RuntimeError("A cross-frame state memory was not detached.")
    for state in model.layer_states:
        previous = (
            torch.zeros_like(state.instant_error)
            if state.previous_dynamic_error is None
            else state.previous_dynamic_error
        )
        expected = 0.207 * state.instant_error + 0.793 * previous
        if float((state.dynamic_error - expected).abs().max().item()) > 1e-6:
            raise RuntimeError("The formal dynamic-error recurrence changed.")


def phase_for_frame(frame_offset):
    if frame_offset < BASELINE_FRAMES:
        return "baseline", frame_offset
    if frame_offset < BASELINE_FRAMES + DISTURBANCE_FRAMES:
        return "disturbance", frame_offset - BASELINE_FRAMES
    return "recovery", frame_offset - BASELINE_FRAMES - DISTURBANCE_FRAMES


def build_base_model(weights_path):
    model = get_model(
        "pvgg_tf",
        pretrained=True,
        pcoder_weights=weights_path,
        task="real_frame_pc",
        dynamic_error=True,
        error_state_mode="ema",
        error_sample_time=0.1035,
        error_time_constant=(0.5, 0.5, 0.5, 0.5, 0.5),
        error_gain=(1.0, 1.0, 1.0, 1.0, 1.0),
        pc_ff_multiplier=(0.2, 0.4, 0.4, 0.5, 0.6),
        pc_fb_multiplier=(0.05, 0.1, 0.1, 0.1, 0.0),
        pc_error_multiplier=(0.01, 0.01, 0.01, 0.01, 0.01),
    ).eval()
    if model.temporal_predictor is not None or model.future_feature_predictor is not None:
        raise RuntimeError("Phase 1 must not construct a future predictor.")
    if model.temporal_fusion_module is not None:
        raise RuntimeError("Phase 1 must not construct temporal fusion.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Phase 1 requires every parameter to be frozen.")
    return model


def build_condition_models(base_model, condition):
    clean_model = copy.deepcopy(base_model).to(DEVICE).eval()
    corrupted_model = copy.deepcopy(base_model).to(DEVICE).eval()
    error_multiplier = 0.0 if condition == "pc_no_error" else 0.01
    with torch.no_grad():
        clean_model.pc_error_multipliers.fill_(error_multiplier)
        corrupted_model.pc_error_multipliers.fill_(error_multiplier)
    return clean_model, corrupted_model


def build_drive_datasets(root, drive, camera, fixed_dt_s, tolerance):
    schedule = ExplicitSeveritySchedule(
        baseline_frames=BASELINE_FRAMES,
        disturbance_severities=(1.0,) * DISTURBANCE_FRAMES,
        recovery_frames=RECOVERY_FRAMES,
    )
    config = ControlledCorruptionConfig(
        corruption_type="gaussian_blur",
        blur_kernel_size=BLUR_KERNEL_SIZE,
        blur_sigma=BLUR_SIGMA,
        seed=0,
    )
    probe = ControlledCorruptionKITTIDataset(
        root,
        drive,
        camera=camera,
        horizons=(1,),
        fixed_dt_s=fixed_dt_s,
        dt_tolerance_s=tolerance,
        schedule=schedule,
        corruption_config=config,
        corruption_enabled=False,
    )
    raw_frames = select_protocol_raw_frames(probe)
    schedule_start = raw_frames[0]
    clean = ControlledCorruptionKITTIDataset(
        root,
        drive,
        camera=camera,
        horizons=(1,),
        fixed_dt_s=fixed_dt_s,
        dt_tolerance_s=tolerance,
        schedule=schedule,
        corruption_config=config,
        schedule_start_raw_index=schedule_start,
        corruption_enabled=False,
    )
    corrupted = ControlledCorruptionKITTIDataset(
        root,
        drive,
        camera=camera,
        horizons=(1,),
        fixed_dt_s=fixed_dt_s,
        dt_tolerance_s=tolerance,
        schedule=schedule,
        corruption_config=config,
        schedule_start_raw_index=schedule_start,
        corruption_enabled=True,
    )
    if clean.frame_paths != corrupted.frame_paths:
        raise RuntimeError("Clean and corrupted datasets do not share raw frames.")
    return clean, corrupted, raw_frames


def evaluate_condition(base_model, condition, datasets, seed):
    clean_model, corrupted_model = build_condition_models(base_model, condition)
    clean_versions = parameter_versions(clean_model)
    corrupted_versions = parameter_versions(corrupted_model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    c_sqrt_initialized = False
    rows = []
    baseline_max_distance = 0.0

    for drive, (clean_dataset, corrupted_dataset, raw_frames) in datasets.items():
        clean_model.reset()
        corrupted_model.reset()
        for frame_offset, raw_frame_index in enumerate(raw_frames):
            if condition == "feedforward":
                clean_model.reset()
                corrupted_model.reset()

            clean_frame = clean_dataset._load_frame(
                clean_dataset.frame_paths[raw_frame_index]
            ).unsqueeze(0).to(DEVICE)
            corrupted_frame = corrupted_dataset._load_frame(
                corrupted_dataset.frame_paths[raw_frame_index]
            ).unsqueeze(0).to(DEVICE)

            clean_model.step_frame(clean_frame)
            if not c_sqrt_initialized:
                with torch.no_grad():
                    corrupted_model.pc_error_c_sqrt.copy_(
                        clean_model.pc_error_c_sqrt
                    )
                c_sqrt_initialized = True
            corrupted_model.step_frame(corrupted_frame)

            expected_update_index = 1 if condition == "feedforward" else frame_offset + 1
            for model in (clean_model, corrupted_model):
                if model.real_frame_update_count != expected_update_index:
                    raise RuntimeError("A real frame did not produce exactly one update.")
                if model.recurrence_outputs["updates_per_layer"] != (1, 1, 1, 1, 1):
                    raise RuntimeError("A PCoder layer updated more than once.")
                if model.recurrence_outputs["used_future_frame"]:
                    raise RuntimeError("The recurrence reported future-frame use.")
                assert_frozen_and_detached(model)

            if not torch.equal(
                clean_model.pc_error_c_sqrt,
                corrupted_model.pc_error_c_sqrt,
            ):
                raise RuntimeError("Clean and corrupted C_sqrt controls diverged.")

            phase, phase_frame_index = phase_for_frame(frame_offset)
            metadata = corrupted_dataset.get_frame_corruption_metadata(
                raw_frame_index
            )
            if metadata["phase"] != phase:
                raise RuntimeError("Corruption metadata phase does not match protocol.")
            for clean_state, corrupted_state in zip(
                clean_model.layer_states,
                corrupted_model.layer_states,
            ):
                distance = normalized_representation_distance(
                    corrupted_state.representation,
                    clean_state.representation,
                )
                if phase == "baseline":
                    baseline_max_distance = max(baseline_max_distance, distance)
                rows.append(
                    {
                        "condition": condition,
                        "drive": drive,
                        "frame_offset": frame_offset,
                        "phase": phase,
                        "phase_frame_index": phase_frame_index,
                        "raw_frame_index": raw_frame_index,
                        "frame_name": clean_dataset.frame_paths[raw_frame_index].name,
                        "blur_sigma": BLUR_SIGMA if phase == "disturbance" else 0.0,
                        "layer": clean_state.layer_index,
                        "internal_frame_index": clean_state.frame_index,
                        "instantaneous_prediction_error_rms": tensor_rms(
                            corrupted_state.instant_error
                        ),
                        "dynamic_error_rms": tensor_rms(
                            corrupted_state.dynamic_error
                        ),
                        "clean_instantaneous_prediction_error_rms": tensor_rms(
                            clean_state.instant_error
                        ),
                        "clean_dynamic_error_rms": tensor_rms(
                            clean_state.dynamic_error
                        ),
                        "representation_normalized_l2": distance,
                    }
                )

            del clean_frame, corrupted_frame

    if baseline_max_distance > 1e-10:
        raise RuntimeError(
            f"Matched baseline streams diverged: max distance={baseline_max_distance}."
        )
    if parameter_versions(clean_model) != clean_versions:
        raise RuntimeError("Clean model parameters changed during Phase 1.")
    if parameter_versions(corrupted_model) != corrupted_versions:
        raise RuntimeError("Corrupted model parameters changed during Phase 1.")
    c_sqrt = [float(value) for value in clean_model.pc_error_c_sqrt.cpu().tolist()]
    del clean_model, corrupted_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return rows, c_sqrt


def mean_field(rows, field):
    if not rows:
        raise ValueError(f"Cannot average empty rows for {field}.")
    return sum(float(row[field]) for row in rows) / len(rows)


def summarize_rows(rows):
    condition_summary = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        phases = {}
        for phase in ("baseline", "disturbance", "recovery"):
            phase_rows = [row for row in condition_rows if row["phase"] == phase]
            phases[phase] = {
                "frame_layer_count": len(phase_rows),
                "mean_representation_normalized_l2": mean_field(
                    phase_rows,
                    "representation_normalized_l2",
                ),
                "mean_instantaneous_prediction_error_rms": mean_field(
                    phase_rows,
                    "instantaneous_prediction_error_rms",
                ),
                "mean_dynamic_error_rms": mean_field(
                    phase_rows,
                    "dynamic_error_rms",
                ),
            }

        per_layer = {}
        per_drive = {}
        for layer in range(1, 6):
            layer_rows = [
                row
                for row in condition_rows
                if row["phase"] == "disturbance" and row["layer"] == layer
            ]
            per_layer[str(layer)] = mean_field(
                layer_rows,
                "representation_normalized_l2",
            )
        for drive in ALLOWED_DRIVES:
            drive_rows = [
                row
                for row in condition_rows
                if row["phase"] == "disturbance" and row["drive"] == drive
            ]
            per_drive[drive] = mean_field(
                drive_rows,
                "representation_normalized_l2",
            )

        recovery_rows = [
            row for row in condition_rows if row["phase"] == "recovery"
        ]
        recovery_first = [
            row
            for row in recovery_rows
            if row["phase_frame_index"] < RECOVERY_WINDOW_FRAMES
        ]
        recovery_last = [
            row
            for row in recovery_rows
            if row["phase_frame_index"] >= RECOVERY_FRAMES - RECOVERY_WINDOW_FRAMES
        ]
        first_mean = mean_field(recovery_first, "representation_normalized_l2")
        last_mean = mean_field(recovery_last, "representation_normalized_l2")
        condition_summary[condition] = {
            "phases": phases,
            "disturbance_per_layer_mean_normalized_l2": per_layer,
            "disturbance_per_drive_mean_normalized_l2": per_drive,
            "recovery_first_10_mean_normalized_l2": first_mean,
            "recovery_last_10_mean_normalized_l2": last_mean,
            "recovery_reduction_percent": (
                100.0 * (first_mean - last_mean) / first_mean
                if first_mean > 0.0
                else 0.0
            ),
        }

    dynamic = condition_summary["pc_dynamic_error"]
    no_error = condition_summary["pc_no_error"]
    dynamic_disturbance = dynamic["phases"]["disturbance"][
        "mean_representation_normalized_l2"
    ]
    no_error_disturbance = no_error["phases"]["disturbance"][
        "mean_representation_normalized_l2"
    ]
    relative_improvement = (
        (no_error_disturbance - dynamic_disturbance) / no_error_disturbance
    )
    per_drive_improved = {
        drive: (
            dynamic["disturbance_per_drive_mean_normalized_l2"][drive]
            < no_error["disturbance_per_drive_mean_normalized_l2"][drive]
        )
        for drive in ALLOWED_DRIVES
    }
    recovery_decreased = (
        dynamic["recovery_last_10_mean_normalized_l2"]
        < dynamic["recovery_first_10_mean_normalized_l2"]
    )
    primary_pass = (
        relative_improvement >= PRACTICAL_IMPROVEMENT_THRESHOLD
        and all(per_drive_improved.values())
    )
    decision = "GO" if primary_pass and recovery_decreased else "NO-GO"
    return {
        "conditions": condition_summary,
        "primary_comparison": {
            "metric": "disturbance_mean_representation_normalized_l2",
            "pc_no_error": no_error_disturbance,
            "pc_dynamic_error": dynamic_disturbance,
            "dynamic_relative_improvement_percent": 100.0 * relative_improvement,
            "minimum_practical_improvement_percent": (
                100.0 * PRACTICAL_IMPROVEMENT_THRESHOLD
            ),
            "per_drive_dynamic_lower_than_no_error": per_drive_improved,
            "passed": primary_pass,
        },
        "recovery_check": {
            "pc_dynamic_error_first_10": dynamic[
                "recovery_first_10_mean_normalized_l2"
            ],
            "pc_dynamic_error_last_10": dynamic[
                "recovery_last_10_mean_normalized_l2"
            ],
            "distance_decreased": recovery_decreased,
        },
        "decision": decision,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path, summary):
    conditions = summary["conditions"]
    lines = [
        "# Real-frame Predictive Coding Phase 1",
        "",
        "Frozen inference on KITTI Val drives 0011 and 0039. The trajectory is "
        "40 clean frames, 80 frames of persistent Gaussian blur, then 40 clean "
        "recovery frames. Frozen Test drives 0051/0056 were not read.",
        "",
        "| Condition | Disturbance normalized L2 | Recovery normalized L2 | Recovery last 10 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        item = conditions[condition]
        lines.append(
            f"| {condition} | "
            f"{item['phases']['disturbance']['mean_representation_normalized_l2']:.9f} | "
            f"{item['phases']['recovery']['mean_representation_normalized_l2']:.9f} | "
            f"{item['recovery_last_10_mean_normalized_l2']:.9f} |"
        )
    comparison = summary["primary_comparison"]
    recovery = summary["recovery_check"]
    lines.extend(
        [
            "",
            f"## {summary['decision']}",
            "",
            "PC-dynamic-error changed disturbance distance relative to PC-no-error "
            f"by {comparison['dynamic_relative_improvement_percent']:.6f}% "
            f"(GO threshold: at least {comparison['minimum_practical_improvement_percent']:.1f}% "
            "with both drives improving).",
            "",
            "Dynamic recovery distance changed from "
            f"{recovery['pc_dynamic_error_first_10']:.9f} in the first 10 recovery "
            f"frames to {recovery['pc_dynamic_error_last_10']:.9f} in the last 10.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="ascii")


def main():
    revision = os.environ["PREDIFY_GIT_REVISION"]
    output_dir = Path(os.environ["PREDIFY_REAL_FRAME_PC_PHASE1_OUTPUT_DIR"])
    root = os.environ["PREDIFY_KITTI_ROOT"]
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    weights_path = Path(os.environ["PREDIFY_PCODER_WEIGHTS"])
    drives = tuple(
        item.strip()
        for item in os.environ["PREDIFY_REAL_FRAME_PC_PHASE1_DRIVES"].split(",")
        if item.strip()
    )
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
    seed = int(os.environ.get("PREDIFY_SEED", "0"))

    if drives != ALLOWED_DRIVES:
        raise ValueError(f"Phase 1 drives must be exactly {ALLOWED_DRIVES}, got {drives}.")
    if any(forbidden in drive for forbidden in FORBIDDEN_TEST_DRIVE_IDS for drive in drives):
        raise RuntimeError("Phase 1 attempted to read a Frozen Test drive.")
    if DEVICE.type != "cuda":
        raise RuntimeError("The formal Phase-1 run requires GPU 0.")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    datasets = {
        drive: build_drive_datasets(
            root,
            drive,
            camera,
            fixed_dt_s,
            tolerance,
        )
        for drive in drives
    }
    base_model = build_base_model(str(weights_path))
    rows = []
    c_sqrt_by_condition = {}
    for condition in CONDITIONS:
        print(f"Evaluating {condition} on {DEVICE}...", flush=True)
        condition_rows, c_sqrt = evaluate_condition(
            base_model,
            condition,
            datasets,
            seed,
        )
        rows.extend(condition_rows)
        c_sqrt_by_condition[condition] = c_sqrt
        print(f"Completed {condition}: {len(condition_rows)} rows", flush=True)
    if not all(
        c_sqrt_by_condition[condition] == c_sqrt_by_condition[CONDITIONS[0]]
        for condition in CONDITIONS[1:]
    ):
        raise RuntimeError("C_sqrt calibration differed across formal conditions.")

    result = summarize_rows(rows)
    weight_files = sorted(weights_path.glob("*.pth"))
    summary = {
        "experiment": "real_frame_predictive_coding_phase1",
        "git_revision": revision,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(DEVICE),
        "drives": drives,
        "frozen_test_drives_read": False,
        "conditions": result["conditions"],
        "primary_comparison": result["primary_comparison"],
        "recovery_check": result["recovery_check"],
        "decision": result["decision"],
        "protocol": {
            "baseline_frames": BASELINE_FRAMES,
            "disturbance_frames": DISTURBANCE_FRAMES,
            "recovery_frames": RECOVERY_FRAMES,
            "corruption": "persistent_gaussian_blur",
            "blur_kernel_size": BLUR_KERNEL_SIZE,
            "blur_sigma": BLUR_SIGMA,
            "fixed_dt_s": fixed_dt_s,
            "fixed_dt_tolerance_s": tolerance,
            "seed": seed,
            "network_training": False,
            "optimizer": False,
            "online_learning": False,
            "future_predictor": False,
            "updates_per_real_frame_per_layer": 1,
            "segment_start_reset": True,
            "cross_frame_state_detached": True,
            "parameter_frozen": True,
            "dynamic_error": "epsilon_t=0.207*r_t+0.793*epsilon_(t-1)",
            "pc_no_error_alpha": 0.0,
            "pc_dynamic_error_alpha": 0.01,
            "beta": (0.2, 0.4, 0.4, 0.5, 0.6),
            "lambda": (0.05, 0.1, 0.1, 0.1, 0.0),
            "clean_reference": (
                "condition-matched clean counterfactual on the identical raw frame"
            ),
        },
        "selected_raw_frames": {
            drive: {
                "start": raw_frames[0],
                "stop_inclusive": raw_frames[-1],
                "count": len(raw_frames),
            }
            for drive, (_, _, raw_frames) in datasets.items()
        },
        "time_filter_stats": {
            drive: clean_dataset.time_filter_stats
            for drive, (clean_dataset, _, _) in datasets.items()
        },
        "c_sqrt": c_sqrt_by_condition,
        "pcoder_weight_sha256": {
            path.name: sha256_file(path) for path in weight_files
        },
        "per_frame_row_count": len(rows),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "per_frame.csv", rows)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    write_readme(output_dir / "README.md", summary)
    print(json.dumps({"decision": summary["decision"], **summary["primary_comparison"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
