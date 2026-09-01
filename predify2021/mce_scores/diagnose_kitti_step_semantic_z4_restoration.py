"""Test whether the current Semantic ConvGRU restores blur-damaged Z4/C4 semantics.

Diagnostic only. No model parameters are updated.

The experiment is intentionally narrow:
1. Host blur baseline.
2. Semantic Z4 with semantic recurrence reset every step (no semantic history).
3. Semantic Z4 with continuous Semantic ConvGRU state.
4. Oracle clean-Z4 residual through the CURRENT writeback path.
5. Oracle clean C4 replacement before the frozen DeepLabV3+ decoder.

It answers two questions:
- Does the current Semantic ConvGRU move blurred Z4 toward same-frame clean Z4?
- Does that recovered Z4 translate into host mIoU through the current writeback?

Blur-Mid / Blur-Max reuse the existing diagnostic blur protocol. Recurrent
states are run from sequence start; metrics are accumulated only after the
10% warmup so they match the Host Failure Diagnosis evaluation window.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
    sequence_groups,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.kitti_step_persistent_blur import warmup_frame_count
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    error_state,
    load_components,
    residual_writeback_host_feature,
    zero_state,
)
from predify2021.mce_scores.train_kitti_step_convgru_decoupling_quick import (
    diagnostic_blur,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures


SEED = 0
CONDITIONS = ("Blur-Mid", "Blur-Max")
VARIANTS = (
    "blur_host",
    "semantic_no_history",
    "semantic_continuous",
    "oracle_z4_writeback",
    "clean_c4_oracle",
    "clean_host",
)
SEMANTIC_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_robust_semantic_reference/"
    "best_robust_semantic_reference.pt"
)


def zero_z4_delta(reference, z4_delta):
    zero = zero_state(reference)
    return UnifiedFeatures(zero.z1, zero.z2, zero.z3, z4_delta)


def decode_host(model, raw, output_size):
    return model.decode_from_host_feature(
        HostFeature(raw.c4, raw.c1, output_size)
    )


def decode_clean_c4_oracle(model, clean_raw, blur_raw, output_size):
    return model.decode_from_host_feature(
        HostFeature(clean_raw.c4, blur_raw.c1, output_size)
    )


def decode_z4_writeback(model, blur_raw, observation, target_z4, output_size):
    delta = zero_z4_delta(observation, target_z4 - observation.z4)
    host_feature = residual_writeback_host_feature(
        model,
        blur_raw,
        delta,
        output_size,
    )
    return model.decode_from_host_feature(host_feature)


def new_feature_aggregate():
    return {
        "element_count": 0,
        "observation_sse": 0.0,
        "semantic_sse": 0.0,
        "direction_cosine_sum": 0.0,
        "direction_cosine_count": 0,
    }


def add_feature_metrics(aggregate, observation_z4, semantic_z4, clean_z4):
    observation_error = observation_z4.float() - clean_z4.float()
    semantic_error = semantic_z4.float() - clean_z4.float()
    aggregate["element_count"] += observation_error.numel()
    aggregate["observation_sse"] += float(observation_error.square().sum().item())
    aggregate["semantic_sse"] += float(semantic_error.square().sum().item())

    predicted_residual = semantic_z4.float() - observation_z4.float()
    oracle_residual = clean_z4.float() - observation_z4.float()
    cosine = F.cosine_similarity(
        predicted_residual,
        oracle_residual,
        dim=1,
        eps=1e-8,
    )
    aggregate["direction_cosine_sum"] += float(cosine.sum().item())
    aggregate["direction_cosine_count"] += cosine.numel()


def finalize_feature_metrics(aggregate):
    elements = max(aggregate["element_count"], 1)
    observation_mse = aggregate["observation_sse"] / elements
    semantic_mse = aggregate["semantic_sse"] / elements
    recovery = 1.0 - semantic_mse / max(observation_mse, 1e-12)
    cosine_count = max(aggregate["direction_cosine_count"], 1)
    return {
        "observation_mse_to_clean_z4": observation_mse,
        "semantic_mse_to_clean_z4": semantic_mse,
        "feature_recovery_fraction": recovery,
        "direction_cosine_to_oracle_residual": (
            aggregate["direction_cosine_sum"] / cosine_count
        ),
    }


def recovery_fraction(baseline, value, upper):
    gap = upper - baseline
    if abs(gap) < 1e-12:
        return 0.0
    return (value - baseline) / gap


def initial_stream(predictor):
    return {
        "hidden": predictor.initial_state(),
        "pending_dynamics": None,
        "pending_semantic": None,
    }


def advance_stream(predictor, observation, stream, reset_semantic_history):
    if stream["pending_dynamics"] is None:
        prediction_error = zero_state(observation)
    else:
        prediction_error = error_state(
            observation,
            stream["pending_dynamics"],
        )

    h4_dyn, h4_sem, h1_dyn, h1_sem = stream["hidden"]
    if reset_semantic_history:
        h4_sem = None
        h1_sem = None

    dynamics, semantic, *hidden = predictor.step(
        observation,
        prediction_error,
        h4_dyn,
        h4_sem,
        h1_dyn,
        h1_sem,
    )
    stream["pending_dynamics"] = dynamics
    stream["pending_semantic"] = semantic
    stream["hidden"] = tuple(hidden)


def evaluate_condition(model, predictor, groups, condition, max_effective_frames=0):
    confusion = {
        name: torch.zeros((19, 19), dtype=torch.int64)
        for name in VARIANTS
    }
    feature = {
        "semantic_no_history": new_feature_aggregate(),
        "semantic_continuous": new_feature_aggregate(),
    }
    effective_frames = 0

    with torch.inference_mode():
        for samples in groups.values():
            if len(samples) < 2:
                continue

            onset = warmup_frame_count(len(samples))
            streams = {
                "semantic_no_history": initial_stream(predictor),
                "semantic_continuous": initial_stream(predictor),
            }

            first_clean = load_image(samples[0])
            first_blur = diagnostic_blur(
                first_clean, 0, len(samples), condition
            )
            first_raw = model.extract_backbone_features(first_blur)
            first_observation = model.encode_backbone_features(first_raw)
            advance_stream(
                predictor,
                first_observation,
                streams["semantic_no_history"],
                reset_semantic_history=True,
            )
            advance_stream(
                predictor,
                first_observation,
                streams["semantic_continuous"],
                reset_semantic_history=False,
            )

            for frame_index in range(1, len(samples)):
                if max_effective_frames and effective_frames >= max_effective_frames:
                    break

                sample = samples[frame_index]
                clean_image = load_image(sample)
                blur_image = diagnostic_blur(
                    clean_image,
                    frame_index,
                    len(samples),
                    condition,
                )
                output_size = tuple(clean_image.shape[-2:])

                clean_raw = model.extract_backbone_features(clean_image)
                blur_raw = model.extract_backbone_features(blur_image)
                clean_state = model.encode_backbone_features(clean_raw)
                observation = model.encode_backbone_features(blur_raw)

                if frame_index >= onset:
                    mask = semantic_mask_from_panoptic_png(sample["mask_path"])

                    no_history_semantic = streams[
                        "semantic_no_history"
                    ]["pending_semantic"]
                    continuous_semantic = streams[
                        "semantic_continuous"
                    ]["pending_semantic"]

                    if no_history_semantic is None or continuous_semantic is None:
                        raise RuntimeError("Semantic reference is missing on an evaluable frame")

                    add_feature_metrics(
                        feature["semantic_no_history"],
                        observation.z4,
                        no_history_semantic.z4,
                        clean_state.z4,
                    )
                    add_feature_metrics(
                        feature["semantic_continuous"],
                        observation.z4,
                        continuous_semantic.z4,
                        clean_state.z4,
                    )

                    logits = {
                        "blur_host": decode_host(model, blur_raw, output_size),
                        "semantic_no_history": decode_z4_writeback(
                            model,
                            blur_raw,
                            observation,
                            no_history_semantic.z4,
                            output_size,
                        ),
                        "semantic_continuous": decode_z4_writeback(
                            model,
                            blur_raw,
                            observation,
                            continuous_semantic.z4,
                            output_size,
                        ),
                        "oracle_z4_writeback": decode_z4_writeback(
                            model,
                            blur_raw,
                            observation,
                            clean_state.z4,
                            output_size,
                        ),
                        "clean_c4_oracle": decode_clean_c4_oracle(
                            model,
                            clean_raw,
                            blur_raw,
                            output_size,
                        ),
                        "clean_host": decode_host(model, clean_raw, output_size),
                    }

                    for name, value in logits.items():
                        prediction = value.argmax(1).squeeze(0).cpu()
                        update_confusion_matrix(confusion[name], prediction, mask)

                    effective_frames += 1

                advance_stream(
                    predictor,
                    observation,
                    streams["semantic_no_history"],
                    reset_semantic_history=True,
                )
                advance_stream(
                    predictor,
                    observation,
                    streams["semantic_continuous"],
                    reset_semantic_history=False,
                )

            if max_effective_frames and effective_frames >= max_effective_frames:
                break

    miou = {
        name: float(torch.nanmean(compute_iou(matrix)).item())
        for name, matrix in confusion.items()
    }
    feature_metrics = {
        name: finalize_feature_metrics(values)
        for name, values in feature.items()
    }

    blur = miou["blur_host"]
    host_upper = miou["clean_c4_oracle"]
    writeback_upper = miou["oracle_z4_writeback"]
    no_history = miou["semantic_no_history"]
    continuous = miou["semantic_continuous"]

    task = {
        "host_c4_gap": host_upper - blur,
        "oracle_z4_writeback_gain": writeback_upper - blur,
        "oracle_z4_writeback_fraction_of_host_c4_gap": recovery_fraction(
            blur, writeback_upper, host_upper
        ),
        "semantic_no_history_gain": no_history - blur,
        "semantic_continuous_gain": continuous - blur,
        "semantic_no_history_fraction_of_host_c4_gap": recovery_fraction(
            blur, no_history, host_upper
        ),
        "semantic_continuous_fraction_of_host_c4_gap": recovery_fraction(
            blur, continuous, host_upper
        ),
        "semantic_no_history_fraction_of_writeback_capacity": recovery_fraction(
            blur, no_history, writeback_upper
        ),
        "semantic_continuous_fraction_of_writeback_capacity": recovery_fraction(
            blur, continuous, writeback_upper
        ),
        "temporal_memory_mIoU_gain": continuous - no_history,
    }

    continuous_feature = feature_metrics["semantic_continuous"]
    no_history_feature = feature_metrics["semantic_no_history"]
    judgement = {
        "semantic_z4_restoration": (
            "SUPPORTED"
            if (
                continuous_feature["feature_recovery_fraction"] > 0.0
                and task["semantic_continuous_gain"] > 0.0
            )
            else "NOT_SUPPORTED"
        ),
        "semantic_temporal_memory_contribution": (
            "SUPPORTED"
            if (
                continuous_feature["feature_recovery_fraction"]
                > no_history_feature["feature_recovery_fraction"]
                and task["temporal_memory_mIoU_gain"] > 0.0
            )
            else "NOT_SUPPORTED"
        ),
        "writeback_can_use_oracle_z4": (
            "SUPPORTED"
            if task["oracle_z4_writeback_gain"] > 0.0
            else "NOT_SUPPORTED"
        ),
    }

    return {
        "condition": condition,
        "effective_frame_count": effective_frames,
        "mIoU": miou,
        "feature": feature_metrics,
        "task_recovery": task,
        "judgement": judgement,
    }


def write_csv(path, results):
    rows = []
    for condition, result in results.items():
        task = result["task_recovery"]
        for variant in VARIANTS:
            row = {
                "condition": condition,
                "variant": variant,
                "mIoU": result["mIoU"][variant],
                "gain_vs_blur": result["mIoU"][variant]
                - result["mIoU"]["blur_host"],
                "fraction_of_clean_c4_gap": recovery_fraction(
                    result["mIoU"]["blur_host"],
                    result["mIoU"][variant],
                    result["mIoU"]["clean_c4_oracle"],
                ),
            }
            if variant in result["feature"]:
                row.update(result["feature"][variant])
            if variant == "oracle_z4_writeback":
                row["fraction_of_current_writeback_capacity"] = 1.0
            elif variant in ("semantic_no_history", "semantic_continuous"):
                row["fraction_of_current_writeback_capacity"] = recovery_fraction(
                    result["mIoU"]["blur_host"],
                    result["mIoU"][variant],
                    result["mIoU"]["oracle_z4_writeback"],
                )
            else:
                row["fraction_of_current_writeback_capacity"] = ""
            row["temporal_memory_mIoU_gain"] = (
                task["temporal_memory_mIoU_gain"]
                if variant == "semantic_continuous"
                else ""
            )
            rows.append(row)

    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/lin/predify/kitti_step",
    )
    parser.add_argument(
        "--semantic-checkpoint",
        default=SEMANTIC_CHECKPOINT_DEFAULT,
        help=(
            "RoleSeparatedRecurrentPredictor checkpoint containing the current "
            "Semantic ConvGRU weights."
        ),
    )
    parser.add_argument(
        "--output",
        default="results/kitti_step_semantic_z4_restoration_test",
    )
    parser.add_argument(
        "--sequence-limit",
        type=int,
        default=0,
        help="0 evaluates all KITTI-STEP validation sequences.",
    )
    parser.add_argument(
        "--max-effective-frames",
        type=int,
        default=0,
        help="0 evaluates every post-warmup frame.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One sequence and at most 32 post-warmup frames.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Semantic Z4 Restoration Test requires CUDA")

    checkpoint = Path(args.semantic_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Semantic checkpoint not found: {checkpoint}"
        )

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model, predictor = load_components(
        STATIC_CHECKPOINT_DEFAULT,
        ADAPTER_CHECKPOINT_DEFAULT,
        str(checkpoint),
        WRITEBACK_CHECKPOINT_DEFAULT,
    )
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    model.eval()
    predictor.eval()

    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root),
        "val",
    )
    groups = sequence_groups(dataset)

    sequence_limit = args.sequence_limit
    max_effective_frames = args.max_effective_frames
    output = Path(args.output)
    if args.smoke:
        sequence_limit = 1
        max_effective_frames = 32
        output = output / "smoke"
    if sequence_limit:
        groups = dict(list(groups.items())[:sequence_limit])

    output.mkdir(parents=True, exist_ok=True)

    results = {}
    for condition in CONDITIONS:
        result = evaluate_condition(
            model,
            predictor,
            groups,
            condition,
            max_effective_frames=max_effective_frames,
        )
        results[condition] = result
        print(
            json.dumps(
                {
                    "condition": condition,
                    "effective_frames": result["effective_frame_count"],
                    "mIoU": result["mIoU"],
                    "feature": result["feature"],
                    "task_recovery": result["task_recovery"],
                    "judgement": result["judgement"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        "experiment": "kitti_step_semantic_z4_restoration_test",
        "diagnostic_only": True,
        "parameters_updated": False,
        "semantic_checkpoint": str(checkpoint),
        "split": "val",
        "sequence_count": len(groups),
        "conditions": {
            "Blur-Mid": "Gaussian blur sigma=2.25 after existing 10% warmup",
            "Blur-Max": "Gaussian blur sigma=3.0 after existing 10% warmup",
        },
        "evaluated_window": "post-warmup frames only; recurrent state runs from sequence start",
        "question": (
            "Can the current Semantic ConvGRU restore the blur-damaged Z4/C4 "
            "high-level semantics required by the frozen host?"
        ),
        "variants": {
            "blur_host": "Blurred host without correction",
            "semantic_no_history": (
                "Current semantic Z4 reference with h4_sem/h1_sem reset at "
                "every predictor step, written back only through Z4"
            ),
            "semantic_continuous": (
                "Current continuous Semantic ConvGRU Z4 reference, written "
                "back only through Z4"
            ),
            "oracle_z4_writeback": (
                "Same-frame clean Z4 - blurred Z4 residual through the current "
                "trained writeback; isolates writeback capacity"
            ),
            "clean_c4_oracle": (
                "Same-frame clean raw C4 replacing blurred C4 before decoder; "
                "host-level C4 upper bound"
            ),
            "clean_host": "Clean C1 + clean C4 host",
        },
        "results": results,
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_csv(output / "comparison.csv", results)

    compact = {
        condition: {
            "blur": result["mIoU"]["blur_host"],
            "no_history": result["mIoU"]["semantic_no_history"],
            "continuous": result["mIoU"]["semantic_continuous"],
            "oracle_z4_writeback": result["mIoU"]["oracle_z4_writeback"],
            "clean_c4_oracle": result["mIoU"]["clean_c4_oracle"],
            "continuous_feature_recovery": result["feature"][
                "semantic_continuous"
            ]["feature_recovery_fraction"],
            "continuous_direction_cosine": result["feature"][
                "semantic_continuous"
            ]["direction_cosine_to_oracle_residual"],
            "temporal_memory_mIoU_gain": result["task_recovery"][
                "temporal_memory_mIoU_gain"
            ],
            "judgement": result["judgement"],
        }
        for condition, result in results.items()
    }
    print(json.dumps({"summary": compact}, indent=2), flush=True)
    print(f"results={output}", flush=True)


if __name__ == "__main__":
    main()
