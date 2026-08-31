import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_semantic_temporal_error_correction import (
    host_from_delta,
    load_new,
    metric_from_confusion,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import update_confusion_matrix
from predify2021.mce_scores.kitti_step_cityscapes_c import (
    CITYSCAPES_C_COMMON_CORRUPTIONS,
    CITYSCAPES_C_SEVERITIES,
    apply_cityscapes_c_corruption,
)
from predify2021.mce_scores.role_separated_direct_state_correction import (
    error_state,
    load_image,
    load_role_components,
    make_paths,
    next_role_prediction,
    zero_state,
)
from predify2021.mce_scores.semantic_temporal_error_step import (
    detach_error_state,
    semantic_temporal_error_step,
    zero_semantic_temporal_state,
)
from predify2021.mce_scores.video_metrics import VideoConsistency
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures


MODEL_NAMES = ("host", "ours")


def parse_csv(value, allowed, cast=str):
    if not value:
        return tuple(allowed)
    selected = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    unknown = tuple(item for item in selected if item not in allowed)
    if unknown:
        raise ValueError(f"Unsupported values: {unknown}")
    return selected


def evaluate_condition(model, predictor, corrections, groups, corruption=None, severity=None):
    confusion = {
        name: torch.zeros((19, 19), dtype=torch.int64)
        for name in MODEL_NAMES
    }
    video_consistency = VideoConsistency(MODEL_NAMES)
    frame_count = 0

    with torch.inference_mode():
        for samples in groups.values():
            video_consistency.reset_sequence()
            hidden = None
            predictor_hidden = predictor.initial_state()
            pending_dynamics = None
            pending_semantic = None

            for frame_index, sample in enumerate(samples):
                image = load_image(sample)
                if corruption is not None:
                    image = apply_cityscapes_c_corruption(image, corruption, severity)

                raw = model.extract_backbone_features(image)
                observation = model.encode_backbone_features(raw)
                output_size = tuple(image.shape[-2:])

                if hidden is None:
                    hidden = zero_semantic_temporal_state(observation)

                if frame_index == 0:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor,
                        observation,
                        zero_state(observation),
                        predictor_hidden,
                    )
                    continue

                error = error_state(observation, pending_dynamics)
                if frame_index == 1:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor,
                        observation,
                        error,
                        predictor_hidden,
                    )
                    continue

                posterior, hidden, _ = semantic_temporal_error_step(
                    corrections,
                    observation,
                    pending_dynamics,
                    pending_semantic,
                    hidden,
                )
                zero = zero_state(observation)
                host_features = {
                    "host": HostFeature(raw.c4, raw.c1, output_size),
                    "ours": host_from_delta(
                        model,
                        raw,
                        UnifiedFeatures(
                            posterior.z1 - observation.z1,
                            zero.z2,
                            zero.z3,
                            posterior.z4 - observation.z4,
                        ),
                        output_size,
                    ),
                }
                predictions = {
                    name: model.decode_from_host_feature(features).argmax(1).squeeze(0).cpu()
                    for name, features in host_features.items()
                }
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name, prediction in predictions.items():
                    update_confusion_matrix(confusion[name], prediction.to(torch.int64), mask)
                video_consistency.append(mask, predictions)
                frame_count += 1

                pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                    predictor,
                    observation,
                    error,
                    predictor_hidden,
                )
                hidden = detach_error_state(hidden)

    mvc = video_consistency.means()
    metrics = {
        name: {
            **metric_from_confusion(confusion[name]),
            "mvc8": mvc[8][name],
            "mvc16": mvc[16][name],
        }
        for name in MODEL_NAMES
    }
    return {
        "metrics": metrics,
        "effective_frame_count": frame_count,
        "mvc_window_counts": video_consistency.window_counts(),
    }


def per_corruption_summary(clean, severity_results):
    output = {}
    for name in MODEL_NAMES:
        clean_miou = clean["metrics"][name]["miou"]
        host_values = [
            severity_results[f"S{severity}"]["metrics"]["host"]["miou"]
            for severity in CITYSCAPES_C_SEVERITIES
        ]
        values = [
            severity_results[f"S{severity}"]["metrics"][name]["miou"]
            for severity in CITYSCAPES_C_SEVERITIES
        ]
        degradation = [1.0 - value for value in values]
        host_degradation = [1.0 - value for value in host_values]
        clean_error = 1.0 - clean_miou
        host_clean_error = 1.0 - clean["metrics"]["host"]["miou"]
        cd = sum(degradation) / sum(host_degradation)
        rcd_denominator = sum(host_degradation) - host_clean_error
        rcd = (sum(degradation) - clean_error) / rcd_denominator

        output[name] = {
            "severity_miou": values,
            "mean_corruption_miou": sum(values) / len(values),
            "mean_corruption_wiou": sum(
                severity_results[f"S{severity}"]["metrics"][name]["wiou"]
                for severity in CITYSCAPES_C_SEVERITIES
            )
            / len(CITYSCAPES_C_SEVERITIES),
            "mean_corruption_mvc8": sum(
                severity_results[f"S{severity}"]["metrics"][name]["mvc8"]
                for severity in CITYSCAPES_C_SEVERITIES
            )
            / len(CITYSCAPES_C_SEVERITIES),
            "mean_corruption_mvc16": sum(
                severity_results[f"S{severity}"]["metrics"][name]["mvc16"]
                for severity in CITYSCAPES_C_SEVERITIES
            )
            / len(CITYSCAPES_C_SEVERITIES),
            "cd_vs_host_reference": cd,
            "rcd_vs_host_reference": rcd,
        }
    return output


def aggregate_suite(clean, results, corruptions):
    per_corruption = {
        corruption: per_corruption_summary(clean, results[corruption])
        for corruption in corruptions
    }
    aggregate = {}
    condition_count = len(corruptions) * len(CITYSCAPES_C_SEVERITIES)

    for name in MODEL_NAMES:
        aggregate[name] = {
            metric: sum(
                results[corruption][f"S{severity}"]["metrics"][name][metric]
                for corruption in corruptions
                for severity in CITYSCAPES_C_SEVERITIES
            )
            / condition_count
            for metric in ("miou", "wiou", "mvc8", "mvc16")
        }
        aggregate[name]["mean_cd_vs_host_reference"] = sum(
            per_corruption[corruption][name]["cd_vs_host_reference"]
            for corruption in corruptions
        ) / len(corruptions)
        aggregate[name]["mean_rcd_vs_host_reference"] = sum(
            per_corruption[corruption][name]["rcd_vs_host_reference"]
            for corruption in corruptions
        ) / len(corruptions)

    aggregate["ours_minus_host"] = {
        metric: aggregate["ours"][metric] - aggregate["host"][metric]
        for metric in ("miou", "wiou", "mvc8", "mvc16")
    }
    return per_corruption, aggregate


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("KITTI-STEP common-corruption evaluation requires CUDA")

    seed = int(os.environ.get("PREDIFY_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    checkpoint = Path(os.environ["PREDIFY_SEMANTIC_TEMPORAL_ERROR_CHECKPOINT"])
    output = Path(
        os.environ.get(
            "PREDIFY_CITYSCAPES_C_OUTPUT_DIR",
            "results/kitti_step_cityscapes_c",
        )
    )
    corruptions = parse_csv(
        os.environ.get("PREDIFY_CITYSCAPES_C_CORRUPTIONS"),
        CITYSCAPES_C_COMMON_CORRUPTIONS,
    )
    severities = parse_csv(
        os.environ.get("PREDIFY_CITYSCAPES_C_SEVERITIES"),
        CITYSCAPES_C_SEVERITIES,
        int,
    )
    max_sequences = int(os.environ.get("PREDIFY_CITYSCAPES_C_MAX_SEQUENCES", "0"))

    paths = make_paths()
    model, predictor = load_role_components(
        paths["static"],
        paths["adapter"],
        paths["predictor"],
        paths["writeback"],
    )
    corrections, _ = load_new(checkpoint)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    if max_sequences > 0:
        groups = dict(list(groups.items())[:max_sequences])

    clean = evaluate_condition(model, predictor, corrections, groups)
    results = {}
    for corruption in corruptions:
        results[corruption] = {}
        for severity in severities:
            print(
                f"Evaluating corruption={corruption} severity={severity}",
                flush=True,
            )
            results[corruption][f"S{severity}"] = evaluate_condition(
                model,
                predictor,
                corrections,
                groups,
                corruption=corruption,
                severity=severity,
            )

    if severities == CITYSCAPES_C_SEVERITIES:
        per_corruption, aggregate = aggregate_suite(clean, results, corruptions)
    else:
        per_corruption, aggregate = {}, {}

    result = {
        "experiment": "kitti_step_cityscapes_c_common_corruption",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(checkpoint),
        "dataset": {
            "split": "val",
            "sequence_count": len(groups),
            "total_frame_count": sum(len(samples) for samples in groups.values()),
            "effective_frame_count_per_condition": clean["effective_frame_count"],
        },
        "protocol": {
            "corruption_implementation": "bethgelab/imagecorruptions",
            "corruption_family": "ImageNet-C common 15 as used for Cityscapes-C style evaluation",
            "corruptions": corruptions,
            "severities": severities,
            "fixed_corruption_and_severity_within_sequence": True,
            "corruption_from_first_frame": True,
            "clean_warmup_frames": 0,
            "model_state_initialization_frames_excluded_from_metrics": 2,
            "training": False,
            "checkpoint_selection": False,
            "correction_feedback_to_predictor": False,
            "cd_reference_model": "host",
        },
        "clean": clean,
        "conditions": results,
        "per_corruption_summary": per_corruption,
        "aggregate": aggregate,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
