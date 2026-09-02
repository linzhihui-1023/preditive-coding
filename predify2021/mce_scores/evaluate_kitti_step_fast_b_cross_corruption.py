"""Fast, frozen FAST-B transfer diagnostic on selected KITTI-STEP sequences.

The evaluator deliberately has no training path.  For one corruption at a time
it generates each corrupted frame once, feeds that frame through the shared
backbone, and compares the frozen Host decode with the frozen FAST-B posterior
decode.  This makes each result directory independently publishable.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
    sequence_groups,
)
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import (
    load_writeback_checkpoint,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.kitti_step_cityscapes_c import (
    CITYSCAPES_C_COMMON_CORRUPTIONS,
    apply_cityscapes_c_corruption_uint8,
    canonical_condition_ordinal,
    corruption_seed,
)
from predify2021.mce_scores.kitti_step_persistent_blur import warmup_frame_count
from predify2021.mce_scores.role_separated_direct_state_correction import (
    error_state,
    make_paths,
    zero_state,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    CITYSCAPES_CLASSES,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorRegulatedSemanticRestorationPredictor,
    HostFeature,
    UnifiedFeatures,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    load_components,
    residual_writeback_host_feature,
)


DEFAULT_FAST_B_CHECKPOINT = (
    "/home/lin/predify/experiments/"
    "kitti_step_semantic_v3_joint_c4_fast_ab_5754714/"
    "fast_b_joint_c4_weak_z4/best.pt"
)
SEQUENCES = ("0002", "0010", "0018")
CORRUPTIONS = (
    "gaussian_noise",
    "fog",
    "impulse_noise",
    "motion_blur",
    "snow",
    "contrast",
)
SEVERITIES = (3, 5)
PROTOCOL_PATH = Path(__file__).parents[1] / "datasets" / "cityscapes_corruption_protocol.json"


def _parse_csv(value, allowed, cast=str):
    selected = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    unknown = tuple(item for item in selected if item not in allowed)
    if unknown:
        raise ValueError(f"Unsupported values: {unknown}; allowed={allowed}")
    return selected


def load_protocol_entry(corruption):
    """Read the published protocol so severities are never locally redefined."""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in protocol["corruptions"]}
    if corruption not in entries:
        raise ValueError(f"{corruption} is absent from {PROTOCOL_PATH}")
    entry = entries[corruption]
    if not entry.get("from_hendrycks"):
        raise ValueError(f"{corruption} has no executable published ImageNet-C parameters")
    return entry


def metric(confusion):
    iou = compute_iou(confusion)
    return {
        "miou": float(torch.nanmean(iou).item()),
        "per_class_iou": {
            name: None if torch.isnan(iou[index]) else float(iou[index].item())
            for index, name in enumerate(CITYSCAPES_CLASSES)
        },
    }


def _slice_state(state, index):
    return UnifiedFeatures(*(value[index:index + 1] for value in state.as_tuple()))


def load_fast_b(checkpoint_path):
    """Load frozen Host components and the complete FAST-B Epoch-3 payload."""
    paths = make_paths()
    model, _ = load_components(
        paths["static"], paths["adapter"], paths["predictor"], paths["writeback"]
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not payload.get("joint_c4_training"):
        raise RuntimeError("FAST-B checkpoint does not declare joint C4 training")
    model.multi_layer_adapter.output_adapters[3].load_state_dict(
        payload["c4_output_adapter_state_dict"], strict=True
    )
    model.host_conditioned_writebacks["3"].load_state_dict(
        payload["c4_writeback_state_dict"], strict=True
    )
    predictor = ErrorRegulatedSemanticRestorationPredictor(
        use_error_temporal_stats=payload.get("use_error_temporal_stats", True)
    ).cuda()
    predictor.load_state_dict(payload["model_state_dict"], strict=True)
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    model.eval()
    predictor.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Host contains trainable parameters after FAST-B load")
    if any(parameter.requires_grad for parameter in predictor.parameters()):
        raise RuntimeError("FAST-B predictor contains trainable parameters")
    return model, predictor, payload, paths


def evaluate_condition(model, predictor, groups, corruption, severity, seed, max_frames):
    confusion = {
        "host": torch.zeros((19, 19), dtype=torch.int64),
        "fast_b": torch.zeros((19, 19), dtype=torch.int64),
    }
    sequence_counts = {}
    total_frames = 0
    effective_frames = 0
    consistency = None

    with torch.inference_mode():
        for sequence_id, samples in groups.items():
            # The state is reset at every sequence boundary.
            semantic_hidden = None
            error_temporal_state = None
            h4_dyn = None
            h1_dyn = None
            pending = None
            onset = warmup_frame_count(len(samples))
            sequence_total = min(len(samples), max_frames) if max_frames else len(samples)
            sequence_effective = 0
            sequence_counts[sequence_id] = {
                "total_frame_count": sequence_total,
                "warmup_frame_count": onset,
            }

            for frame_index, sample in enumerate(samples[:sequence_total]):
                clean_image = load_image(sample)
                if frame_index < onset:
                    corrupted_image = clean_image
                else:
                    # One deterministic uint8 corruption is shared by Host and FAST-B.
                    image_uint8 = (
                        clean_image.squeeze(0)
                        .clamp(0.0, 1.0)
                        .mul(255.0)
                        .round()
                        .to(torch.uint8)
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
                    frame_seed = corruption_seed(
                        seed,
                        len(samples),
                        frame_index,
                        corruption,
                        severity,
                    )
                    corrupted_uint8 = apply_cityscapes_c_corruption_uint8(
                        image_uint8, corruption, severity, seed=frame_seed
                    )
                    corrupted_image = (
                        torch.from_numpy(corrupted_uint8)
                        .permute(2, 0, 1)
                        .to(device=clean_image.device, dtype=clean_image.dtype)
                        .div_(255.0)
                        .unsqueeze(0)
                    )

                raw = model.extract_backbone_features(corrupted_image)
                observation = model.encode_backbone_features(raw)
                output_size = tuple(corrupted_image.shape[-2:])
                if semantic_hidden is None:
                    semantic_hidden = predictor.initial_semantic_state(observation)
                if frame_index == 0:
                    pending, h4_dyn, h1_dyn = predictor.predict_next(
                        observation, zero_state(observation), h4_dyn, h1_dyn
                    )
                    total_frames += 1
                    continue

                error = error_state(observation, pending)
                if frame_index == 1:
                    pending, h4_dyn, h1_dyn = predictor.predict_next(
                        observation, error, h4_dyn, h1_dyn
                    )
                    total_frames += 1
                    continue

                restored, semantic_hidden, diagnostics = predictor.restore_current(
                    observation,
                    pending,
                    semantic_hidden,
                    error_temporal_state=error_temporal_state,
                )
                error_temporal_state = diagnostics["error_temporal_state"]
                if frame_index >= onset:
                    host_feature = HostFeature(raw.c4, raw.c1, output_size)
                    zero = zero_state(observation)
                    fast_delta = UnifiedFeatures(
                        restored.z1 - observation.z1,
                        zero.z2,
                        zero.z3,
                        restored.z4 - observation.z4,
                    )
                    fast_feature = residual_writeback_host_feature(
                        model, raw, fast_delta, output_size
                    )
                    predictions = {
                        "host": model.decode_from_host_feature(host_feature)
                        .argmax(1)
                        .squeeze(0)
                        .cpu(),
                        "fast_b": model.decode_from_host_feature(fast_feature)
                        .argmax(1)
                        .squeeze(0)
                        .cpu(),
                    }
                    mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                    for name, prediction in predictions.items():
                        update_confusion_matrix(confusion[name], prediction, mask)
                    effective_frames += 1
                    sequence_effective += 1

                pending, h4_dyn, h1_dyn = predictor.predict_next(
                    observation, error, h4_dyn, h1_dyn
                )
                semantic_hidden = semantic_hidden.detach()
                if error_temporal_state is not None:
                    error_temporal_state = error_temporal_state.detach()
                h4_dyn = h4_dyn.detach()
                h1_dyn = h1_dyn.detach()
                pending = UnifiedFeatures(*(value.detach() for value in pending.as_tuple()))
                total_frames += 1
            sequence_counts[sequence_id]["effective_frame_count"] = sequence_effective

    metrics = {name: metric(value) for name, value in confusion.items()}
    return metrics, {
        "total_frame_count": total_frames,
        "effective_frame_count": effective_frames,
        "sequence_frame_counts": sequence_counts,
    }


def write_readme(path, corruption, checkpoint, sequences, severity_results, mean_delta):
    label = corruption.replace("_", " ").title()
    lines = [
        f"# FAST-B Cross-Corruption Fast Diagnostic — {label}",
        "",
        f"FAST-B checkpoint: `{checkpoint}`",
        "",
        f"Sequences: {' / '.join(sequences)}",
        "",
        "No retraining.  No test-time adaptation.",
        "",
        "| Severity | Host mIoU | FAST-B mIoU | ΔmIoU |",
        "| --- | ---: | ---: | ---: |",
    ]
    for severity in sorted(severity_results):
        row = severity_results[severity]
        lines.append(
            f"| {severity} | {row['host_mIoU']:.9f} | "
            f"{row['fast_b_mIoU']:.9f} | {row['delta_mIoU']:.9f} |"
        )
    lines.extend(
        [
            "",
            f"Mean ΔmIoU: {mean_delta:.9f}",
            "",
            "POSITIVE" if mean_delta > 0 else "NEGATIVE" if mean_delta < 0 else "NEUTRAL",
            "",
            "FAST-B improves under this corruption." if mean_delta > 0 else
            "FAST-B degrades under this corruption." if mean_delta < 0 else
            "FAST-B is neutral under this corruption.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    parser.add_argument("--checkpoint", default=os.environ.get("PREDIFY_FAST_B_CHECKPOINT", DEFAULT_FAST_B_CHECKPOINT))
    parser.add_argument("--corruption", default=os.environ.get("PREDIFY_FAST_B_CORRUPTION", "gaussian_noise"), choices=CORRUPTIONS)
    parser.add_argument("--severities", default=os.environ.get("PREDIFY_FAST_B_SEVERITIES", "3,5"))
    parser.add_argument("--output-root", default=os.environ.get("PREDIFY_FAST_B_OUTPUT_ROOT", "results/fast_b_cross_corruption_fast"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("PREDIFY_SEED", "0")))
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames-per-sequence", type=int, default=0)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("FAST-B cross-corruption evaluation requires CUDA")
    severities = _parse_csv(args.severities, SEVERITIES, int)
    if not severities:
        raise ValueError("At least one severity is required")
    protocol_entry = load_protocol_entry(args.corruption)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FAST-B checkpoint not found: {checkpoint}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, predictor, payload, paths = load_fast_b(checkpoint)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    all_groups = sequence_groups(dataset)
    missing = [sequence for sequence in SEQUENCES if sequence not in all_groups]
    if missing:
        raise RuntimeError(f"Missing required FAST-B validation sequences: {missing}")
    selected = {sequence: all_groups[sequence] for sequence in SEQUENCES}
    if args.max_sequences:
        selected = dict(list(selected.items())[: args.max_sequences])

    start = time.perf_counter()
    severity_results = {}
    condition_counts = {}
    for severity in severities:
        print(f"Evaluating {args.corruption} S{severity}", flush=True)
        metrics, counts = evaluate_condition(
            model,
            predictor,
            selected,
            args.corruption,
            severity,
            args.seed,
            args.max_frames_per_sequence,
        )
        host_miou = metrics["host"]["miou"]
        fast_miou = metrics["fast_b"]["miou"]
        severity_results[f"S{severity}"] = {
            "host_mIoU": host_miou,
            "fast_b_mIoU": fast_miou,
            "delta_mIoU": fast_miou - host_miou,
            "effective_frame_count": counts["effective_frame_count"],
            "total_frame_count": counts["total_frame_count"],
            "metrics": metrics,
        }
        condition_counts[f"S{severity}"] = counts
        print(json.dumps(severity_results[f"S{severity}"], sort_keys=True), flush=True)

    mean_delta = sum(row["delta_mIoU"] for row in severity_results.values()) / len(severity_results)
    output = Path(args.output_root) / args.corruption
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "fast_b_cross_corruption_fast",
        "corruption": args.corruption,
        "protocol_entry": protocol_entry,
        "sequences": list(selected),
        "fast_b_checkpoint": str(checkpoint),
        "source_dynamics_checkpoint": payload.get("source_dynamics_checkpoint"),
        "base_checkpoints": {name: str(path) for name, path in paths.items()},
        "severity_results": severity_results,
        "mean_delta_mIoU": mean_delta,
        "dataset": {
            "split": "val",
            "sequence_count": len(selected),
            "total_frame_count": sum(value["total_frame_count"] for value in condition_counts.values()) / len(condition_counts),
            "effective_frame_count": sum(value["effective_frame_count"] for value in condition_counts.values()) / len(condition_counts),
            "per_severity": condition_counts,
        },
        "protocol": {
            "severities": list(severities),
            "warmup": "predify2021.mce_scores.kitti_step_persistent_blur.warmup_frame_count",
            "corruption_seed_policy": "canonical_condition_ordinal_plus_global_frame_index",
            "corruption_generated_once_per_frame": True,
            "host_fast_b_shared_input": True,
            "state_reset_per_sequence": True,
        },
        "parameter_updates": False,
        "test_time_adaptation": False,
        "training": False,
        "timing": {"wall_seconds": time.perf_counter() - start},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(output / "README.md", args.corruption, checkpoint, tuple(selected), severity_results, mean_delta)
    print(json.dumps({"output": str(output), "mean_delta_mIoU": mean_delta}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
