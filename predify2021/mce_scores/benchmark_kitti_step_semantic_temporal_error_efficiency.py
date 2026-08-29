import json
import os
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.kitti_step_persistent_blur import (
    persistent_gaussian_blur,
    warmup_frame_count,
)
from predify2021.mce_scores.role_separated_direct_state_correction import (
    error_state,
    load_image,
    load_role_components,
    make_paths,
    next_role_prediction,
    zero_state,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    residual_writeback_host_feature,
)
from predify2021.mce_scores.semantic_temporal_error_step import (
    detach_error_state,
    semantic_temporal_error_step,
    zero_error_state,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import (
    build_semantic_temporal_corrections,
)


CHECKPOINT_DEFAULT = (
    "/home/lin/experiments/"
    "kitti_step_semantic_temporal_error_correction_10epoch_ffb1a74/"
    "best_semantic_temporal_error_correction.pt"
)


def load_corrections(path):
    corrections = build_semantic_temporal_corrections()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    corrections.load_state_dict(payload["corrections"], strict=True)
    corrections.requires_grad_(False)
    corrections.eval()
    return corrections


def corrected_host_feature(model, raw_features, observation, posterior, output_size):
    delta = UnifiedFeatures(
        posterior.z1 - observation.z1,
        torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3),
        posterior.z4 - observation.z4,
    )
    return residual_writeback_host_feature(model, raw_features, delta, output_size)


def elapsed_fps(events):
    torch.cuda.synchronize()
    milliseconds = sum(start.elapsed_time(end) for start, end in events)
    frame_count = len(events)
    return {
        "frames": frame_count,
        "latency_ms_per_frame": milliseconds / frame_count,
        "fps": frame_count / (milliseconds / 1000.0),
    }


def benchmark_host(model, groups):
    events = []
    with torch.inference_mode():
        for samples in groups.values():
            first_timed = warmup_frame_count(len(samples))
            for frame_index, sample in enumerate(samples):
                image = load_image(sample)
                corrupted = persistent_gaussian_blur(image, frame_index, len(samples))
                timed = frame_index >= first_timed
                if timed:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                raw = model.extract_backbone_features(corrupted)
                output_size = tuple(corrupted.shape[-2:])
                model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))
                if timed:
                    end.record()
                    events.append((start, end))
    return elapsed_fps(events)


def benchmark_current_model(model, predictor, corrections, groups):
    events = []
    with torch.inference_mode():
        for samples in groups.values():
            correction_hidden = None
            predictor_hidden = predictor.initial_state()
            pending_dynamics = None
            pending_semantic = None
            first_timed = warmup_frame_count(len(samples))

            for frame_index, sample in enumerate(samples):
                image = load_image(sample)
                corrupted = persistent_gaussian_blur(image, frame_index, len(samples))
                timed = frame_index >= first_timed
                if timed:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                raw = model.extract_backbone_features(corrupted)
                observation = model.encode_backbone_features(raw)
                output_size = tuple(corrupted.shape[-2:])

                if correction_hidden is None:
                    correction_hidden = zero_error_state(observation)

                if pending_dynamics is None:
                    prediction_error = zero_state(observation)
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor,
                        observation,
                        prediction_error,
                        predictor_hidden,
                    )
                    posterior = observation
                else:
                    prediction_error = error_state(observation, pending_dynamics)
                    posterior, correction_hidden, _ = semantic_temporal_error_step(
                        corrections,
                        observation,
                        pending_dynamics,
                        pending_semantic,
                        correction_hidden,
                    )

                corrected = corrected_host_feature(
                    model,
                    raw,
                    observation,
                    posterior,
                    output_size,
                )
                model.decode_from_host_feature(corrected)

                if pending_dynamics is not None and frame_index > 0:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor,
                        observation,
                        prediction_error,
                        predictor_hidden,
                    )

                correction_hidden = detach_error_state(correction_hidden)

                if timed:
                    end.record()
                    events.append((start, end))

    return elapsed_fps(events)


def parameter_counts(model, predictor, corrections):
    adapter = sum(parameter.numel() for parameter in model.multi_layer_adapter.parameters())
    writeback = sum(
        parameter.numel()
        for parameter in model.host_conditioned_writebacks.parameters()
    )
    predictor_count = sum(parameter.numel() for parameter in predictor.parameters())
    correction = sum(parameter.numel() for parameter in corrections.parameters())
    return {
        "adapter": adapter,
        "writeback": writeback,
        "predictor": predictor_count,
        "correction": correction,
        "additional_inference_parameters": adapter + writeback + predictor_count + correction,
        "trainable_correction_parameters": correction,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Efficiency benchmark requires CUDA.")

    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    checkpoint = Path(
        os.environ.get("PREDIFY_SEMANTIC_TEMPORAL_ERROR_CHECKPOINT", CHECKPOINT_DEFAULT)
    )
    output = Path(
        os.environ.get(
            "PREDIFY_SEMANTIC_TEMPORAL_ERROR_EFFICIENCY_OUTPUT",
            "results/kitti_step_semantic_temporal_error_efficiency.json",
        )
    )

    paths = make_paths()
    model, predictor = load_role_components(
        paths["static"],
        paths["adapter"],
        paths["predictor"],
        paths["writeback"],
    )
    corrections = load_corrections(checkpoint)

    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)

    host = benchmark_host(model, groups)
    current = benchmark_current_model(model, predictor, corrections, groups)

    result = {
        "experiment": "kitti_step_semantic_temporal_error_efficiency",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(checkpoint),
        "protocol": {
            "split": "val",
            "batch_size": 1,
            "precision": "fp32",
            "timing": "CUDA events with synchronization after collection",
            "state_policy": "ordered per-sequence steady-state inference",
            "timed_frames": "post 10-percent state warm-up",
            "included_full_model": [
                "host backbone",
                "state adapter",
                "prediction error",
                "local feature correlation",
                "semantic error encoder",
                "temporal error ConvGRU",
                "semantic correction",
                "residual writeback",
                "segmentation decoder",
                "predictor update for next frame",
            ],
            "excluded": [
                "clean reference",
                "Corrected B",
                "Error Decomposition",
                "ground truth",
                "mIoU",
                "wIoU",
                "mVC8",
                "mVC16",
                "diagnostics",
                "argmax and CPU transfer",
                "image loading",
                "Gaussian Blur generation",
                "disk I/O",
                "JSON writing",
            ],
        },
        "parameters": parameter_counts(model, predictor, corrections),
        "host_only": host,
        "full_current_model": current,
        "fps_retention": current["fps"] / host["fps"],
        "fps_drop_percentage": 100.0 * (1.0 - current["fps"] / host["fps"]),
        "latency_overhead_ms": (
            current["latency_ms_per_frame"] - host["latency_ms_per_frame"]
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
