"""Fast mIoU bottleneck diagnosis for the trained V3-TemporalStats checkpoint."""

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.kitti_step_persistent_blur import BLUR_KERNEL_SIZE, warmup_frame_count
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT, STATIC_CHECKPOINT_DEFAULT, WRITEBACK_CHECKPOINT_DEFAULT,
    error_state, load_components, residual_writeback_host_feature, zero_state,
)
from predify2021.mce_scores.train_kitti_step_semantic_v3 import FAST_VALIDATION_SEQUENCES
from predify2021.model_factory.deeplabv3plus_resnet50 import ErrorRegulatedSemanticRestorationPredictor, HostFeature, UnifiedFeatures

CONDITIONS = ("Blur-Mid", "Blur-Max")
SIGMA = {"Blur-Mid": 2.25, "Blur-Max": 3.0}


def slice_state(state, index):
    return UnifiedFeatures(*(value[index:index + 1] for value in state.as_tuple()))


def encode_frame(model, sample, frame, total):
    clean = load_image(sample)
    mid = TF.gaussian_blur(clean, [BLUR_KERNEL_SIZE] * 2, [SIGMA["Blur-Mid"]] * 2) if frame >= warmup_frame_count(total) else clean
    maximum = TF.gaussian_blur(clean, [BLUR_KERNEL_SIZE] * 2, [SIGMA["Blur-Max"]] * 2) if frame >= warmup_frame_count(total) else clean
    with torch.inference_mode():
        raw = model.extract_backbone_features(torch.cat((clean, mid, maximum), dim=0))
        states = model.encode_backbone_features(raw)
    raws = tuple(type(raw)(*(value[i:i + 1] for value in raw.as_tuple())) for i in range(3))
    encoded = tuple(slice_state(states, i) for i in range(3))
    return encoded[0], encoded[1], encoded[2], raws[0], raws[1], raws[2], tuple(clean.shape[-2:])


def host_logits(model, raw, delta, output_size):
    return model.decode_from_host_feature(residual_writeback_host_feature(model, raw, delta, output_size))


def direct_logits(model, raw, output_size):
    return model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))


def run(model, predictor, groups):
    results = {}
    for condition in CONDITIONS:
        confusion = {name: torch.zeros((19, 19), dtype=torch.int64) for name in ("corrupted", "learned", "oracle_z4_writeback", "clean_c4")}
        z4_sse = {name: 0.0 for name in ("corrupted", "learned", "oracle")}
        frames = 0
        for samples in groups.values():
            if len(samples) < 2:
                continue
            clean0, mid0, max0, _, _, _, _ = encode_frame(model, samples[0], 0, len(samples))
            observation0 = mid0 if condition == "Blur-Mid" else max0
            pending, h4, h1 = predictor.predict_next(observation0, zero_state(observation0), None, None)
            hidden = predictor.initial_semantic_state(observation0)
            error_stats = predictor.initial_error_temporal_statistics()
            onset = warmup_frame_count(len(samples))
            for frame, sample in enumerate(samples[1:], start=1):
                clean, mid, maximum, raw_clean, raw_mid, raw_max, output_size = encode_frame(model, sample, frame, len(samples))
                observation, raw_observed = (mid, raw_mid) if condition == "Blur-Mid" else (maximum, raw_max)
                prediction_error = error_state(observation, pending)
                with torch.inference_mode():
                    learned, hidden, diagnostics = predictor.restore_current(observation, pending, hidden, error_temporal_state=error_stats)
                error_stats = diagnostics["error_temporal_state"]
                if frame < onset:
                    pending, h4, h1 = predictor.predict_next(observation, prediction_error, h4, h1)
                    continue
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                zero = zero_state(observation)
                learned_delta = UnifiedFeatures(zero.z1, zero.z2, zero.z3, learned.z4 - observation.z4)
                oracle_delta = UnifiedFeatures(zero.z1, zero.z2, zero.z3, clean.z4 - observation.z4)
                with torch.inference_mode():
                    logits = {
                        "corrupted": direct_logits(model, raw_observed, output_size),
                        "learned": host_logits(model, raw_observed, learned_delta, output_size),
                        "oracle_z4_writeback": host_logits(model, raw_observed, oracle_delta, output_size),
                        "clean_c4": direct_logits(model, raw_clean, output_size),
                    }
                for name, value in logits.items():
                    update_confusion_matrix(confusion[name], value.argmax(1).squeeze(0).cpu(), mask)
                z4_sse["corrupted"] += float((observation.z4.float() - clean.z4.float()).square().mean().item())
                z4_sse["learned"] += float((learned.z4.float() - clean.z4.float()).square().mean().item())
                z4_sse["oracle"] += 0.0
                frames += 1
                with torch.inference_mode():
                    pending, h4, h1 = predictor.predict_next(observation, prediction_error, h4, h1)
        mious = {name: float(torch.nanmean(compute_iou(value)).item()) for name, value in confusion.items()}
        corrupt, learned, oracle, clean_c4 = (mious[k] for k in ("corrupted", "learned", "oracle_z4_writeback", "clean_c4"))
        oracle_gap = oracle - corrupt
        clean_gap = clean_c4 - corrupt
        results[condition] = {
            "effective_frame_count": frames,
            "mIoU": mious,
            "z4_mse": {k: v / max(frames, 1) for k, v in z4_sse.items()},
            "writeback_transfer_ratio": oracle_gap / clean_gap if abs(clean_gap) > 1e-12 else None,
            "learned_transfer_efficiency": (learned - corrupt) / oracle_gap if abs(oracle_gap) > 1e-12 else None,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _ = load_components(STATIC_CHECKPOINT_DEFAULT, ADAPTER_CHECKPOINT_DEFAULT, payload["source_dynamics_checkpoint"], WRITEBACK_CHECKPOINT_DEFAULT)
    predictor = ErrorRegulatedSemanticRestorationPredictor(use_error_temporal_stats=payload.get("use_error_temporal_stats", False)).cuda()
    predictor.load_state_dict(payload["model_state_dict"], strict=True)
    predictor.freeze_dynamics(); model.requires_grad_(False); model.eval(); predictor.eval()
    all_groups = sequence_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val"))
    missing = [sequence for sequence in FAST_VALIDATION_SEQUENCES if sequence not in all_groups]
    if missing:
        raise RuntimeError(f"Missing fixed fast-validation sequences: {missing}")
    groups = {sequence: all_groups[sequence] for sequence in FAST_VALIDATION_SEQUENCES}
    results = run(model, predictor, groups)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    summary = {"experiment": "semantic_v3_temporal_stats_fast_miou_bottleneck", "checkpoint": args.checkpoint, "checkpoint_epoch": payload.get("epoch"), "sequences": FAST_VALIDATION_SEQUENCES, "conditions": CONDITIONS, "sigma": SIGMA, "results": results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
