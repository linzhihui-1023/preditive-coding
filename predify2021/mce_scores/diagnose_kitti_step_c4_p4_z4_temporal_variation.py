"""Diagnose temporal variation preserved by the C4 -> P4 -> Z4 -> T path.

This is inference-only.  FAST-B's C4 adapter/writeback and the frozen Stage-T
temporal encoder are loaded exactly as in the Stage-T evaluator.  For each
Full9 adjacent-frame pair the script reports
||F_t - F_{t-1}|| / ||F_t|| for raw C4, pre-normalization P4, normalized Z4,
and learned temporal state T.  Region summaries split static/moving pixels
using frozen RAFT and semantic boundary/interior pixels using the annotation.
No model parameters are changed.
"""

import argparse
import json
import math
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
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    load_components,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50 import AuxiliaryTemporalStateEncoder


FULL9 = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
FAST_B_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_semantic_v3_joint_c4_fast_ab_5754714/"
    "fast_b_joint_c4_weak_z4/best.pt"
)
STAGE_T_CHECKPOINT_DEFAULT = "/home/lin/predify/experiments/kitti_step_v2_auxiliary_stage_t/best.pt"
REGIONS = ("all", "static", "motion", "boundary", "interior")
MOTION_THRESHOLD_PX = 1.0


def load_fast_b_model(fast_b_checkpoint):
    model, _ = load_components(
        STATIC_CHECKPOINT_DEFAULT,
        ADAPTER_CHECKPOINT_DEFAULT,
        ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
        WRITEBACK_CHECKPOINT_DEFAULT,
    )
    payload = torch.load(fast_b_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.output_adapters[3].load_state_dict(
        payload["c4_output_adapter_state_dict"], strict=True
    )
    model.host_conditioned_writebacks["3"].load_state_dict(
        payload["c4_writeback_state_dict"], strict=True
    )
    model.requires_grad_(False).eval()
    return model


@torch.inference_mode()
def encode(model, sample):
    image = load_image(sample)
    raw = model.extract_backbone_features(image)
    observation = model.encode_backbone_features(raw)
    adapter = model.multi_layer_adapter.input_adapters[3]
    p4 = adapter.projection(raw.c4)
    z4 = adapter.norm(p4)
    return image, raw.c4, p4, z4


def boundary_mask(mask):
    valid = mask != 255
    boundary = torch.zeros_like(valid)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = torch.full_like(mask, 255)
        ys = slice(max(0, dy), min(mask.shape[0], mask.shape[0] + dy))
        xs = slice(max(0, dx), min(mask.shape[1], mask.shape[1] + dx))
        ysrc = slice(max(0, -dy), min(mask.shape[0], mask.shape[0] - dy))
        xsrc = slice(max(0, -dx), min(mask.shape[1], mask.shape[1] - dx))
        shifted[ys, xs] = mask[ysrc, xsrc]
        boundary |= valid & (shifted != 255) & (shifted != mask)
    return boundary


def resize_region(region, size):
    return F.interpolate(
        region.float()[None, None], size=size, mode="nearest"
    )[0, 0].bool()


def variation_ratio(current, previous, region):
    if region is None:
        values_current = current.reshape(-1)
        values_delta = (current - previous).reshape(-1)
    else:
        values_current = current.permute(1, 2, 0)[region]
        values_delta = (current - previous).permute(1, 2, 0)[region]
    if values_current.numel() == 0:
        return float("nan")
    denominator = torch.linalg.vector_norm(values_current.float())
    if denominator <= 1e-12:
        return float("nan")
    return float((torch.linalg.vector_norm(values_delta.float()) / denominator).item())


def update_summary(bucket, value):
    if math.isfinite(value):
        bucket.append(value)


def summarize(values):
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "count": len(values),
    }


@torch.inference_mode()
def evaluate(model, encoder, groups, raft):
    global_values = {name: {region: [] for region in REGIONS} for name in ("C4", "P4", "Z4", "T")}
    per_sequence = {}
    encoder.eval()
    for sequence in FULL9:
        previous = None
        seq_values = {name: {region: [] for region in REGIONS} for name in global_values}
        motion_fractions = []
        for index, sample in enumerate(groups[sequence]):
            image, c4, p4, z4 = encode(model, sample)
            temporal = encoder(z4)
            if previous is not None:
                previous_image, previous_features, previous_mask = previous
                flow = raft.current_to_previous(image, previous_image)
                motion = flow.square().sum(dim=1).sqrt()[0] > MOTION_THRESHOLD_PX
                current_mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                boundary = boundary_mask(current_mask)
                static = ~motion
                feature_size = c4.shape[-2:]
                masks = {
                    "all": None,
                    "static": resize_region(static, feature_size),
                    "motion": resize_region(motion, feature_size),
                    "boundary": resize_region(boundary, feature_size),
                    "interior": resize_region((current_mask != 255) & ~boundary, feature_size),
                }
                motion_fractions.append(float(motion.float().mean().item()))
                current_features = {"C4": c4[0], "P4": p4[0], "Z4": z4[0], "T": temporal[0]}
                for name, current in current_features.items():
                    old = previous_features[name]
                    for region, region_mask in masks.items():
                        value = variation_ratio(current, old, region_mask)
                        update_summary(global_values[name][region], value)
                        update_summary(seq_values[name][region], value)
            previous = (
                image,
                {"C4": c4[0], "P4": p4[0], "Z4": z4[0], "T": temporal[0]},
                semantic_mask_from_panoptic_png(sample["mask_path"]),
            )
        per_sequence[sequence] = {
            name: {region: summarize(values) for region, values in regions.items()}
            for name, regions in seq_values.items()
        }
        per_sequence[sequence]["motion_fraction"] = summarize(motion_fractions)
    return {
        "features": {
            name: {region: summarize(values) for region, values in regions.items()}
            for name, regions in global_values.items()
        },
        "per_sequence": per_sequence,
        "valid_frame_pairs": sum(
            per_sequence[s]["C4"]["all"]["count"] for s in FULL9
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--stage-t-checkpoint", default=STAGE_T_CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default="results/kitti_step_c4_p4_z4_temporal_variation.json")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = load_fast_b_model(args.fast_b_checkpoint)
    stage_t = torch.load(args.stage_t_checkpoint, map_location="cpu", weights_only=False)
    encoder = AuxiliaryTemporalStateEncoder().cuda()
    encoder.load_state_dict(stage_t["encoder_state_dict"], strict=True)
    encoder.requires_grad_(False).eval()
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    groups = sequence_groups(dataset)
    result = {
        "experiment": "C4-P4-Z4-T temporal variation diagnostic",
        "full9": FULL9,
        "fast_b_checkpoint": args.fast_b_checkpoint,
        "stage_t_checkpoint": args.stage_t_checkpoint,
        "motion_threshold_px": MOTION_THRESHOLD_PX,
        "regions": REGIONS,
        "results": evaluate(model, encoder, groups, FrozenRAFT()),
    }
    output = Path(args.result_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
