"""Measure whether Stage-T prediction error identifies Host segmentation errors.

This is an inference-only diagnostic.  For each causal frame t, the error
magnitude ||Z4_t - T_hat_t|| is compared with the frozen Host prediction's
pixel correctness.  Statistics are reported globally and in semantic
boundary/interior and RAFT motion/static regions.  GT is used only to form
the offline diagnostic labels; it is never an input to the predictor.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    load_components,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    AuxiliaryTemporalPredictor,
    AuxiliaryTemporalStateEncoder,
    HostFeature,
)


FULL9 = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
FAST_B_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_semantic_v3_joint_c4_fast_ab_5754714/"
    "fast_b_joint_c4_weak_z4/best.pt"
)
STAGE_T_CHECKPOINT_DEFAULT = "/home/lin/predify/experiments/kitti_step_v2_auxiliary_stage_t/best.pt"
MOTION_THRESHOLD_PX = 1.0
REGIONS = ("all", "boundary", "interior", "motion", "static")


def load_fast_b_model(path):
    model, _ = load_components(
        STATIC_CHECKPOINT_DEFAULT, ADAPTER_CHECKPOINT_DEFAULT,
        ROLE_PREDICTOR_CHECKPOINT_DEFAULT, WRITEBACK_CHECKPOINT_DEFAULT,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.output_adapters[3].load_state_dict(
        payload["c4_output_adapter_state_dict"], strict=True
    )
    model.host_conditioned_writebacks["3"].load_state_dict(
        payload["c4_writeback_state_dict"], strict=True
    )
    model.requires_grad_(False).eval()
    return model


def boundary_mask(mask):
    valid = mask != 255
    result = torch.zeros_like(valid)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = torch.full_like(mask, 255)
        ys = slice(max(0, dy), min(mask.shape[0], mask.shape[0] + dy))
        xs = slice(max(0, dx), min(mask.shape[1], mask.shape[1] + dx))
        ysrc = slice(max(0, -dy), min(mask.shape[0], mask.shape[0] - dy))
        xsrc = slice(max(0, -dx), min(mask.shape[1], mask.shape[1] - dx))
        shifted[ys, xs] = mask[ysrc, xsrc]
        result |= valid & (shifted != 255) & (shifted != mask)
    return result


def resize_mask(mask, size):
    return F.interpolate(mask.float()[None, None], size=size, mode="nearest")[0, 0].bool()


def auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positive = labels == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    rank_sum = ranks[order][positive].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def region_stats(scores, wrong):
    scores = np.asarray(scores, dtype=np.float64)
    wrong = np.asarray(wrong, dtype=bool)
    correct = ~wrong
    result = {
        "count": int(scores.size),
        "wrong_rate": float(wrong.mean()) if scores.size else float("nan"),
        "correct_mean_error": float(scores[correct].mean()) if correct.any() else float("nan"),
        "wrong_mean_error": float(scores[wrong].mean()) if wrong.any() else float("nan"),
        "auroc_error_to_wrong": auc(scores, wrong.astype(np.int64)),
    }
    if scores.size:
        threshold = np.quantile(scores, 0.90)
        top = scores >= threshold
        result["top10_error_threshold"] = float(threshold)
        result["top10_count"] = int(top.sum())
        result["top10_wrong_rate"] = float(wrong[top].mean()) if top.any() else float("nan")
    else:
        result.update({"top10_error_threshold": float("nan"), "top10_count": 0, "top10_wrong_rate": float("nan")})
    return result


@torch.inference_mode()
def evaluate(model, encoder, predictor, groups, raft):
    samples_by_region = {region: {"scores": [], "wrong": []} for region in REGIONS}
    per_sequence = {}
    for sequence in FULL9:
        seq_data = {region: {"scores": [], "wrong": []} for region in REGIONS}
        previous_image = None
        previous_state = None
        pending = None
        hidden = None
        for index, sample in enumerate(groups[sequence]):
            image = load_image(sample)
            raw = model.extract_backbone_features(image)
            observation = model.encode_backbone_features(raw)
            state = encoder(observation.z4)
            output_size = tuple(image.shape[-2:])
            host_prediction = model.decode_from_host_feature(
                HostFeature(raw.c4, raw.c1, output_size)
            ).argmax(1).squeeze(0).cpu()
            gt = semantic_mask_from_panoptic_png(sample["mask_path"])
            if index == 0:
                pending, hidden = predictor.predict_next(state, torch.zeros_like(state), None)
            else:
                flow = raft.current_to_previous(image, previous_image)
                error_magnitude = (state - pending).square().sum(dim=1).sqrt()[0]
                valid = gt != 255
                host_wrong_image = valid & (host_prediction != gt)
                boundary_image = boundary_mask(gt)
                motion_image = (
                    flow.square().sum(dim=1).sqrt()[0] > MOTION_THRESHOLD_PX
                ).cpu()
                feature_size = error_magnitude.shape
                masks = {
                    "all": resize_mask(valid, feature_size),
                    "boundary": resize_mask(valid & boundary_image, feature_size),
                    "interior": resize_mask(valid & ~boundary_image, feature_size),
                    "motion": resize_mask(valid & motion_image, feature_size),
                    "static": resize_mask(valid & ~motion_image, feature_size),
                }
                wrong = resize_mask(host_wrong_image, feature_size)
                for region, region_mask in masks.items():
                    scores = error_magnitude[region_mask].float().cpu().numpy()
                    errors = wrong[region_mask].cpu().numpy().astype(bool)
                    samples_by_region[region]["scores"].append(scores)
                    samples_by_region[region]["wrong"].append(errors)
                    seq_data[region]["scores"].append(scores)
                    seq_data[region]["wrong"].append(errors)
                error = state - pending
                pending, hidden = predictor.predict_next(state, error, hidden)
            previous_image = image
            previous_state = state
        per_sequence[sequence] = {
            region: region_stats(
                np.concatenate(data["scores"]) if data["scores"] else np.empty(0),
                np.concatenate(data["wrong"]) if data["wrong"] else np.empty(0, dtype=bool),
            )
            for region, data in seq_data.items()
        }
    global_result = {
        region: region_stats(
            np.concatenate(data["scores"]) if data["scores"] else np.empty(0),
            np.concatenate(data["wrong"]) if data["wrong"] else np.empty(0, dtype=bool),
        )
        for region, data in samples_by_region.items()
    }
    return {"global": global_result, "per_sequence": per_sequence}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--stage-t-checkpoint", default=STAGE_T_CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default="results/kitti_step_error_task_relevance.json")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = load_fast_b_model(args.fast_b_checkpoint)
    payload = torch.load(args.stage_t_checkpoint, map_location="cpu", weights_only=False)
    encoder = AuxiliaryTemporalStateEncoder().cuda(); encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    predictor = AuxiliaryTemporalPredictor().cuda(); predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    encoder.requires_grad_(False).eval(); predictor.requires_grad_(False).eval()
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    result = {
        "experiment": "Prediction Error-Task Relevance diagnostic",
        "full9": FULL9,
        "motion_threshold_px": MOTION_THRESHOLD_PX,
        "fast_b_checkpoint": args.fast_b_checkpoint,
        "stage_t_checkpoint": args.stage_t_checkpoint,
        "results": evaluate(model, encoder, predictor, sequence_groups(dataset), FrozenRAFT()),
    }
    output = Path(args.result_output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
