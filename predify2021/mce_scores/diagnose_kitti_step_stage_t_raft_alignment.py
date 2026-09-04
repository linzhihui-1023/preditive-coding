"""Oracle RAFT alignment diagnostic for the frozen Stage-T predictor.

No weights are changed.  The only diagnostic intervention is warping the
previous ConvGRU hidden state into the current-frame coordinates before the
recurrent update.  Standard and aligned streams share the same Full9 frames
and frozen RAFT evaluator.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import VideoConsistency
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT, ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT, WRITEBACK_CHECKPOINT_DEFAULT,
    load_components, residual_writeback_host_feature,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT, flow_grid
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    AuxiliaryTemporalPredictor, AuxiliaryTemporalStateEncoder, HostFeature, UnifiedFeatures,
)


FULL9 = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
NUM_CLASSES = 19
DEFAULT_CHECKPOINT = "/home/lin/predify/experiments/kitti_step_v2_auxiliary_stage_t/best.pt"


def encode_clean(model, sample):
    image = load_image(sample)
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        observation = model.encode_backbone_features(raw)
    return image, observation, raw, tuple(image.shape[-2:])


def probe_logits(model, raw, observation, state, output_size):
    delta = UnifiedFeatures(
        torch.zeros_like(observation.z1), torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3), state - observation.z4,
    )
    return model.decode_from_host_feature(
        residual_writeback_host_feature(model, raw, delta, output_size)
    )


def warp_hidden(hidden, flow):
    grid, valid = flow_grid(flow, hidden.shape[-2], hidden.shape[-1])
    warped = F.grid_sample(hidden, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped * valid.unsqueeze(1).to(warped.dtype)


def add_mtc(previous_prediction, current_prediction, flow):
    grid, valid = flow_grid(flow, current_prediction.shape[-2], current_prediction.shape[-1])
    warped = F.grid_sample(
        previous_prediction.float().unsqueeze(0), grid, mode="nearest",
        padding_mode="zeros", align_corners=True,
    ).squeeze(0).squeeze(0).to(torch.int64)
    keep = valid.squeeze(0)
    a, b = warped[keep].cpu(), current_prediction.squeeze(0)[keep].cpu()
    if not a.numel():
        return float("nan")
    confusion = torch.bincount(NUM_CLASSES * a + b, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    return float(torch.nanmean(compute_iou(confusion)).item())


@torch.inference_mode()
def evaluate(model, encoder, predictor, groups, raft, aligned):
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    vc_sums = {8: 0.0, 16: 0.0}; vc_counts = {8: 0, 16: 0}
    mtc_sum = 0.0; mtc_count = 0
    temporal = {key: 0.0 for key in ("pred_mse", "copy_mse", "true_delta", "state_sq", "raw_delta", "pred_motion")}
    temporal_pairs = 0; per_sequence = {}
    encoder.eval(); predictor.eval()
    for sequence in FULL9:
        samples = groups[sequence]
        seq_confusion = torch.zeros_like(confusion); seq_vc = VideoConsistency()
        seq_mtc_sum = 0.0; seq_mtc_count = 0
        previous_image = None; previous_prediction = None; previous_state = None; previous_observation = None
        pending = None; hidden = None
        for index, sample in enumerate(samples):
            image, observation, raw, output_size = encode_clean(model, sample)
            state = encoder(observation.z4)
            if index == 0:
                pending, hidden = predictor.predict_next(state, torch.zeros_like(state), None)
                predicted_state = state
            else:
                predicted_state = pending
            prediction = probe_logits(model, raw, observation, predicted_state, output_size).argmax(1)
            mask = semantic_mask_from_panoptic_png(sample["mask_path"])
            update_confusion_matrix(confusion, prediction.squeeze(0).cpu(), mask)
            update_confusion_matrix(seq_confusion, prediction.squeeze(0).cpu(), mask)
            seq_vc.update(mask, prediction)
            if previous_image is not None:
                flow = raft.current_to_previous(image, previous_image)
                score = add_mtc(previous_prediction, prediction, flow)
                if math.isfinite(score):
                    mtc_sum += score; mtc_count += 1; seq_mtc_sum += score; seq_mtc_count += 1
                temporal["pred_mse"] += float(F.mse_loss(pending, state).item())
                temporal["copy_mse"] += float(F.mse_loss(previous_state, state).item())
                temporal["true_delta"] += float((state - previous_state).square().mean().item())
                temporal["state_sq"] += float(state.square().mean().item())
                temporal["raw_delta"] += float((observation.z4 - previous_observation).square().mean().item())
                temporal["pred_motion"] += float((pending - previous_state).square().mean().item())
                temporal_pairs += 1
                error = state - pending
                next_hidden_input = warp_hidden(hidden, flow) if aligned else hidden
                pending, hidden = predictor.predict_next(state, error, next_hidden_input)
            previous_image = image; previous_prediction = prediction; previous_state = state; previous_observation = observation.z4
        stats = seq_vc.stats()
        for length in (8, 16):
            vc_sums[length] += stats[length]["sum"]; vc_counts[length] += stats[length]["count"]
        iou = compute_iou(seq_confusion)
        per_sequence[sequence] = {
            "mIoU": float(torch.nanmean(iou).item()),
            "mVC8": seq_vc.values()[8], "mVC16": seq_vc.values()[16],
            "mTC": seq_mtc_sum / seq_mtc_count if seq_mtc_count else float("nan"),
        }
    pred_mse = temporal["pred_mse"] / max(temporal_pairs, 1); copy_mse = temporal["copy_mse"] / max(temporal_pairs, 1)
    true_delta = temporal["true_delta"] / max(temporal_pairs, 1); state_sq = temporal["state_sq"] / max(temporal_pairs, 1)
    raw_delta = temporal["raw_delta"] / max(temporal_pairs, 1); pred_motion = temporal["pred_motion"] / max(temporal_pairs, 1)
    return {
        "Rpred": pred_mse / max(copy_mse, 1e-12),
        "pred_motion_ratio": math.sqrt(pred_motion) / math.sqrt(true_delta + 1e-12),
        "mIoU": float(torch.nanmean(compute_iou(confusion)).item()),
        "mTC": mtc_sum / mtc_count if mtc_count else float("nan"),
        "mVC8": vc_sums[8] / vc_counts[8] if vc_counts[8] else float("nan"),
        "mVC16": vc_sums[16] / vc_counts[16] if vc_counts[16] else float("nan"),
        "dynamic_ratio": math.sqrt(true_delta) / math.sqrt(raw_delta + 1e-12),
        "per_sequence": per_sequence, "valid_frame_pairs": mtc_count,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--result-output", default="results/kitti_step_stage_t_raft_alignment.json")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model, _ = load_components(STATIC_CHECKPOINT_DEFAULT, ADAPTER_CHECKPOINT_DEFAULT, ROLE_PREDICTOR_CHECKPOINT_DEFAULT, WRITEBACK_CHECKPOINT_DEFAULT)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = AuxiliaryTemporalStateEncoder().cuda(); encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    predictor = AuxiliaryTemporalPredictor().cuda(); predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    model.requires_grad_(False).eval(); encoder.requires_grad_(False).eval(); predictor.requires_grad_(False).eval()
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    groups = sequence_groups(dataset); raft = FrozenRAFT()
    standard = evaluate(model, encoder, predictor, groups, raft, aligned=False)
    aligned = evaluate(model, encoder, predictor, groups, raft, aligned=True)
    result = {
        "experiment": "Stage-T Oracle RAFT hidden alignment diagnostic", "checkpoint": args.checkpoint,
        "full9": FULL9, "standard": standard, "raft_aligned": aligned,
        "delta_aligned_minus_standard": {
            key: aligned[key] - standard[key] for key in ("Rpred", "pred_motion_ratio", "mIoU", "mTC")
        },
    }
    output = Path(args.result_output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
