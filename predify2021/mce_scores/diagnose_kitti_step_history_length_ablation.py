"""Measure how much Stage-T ConvGRU benefits from long hidden-state history.

The frozen FAST-B interface, Stage-T encoder and predictor are unchanged.  A
single Full9 pass shares backbone features and RAFT flows while evaluating
hidden resets every K frames for K={1,2,4,8,16,32,Full}.  Only the hidden state
is reset; the causal state/error and predictor weights are never changed.
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
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import VideoConsistency
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    load_components,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT, flow_grid
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    AuxiliaryTemporalPredictor,
    AuxiliaryTemporalStateEncoder,
    HostFeature,
    UnifiedFeatures,
)


FULL9 = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
RESET_LENGTHS = (1, 2, 4, 8, 16, 32, None)
FAST_B_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_semantic_v3_joint_c4_fast_ab_5754714/"
    "fast_b_joint_c4_weak_z4/best.pt"
)
STAGE_T_CHECKPOINT_DEFAULT = "/home/lin/predify/experiments/kitti_step_v2_auxiliary_stage_t/best.pt"


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
    return image, raw, observation, tuple(image.shape[-2:])


def probe_logits(model, raw, observation, state, output_size):
    # Match Stage-T's residual writeback exactly: decode the state delta and
    # subtract the zero-delta conditioned baseline.
    zero = UnifiedFeatures(
        torch.zeros_like(observation.z1), torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3), torch.zeros_like(observation.z4),
    )
    delta = UnifiedFeatures(
        torch.zeros_like(observation.z1), torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3), state - observation.z4,
    )
    applied = model.decode_conditioned_adapter_deltas(raw, delta)
    baseline = model.decode_conditioned_adapter_deltas(raw, zero)
    host_feature = HostFeature(raw.c4 + applied.c4 - baseline.c4, raw.c1, output_size)
    return model.decode_from_host_feature(host_feature)


def mtc(previous_prediction, current_prediction, flow):
    grid, valid = flow_grid(flow, current_prediction.shape[-2], current_prediction.shape[-1])
    warped = F.grid_sample(
        previous_prediction.float().unsqueeze(0), grid, mode="nearest",
        padding_mode="zeros", align_corners=True,
    ).squeeze(0).squeeze(0).to(torch.int64)
    keep = valid.squeeze(0)
    old, current = warped[keep].cpu(), current_prediction.squeeze(0)[keep].cpu()
    if not old.numel():
        return float("nan")
    confusion = torch.bincount(19 * old + current, minlength=19 * 19).reshape(19, 19)
    return float(torch.nanmean(compute_iou(confusion)).item())


def init_metric():
    return {
        "confusion": torch.zeros((19, 19), dtype=torch.int64),
        "vc": VideoConsistency(),
        "mtc_sum": 0.0,
        "mtc_count": 0,
        "pred_mse": 0.0,
        "copy_mse": 0.0,
        "true_delta": 0.0,
        "pred_motion": 0.0,
        "pairs": 0,
    }


def finish_metric(metric, vc_sums, vc_counts):
    iou = compute_iou(metric["confusion"])
    vc_stats = metric["vc"].stats()
    for length in (8, 16):
        vc_sums[length] += vc_stats[length]["sum"]
        vc_counts[length] += vc_stats[length]["count"]
    pairs = max(metric["pairs"], 1)
    return {
        "mIoU": float(torch.nanmean(iou).item()),
        "mVC8": metric["vc"].values()[8],
        "mVC16": metric["vc"].values()[16],
        "mTC": metric["mtc_sum"] / max(metric["mtc_count"], 1),
        "Rpred": metric["pred_mse"] / max(metric["copy_mse"], 1e-12),
        "pred_motion_ratio": math.sqrt(metric["pred_motion"] / pairs) / math.sqrt(metric["true_delta"] / pairs + 1e-12),
        "valid_frame_pairs": metric["mtc_count"],
    }


@torch.inference_mode()
def evaluate(model, encoder, predictor, groups, raft):
    labels = {str(k) if k is not None else "Full": k for k in RESET_LENGTHS}
    global_metrics = {name: init_metric() for name in labels}
    global_vc_sums = {name: {8: 0.0, 16: 0.0} for name in labels}
    global_vc_counts = {name: {8: 0, 16: 0} for name in labels}
    per_sequence = {}
    for sequence in FULL9:
        metrics = {name: init_metric() for name in labels}
        previous_image = None
        previous_state = None
        previous_predictions = {}
        pending = {}
        hidden = {}
        for index, sample in enumerate(groups[sequence]):
            image, raw, observation, output_size = encode(model, sample)
            state = encoder(observation.z4)
            mask = semantic_mask_from_panoptic_png(sample["mask_path"])
            if index == 0:
                for name in labels:
                    pending[name], hidden[name] = predictor.predict_next(
                        state, torch.zeros_like(state), None
                    )
            predictions = {}
            for name in labels:
                predicted_state = state if index == 0 else pending[name]
                predictions[name] = probe_logits(
                    model, raw, observation, predicted_state, output_size
                ).argmax(1)
                metric = metrics[name]
                global_metric = global_metrics[name]
                update_confusion_matrix(metric["confusion"], predictions[name].squeeze(0).cpu(), mask)
                metric["vc"].update(mask, predictions[name])
                update_confusion_matrix(global_metric["confusion"], predictions[name].squeeze(0).cpu(), mask)
            if previous_image is not None:
                flow = raft.current_to_previous(image, previous_image)
                for name, reset_length in labels.items():
                    metric = metrics[name]; global_metric = global_metrics[name]
                    score = mtc(previous_predictions[name], predictions[name], flow)
                    if math.isfinite(score):
                        metric["mtc_sum"] += score; metric["mtc_count"] += 1
                        global_metric["mtc_sum"] += score; global_metric["mtc_count"] += 1
                    pred_mse = F.mse_loss(pending[name], state).item()
                    copy_mse = F.mse_loss(previous_state, state).item()
                    true_delta = (state - previous_state).square().mean().item()
                    pred_motion = (pending[name] - previous_state).square().mean().item()
                    for target in (metric, global_metric):
                        target["pred_mse"] += pred_mse; target["copy_mse"] += copy_mse
                        target["true_delta"] += true_delta; target["pred_motion"] += pred_motion
                        target["pairs"] += 1
                    error = state - pending[name]
                    reset_hidden = reset_length is not None and index % reset_length == 0
                    next_hidden = None if reset_hidden else hidden[name]
                    pending[name], hidden[name] = predictor.predict_next(state, error, next_hidden)
            previous_image = image
            previous_state = state
            previous_predictions = predictions
        per_sequence[sequence] = {
            name: finish_metric(metric, global_vc_sums[name], global_vc_counts[name])
            for name, metric in metrics.items()
        }
    global_result = {}
    for name, metric in global_metrics.items():
        global_result[name] = finish_metric(metric, global_vc_sums[name], global_vc_counts[name])
        global_result[name]["mVC8"] = global_vc_sums[name][8] / max(global_vc_counts[name][8], 1)
        global_result[name]["mVC16"] = global_vc_sums[name][16] / max(global_vc_counts[name][16], 1)
    return {"global": global_result, "per_sequence": per_sequence}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--stage-t-checkpoint", default=STAGE_T_CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default="results/kitti_step_history_length_ablation.json")
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
        "experiment": "Stage-T ConvGRU hidden history length ablation",
        "full9": FULL9,
        "reset_lengths": {str(k) if k is not None else "Full": k for k in RESET_LENGTHS},
        "fast_b_checkpoint": args.fast_b_checkpoint,
        "stage_t_checkpoint": args.stage_t_checkpoint,
        "results": evaluate(model, encoder, predictor, sequence_groups(dataset), FrozenRAFT()),
    }
    output = Path(args.result_output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
