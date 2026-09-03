"""Inference-only upper bound for direct Z4 prediction-error utility.

For each clean KITTI-STEP frame, the frozen Stage-P predictor supplies
``e_t = Z4_t - Z4hat_t``.  The semantic state, correction head, and gate are
not used: the existing frozen C4 writeback/decoder receives ``Z4_t + beta*e_t``.
The beta sweep and a noncausal per-frame beta oracle are evaluated on Full9.
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
from predify2021.mce_scores.evaluate_kitti_step_raft_error_memory import FrozenRAFT
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import (
    VideoConsistency,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    load_components,
    residual_writeback_host_feature,
)
from predify2021.mce_scores.train_kitti_step_z4_only import (
    FAST_B_DEFAULT,
    z4_predict_next,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorRegulatedSemanticRestorationPredictor,
    HostFeature,
    UnifiedFeatures,
)
from predify2021.mce_scores.train_kitti_step_z4_only_stage_c import (
    STAGE_P_DEFAULT,
)

NUM_CLASSES = 19
IGNORE = 255
BETAS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
SEQUENCES = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
CHECKPOINT_DEFAULT = "/home/lin/predify/experiments/kitti_step_z4_only_stage_c_balance/best.pt"
RESULT_DEFAULT = "results/kitti_step_prediction_error_utility_upper_bound"


def load_frozen_stage_c(checkpoint_path, fast_b, stage_p, dynamics):
    model, _ = load_components(
        STATIC_CHECKPOINT_DEFAULT,
        ADAPTER_CHECKPOINT_DEFAULT,
        dynamics,
        WRITEBACK_CHECKPOINT_DEFAULT,
    )
    fast_payload = torch.load(fast_b, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.output_adapters[3].load_state_dict(
        fast_payload["c4_output_adapter_state_dict"], strict=True
    )
    model.host_conditioned_writebacks["3"].load_state_dict(
        fast_payload["c4_writeback_state_dict"], strict=True
    )
    stage_payload = torch.load(stage_p, map_location="cpu", weights_only=False)
    predictor = ErrorRegulatedSemanticRestorationPredictor(
        use_error_temporal_stats=True
    ).cuda()
    predictor.load_state_dict(stage_payload["model_state_dict"], strict=True)
    # The Stage-C checkpoint contains the unchanged predictor plus semantic
    # modules. Loading it preserves the exact Epoch-3 inference weights, but
    # this diagnostic never invokes the semantic branch.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    predictor.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().cuda()
    predictor.eval().requires_grad_(False)
    return model, predictor, payload


def decode_beta_batch(model, raw, observation, error_z4, output_size):
    """Decode all beta variants in one batched frozen writeback/decoder call."""
    count = len(BETAS)
    delta = UnifiedFeatures(
        torch.zeros_like(observation.z1).expand(count, -1, -1, -1),
        torch.zeros_like(observation.z2).expand(count, -1, -1, -1),
        torch.zeros_like(observation.z3).expand(count, -1, -1, -1),
        torch.cat(tuple(float(beta) * error_z4 for beta in BETAS), dim=0),
    )
    updated = residual_writeback_host_feature(model, raw, delta, output_size)
    return model.decode_from_host_feature(updated)


def direct_metrics(groups, model, predictor, raft):
    names = tuple(f"beta_{beta:g}" for beta in BETAS) + ("oracle",)
    confusion = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in names}
    mvc_sums = {name: {8: 0.0, 16: 0.0} for name in names}
    mvc_counts = {name: {8: 0, 16: 0} for name in names}
    mtc_sum = {name: 0.0 for name in names}
    mtc_count = {name: 0 for name in names}
    per_sequence = {}

    for sequence in SEQUENCES:
        samples = groups[sequence]
        seq_confusion = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in names}
        seq_vc = {name: VideoConsistency() for name in names}
        seq_mtc_sum = {name: 0.0 for name in names}
        seq_mtc_count = {name: 0 for name in names}
        previous_image = None
        previous_predictions = None

        image, observation, raw, output_size = _encode(model, samples[0])
        host_logits = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))
        mask = semantic_mask_from_panoptic_png(samples[0]["mask_path"])
        predictions = {name: host_logits.argmax(1) for name in names}
        _accumulate(names, predictions, mask, confusion, seq_confusion, seq_vc)
        previous_image = image
        previous_predictions = {name: pred.detach() for name, pred in predictions.items()}

        pending_z4, hidden = z4_predict_next(
            predictor, observation.z4, torch.zeros_like(observation.z4), None
        )

        for sample in samples[1:]:
            image, observation, raw, output_size = _encode(model, sample)
            mask = semantic_mask_from_panoptic_png(sample["mask_path"])
            host_logits = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))
            error_z4 = observation.z4 - pending_z4
            logits = decode_beta_batch(model, raw, observation, error_z4, output_size)

            target = mask.to(logits.device, non_blocking=True)
            valid = target != IGNORE
            if valid.any():
                losses = F.cross_entropy(
                    logits,
                    target.unsqueeze(0).expand(logits.shape[0], -1, -1),
                    ignore_index=IGNORE,
                    reduction="none",
                )[:, valid].mean(1)
                oracle_index = int(losses.argmin().item())
            else:
                oracle_index = BETAS.index(0.0)
            beta_predictions = {
                name: logits[index:index + 1].argmax(1)
                for index, name in enumerate(names[:-1])
            }
            beta_predictions["oracle"] = logits[oracle_index:oracle_index + 1].argmax(1)
            predictions = beta_predictions
            _accumulate(names, predictions, mask, confusion, seq_confusion, seq_vc)

            backward_flow = raft.backward_flow(image, previous_image)
            for name in names:
                score = _pair_mtc(previous_predictions[name], predictions[name], backward_flow)
                if math.isfinite(score):
                    mtc_sum[name] += score
                    mtc_count[name] += 1
                    seq_mtc_sum[name] += score
                    seq_mtc_count[name] += 1
            previous_image = image
            previous_predictions = {name: pred.detach() for name, pred in predictions.items()}
            pending_z4, hidden = z4_predict_next(predictor, observation.z4, error_z4, hidden)

        for name in names:
            stats = seq_vc[name].stats()
            for length in (8, 16):
                mvc_sums[name][length] += stats[length]["sum"]
                mvc_counts[name][length] += stats[length]["count"]
        per_sequence[sequence] = {
            name: {
                "mIoU": float(torch.nanmean(compute_iou(seq_confusion[name])).item()),
                "mVC8": seq_vc[name].values()[8],
                "mVC16": seq_vc[name].values()[16],
                "mTC": seq_mtc_sum[name] / max(seq_mtc_count[name], 1),
                "valid_frame_pairs": seq_mtc_count[name],
            }
            for name in names
        }

    metrics = {}
    for name in names:
        metrics[name] = {
            "mIoU": float(torch.nanmean(compute_iou(confusion[name])).item()),
            "mVC8": mvc_sums[name][8] / max(mvc_counts[name][8], 1),
            "mVC16": mvc_sums[name][16] / max(mvc_counts[name][16], 1),
            "mTC": mtc_sum[name] / max(mtc_count[name], 1),
            "valid_frame_pairs": mtc_count[name],
        }
    host = metrics["beta_0"]
    for name in names:
        metrics[name]["delta_vs_host"] = {
            key: metrics[name][key] - host[key]
            for key in ("mIoU", "mVC8", "mVC16", "mTC")
        }
    metrics["per_sequence"] = per_sequence
    metrics["host_variant"] = "beta_0"
    return metrics


def _encode(model, sample):
    image = load_image(sample)
    raw = model.extract_backbone_features(image)
    observation = model.encode_backbone_features(raw)
    return image, observation, raw, tuple(image.shape[-2:])


def _accumulate(names, predictions, mask, confusion, seq_confusion, seq_vc):
    for name in names:
        prediction = predictions[name][0].cpu()
        update_confusion_matrix(confusion[name], prediction, mask)
        update_confusion_matrix(seq_confusion[name], prediction, mask)
        seq_vc[name].update(mask, prediction)


def _pair_mtc(previous_prediction, current_prediction, backward_flow):
    height, width = current_prediction.shape[-2:]
    flow = backward_flow
    source_h, source_w = flow.shape[-2:]
    if (source_h, source_w) != (height, width):
        flow = F.interpolate(flow, size=(height, width), mode="bilinear", align_corners=True)
        flow = flow.clone()
        flow[:, 0].mul_(width / source_w)
        flow[:, 1].mul_(height / source_h)
    y, x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    source_x = x.unsqueeze(0) + flow[:, 0]
    source_y = y.unsqueeze(0) + flow[:, 1]
    valid = (
        (source_x >= 0) & (source_x <= width - 1)
        & (source_y >= 0) & (source_y <= height - 1)
    )
    grid = torch.stack(
        (2.0 * source_x / max(width - 1, 1) - 1.0,
         2.0 * source_y / max(height - 1, 1) - 1.0), dim=-1
    )
    warped = F.grid_sample(
        previous_prediction.float().unsqueeze(1), grid, mode="nearest",
        padding_mode="zeros", align_corners=True,
    )[0, 0].long()
    keep = valid[0]
    a, b = warped[keep].cpu(), current_prediction[0][keep].cpu()
    if not a.numel():
        return float("nan")
    confusion = torch.bincount(NUM_CLASSES * a + b, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    return float(torch.nanmean(compute_iou(confusion)).item())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_DEFAULT)
    parser.add_argument("--stage-p-checkpoint", default=STAGE_P_DEFAULT)
    parser.add_argument("--dynamics-checkpoint", default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("Prediction-error utility evaluation requires CUDA")
    model, predictor, payload = load_frozen_stage_c(
        args.checkpoint, args.fast_b_checkpoint, args.stage_p_checkpoint, args.dynamics_checkpoint
    )
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    groups = sequence_groups(dataset)
    groups = {sequence: groups[sequence] for sequence in SEQUENCES}
    raft = FrozenRAFT()
    with torch.inference_mode():
        metrics = direct_metrics(groups, model, predictor, raft)
    result = {
        "experiment": "Prediction Error Utility Upper Bound",
        "inference_only": True,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": payload.get("epoch"),
        "formula": "Z4_test = Z4 + beta * (Z4 - Z4_prediction)",
        "betas": list(BETAS),
        "oracle": "per-frame minimum segmentation cross-entropy over beta candidates",
        "sequences": list(SEQUENCES),
        "valid_frame_pairs": {sequence: len(groups[sequence]) - 1 for sequence in SEQUENCES},
        "metrics": metrics,
    }
    output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text(
        "# Prediction Error Utility Upper Bound\n\n"
        "Inference-only Full9 diagnostic. The frozen Stage-C Epoch-3 checkpoint is unchanged; "
        "semantic state, correction head, and gate are bypassed. Existing frozen C4 writeback "
        "and decoder receive `Z4 + beta * (Z4 - Z4_prediction)`. The oracle selects beta per "
        "frame using ground-truth segmentation cross-entropy and is noncausal.\n"
    )
    print(json.dumps({"result": str(output / "summary.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
