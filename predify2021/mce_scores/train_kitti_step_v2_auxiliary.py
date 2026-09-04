"""V2-Auxiliary Stage T: train a semantic-anchor-aligned temporal state.

The Host, C4 adapter, writeback and decoder remain frozen.  Stage T trains only
the 128-channel residual temporal encoder and its causal ConvGRU predictor.  No
correction is enabled in this stage; decoded temporal probes are diagnostics.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import VideoConsistency
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    residual_writeback_host_feature,
    zero_state,
    load_components,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT, flow_grid
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    AuxiliaryTemporalPredictor,
    AuxiliaryTemporalStateEncoder,
    HostFeature,
    UnifiedFeatures,
)


SEED = 0
TBPTT_STEPS = 16
MAX_EPOCHS = 15
PATIENCE = 3
LR = 1e-5
WEIGHT_DECAY = 1e-2
LAMBDA_ANCHOR = 0.1
NUM_CLASSES = 19
IGNORE_LABEL = 255
FULL9 = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
FAST_B_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_semantic_v3_joint_c4_fast_ab_5754714/"
    "fast_b_joint_c4_weak_z4/best.pt"
)
OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_v2_auxiliary_stage_t"
RESULT_DEFAULT = "results/kitti_step_v2_auxiliary_stage_t"


def encode_clean(model, sample):
    image = load_image(sample)
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        observation = model.encode_backbone_features(raw)
    return image, observation, raw, tuple(image.shape[-2:])


def probe_logits(model, raw, observation, probe_state, output_size):
    delta = UnifiedFeatures(
        torch.zeros_like(observation.z1),
        torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3),
        probe_state - observation.z4,
    )
    feature = residual_writeback_host_feature(model, raw, delta, output_size)
    return model.decode_from_host_feature(feature)


def detach_tensor(value):
    return value.detach()


def load_training_components(args):
    model, _ = load_components(
        STATIC_CHECKPOINT_DEFAULT,
        ADAPTER_CHECKPOINT_DEFAULT,
        args.dynamics_checkpoint,
        WRITEBACK_CHECKPOINT_DEFAULT,
    )
    payload = torch.load(args.fast_b_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.output_adapters[3].load_state_dict(
        payload["c4_output_adapter_state_dict"], strict=True
    )
    model.host_conditioned_writebacks["3"].load_state_dict(
        payload["c4_writeback_state_dict"], strict=True
    )
    encoder = AuxiliaryTemporalStateEncoder().cuda()
    predictor = AuxiliaryTemporalPredictor().cuda()
    model.requires_grad_(False).eval()
    encoder.train(); predictor.train()
    return model, encoder, predictor, payload


def assert_freeze_contract(model, encoder, predictor):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("V2-Auxiliary Host/adapter/writeback/decoder must be frozen")
    if not all(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("Temporal encoder must be trainable")
    if not all(parameter.requires_grad for parameter in predictor.parameters()):
        raise RuntimeError("Temporal predictor must be trainable")


@torch.no_grad()
def zero_step_check(model, encoder, predictor, sample):
    _, observation, _, _ = encode_clean(model, sample)
    state = encoder(observation.z4)
    pending, _ = predictor.predict_next(state.detach(), torch.zeros_like(state), None)
    encoder_error = float((state - observation.z4).abs().max().item())
    prediction_error = float((pending - state).abs().max().item())
    if encoder_error > 1e-6 or prediction_error > 1e-6:
        raise RuntimeError(
            f"V2-Auxiliary E0 contract failed: encoder={encoder_error}, predictor={prediction_error}"
        )
    return {"state_equals_z4_max_abs": encoder_error, "pending_equals_state_max_abs": prediction_error}


def _rms_sum(value):
    return float(value.detach().square().mean().sqrt().item())


def train_sequence(model, encoder, predictor, samples, optimizer, max_steps=0):
    if len(samples) < 2:
        return {"frames": 0, "windows": 0, "prediction_loss": 0.0, "anchor_loss": 0.0, "total_loss": 0.0,
                "pred_mse_sum": 0.0, "copy_mse_sum": 0.0, "state_rms_sum": 0.0, "state_std_sum": 0.0,
                "true_delta_sq_sum": 0.0, "state_sq_sum": 0.0, "raw_delta_sq_sum": 0.0, "pred_motion_sq_sum": 0.0}
    limit = min(len(samples) - 1, max_steps) if max_steps else len(samples) - 1
    _, first_observation, _, _ = encode_clean(model, samples[0])
    previous_state = encoder(first_observation.z4)
    pending, hidden = predictor.predict_next(previous_state.detach(), torch.zeros_like(previous_state), None)
    losses = []
    totals = {"frames": 0, "windows": 0, "prediction_loss": 0.0, "anchor_loss": 0.0, "total_loss": 0.0,
              "pred_mse_sum": 0.0, "copy_mse_sum": 0.0, "state_rms_sum": 0.0, "state_std_sum": 0.0,
              "true_delta_sq_sum": 0.0, "state_sq_sum": 0.0, "raw_delta_sq_sum": 0.0, "pred_motion_sq_sum": 0.0}
    for offset in range(limit):
        _, observation, _, _ = encode_clean(model, samples[offset + 1])
        state = encoder(observation.z4)
        target = state.detach()
        prediction_loss = F.smooth_l1_loss(pending, target)
        anchor_loss = F.smooth_l1_loss(state, observation.z4.detach())
        losses.append(prediction_loss + LAMBDA_ANCHOR * anchor_loss)
        totals["frames"] += 1
        totals["prediction_loss"] += float(prediction_loss.detach().item())
        totals["anchor_loss"] += float(anchor_loss.detach().item())
        totals["pred_mse_sum"] += float(F.mse_loss(pending.detach(), target).item())
        totals["copy_mse_sum"] += float(F.mse_loss(previous_state, target).item())
        totals["state_rms_sum"] += _rms_sum(state)
        totals["state_std_sum"] += float(state.detach().std().item())
        totals["true_delta_sq_sum"] += float((target - previous_state).detach().square().mean().item())
        totals["state_sq_sum"] += float(target.square().mean().item())
        totals["raw_delta_sq_sum"] += float((observation.z4.detach() - previous_observation_z4).square().mean().item()) if offset > 0 else 0.0
        totals["pred_motion_sq_sum"] += float((pending.detach() - previous_state).square().mean().item())
        error = target - pending.detach()
        previous_state = target
        previous_observation_z4 = observation.z4.detach()
        boundary = len(losses) == TBPTT_STEPS or offset == limit - 1
        if not boundary:
            pending, hidden = predictor.predict_next(previous_state, error, hidden)
            continue
        total_loss = torch.stack(losses).mean()
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite V2-Auxiliary Stage T loss")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        totals["windows"] += 1
        totals["total_loss"] += float(total_loss.detach().item())
        hidden = hidden.detach() if hidden is not None else None
        # Recompute the first prediction after the optimizer update.  This is
        # the causal TBPTT boundary: no state is consumed twice.
        pending, hidden = predictor.predict_next(previous_state, error, hidden)
        losses = []
    for key in ("prediction_loss", "anchor_loss", "total_loss", "state_rms_sum", "state_std_sum"):
        totals[key] /= max(totals["frames"], 1) if key != "total_loss" else max(totals["windows"], 1)
    return totals


def _add_tc(previous_prediction, current_prediction, flow, sums, counts):
    grid, valid = flow_grid(flow, current_prediction.shape[-2], current_prediction.shape[-1])
    warped = F.grid_sample(
        previous_prediction.float().unsqueeze(0), grid, mode="nearest", padding_mode="zeros", align_corners=True
    ).squeeze(0).squeeze(0).to(torch.int64)
    keep = valid.squeeze(0)
    a, b = warped[keep].cpu(), current_prediction.squeeze(0)[keep].cpu()
    pair_confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    if a.numel():
        pair_confusion += torch.bincount(NUM_CLASSES * a + b, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
        score = float(torch.nanmean(compute_iou(pair_confusion)).item())
        if math.isfinite(score):
            sums += score; counts += 1
    return sums, counts


@torch.inference_mode()
def evaluate(model, encoder, predictor, groups, raft):
    names = ("host", "current_t", "predicted_t")
    confusion = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in names}
    vc = {name: VideoConsistency() for name in names}
    mtc_sum = {name: 0.0 for name in names}; mtc_count = {name: 0 for name in names}
    per_sequence = {}
    predictor.eval(); encoder.eval()
    for sequence in FULL9:
        samples = groups[sequence]
        seq_conf = {name: torch.zeros_like(confusion[name]) for name in names}
        seq_vc = {name: VideoConsistency() for name in names}
        seq_mtc_sum = {name: 0.0 for name in names}; seq_mtc_count = {name: 0 for name in names}
        previous_image = None; previous_predictions = {}
        previous_state = None; pending = None; hidden = None
        for index, sample in enumerate(samples):
            image, observation, raw, output_size = encode_clean(model, sample)
            state = encoder(observation.z4)
            if index == 0:
                pending, hidden = predictor.predict_next(state, torch.zeros_like(state), None)
            mask = semantic_mask_from_panoptic_png(sample["mask_path"])
            host_logits = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))
            host_prediction = host_logits.argmax(1)
            current_logits = probe_logits(model, raw, observation, state, output_size)
            predicted_state = state if index == 0 else pending
            predicted_logits = probe_logits(model, raw, observation, predicted_state, output_size)
            predictions = {"host": host_prediction, "current_t": current_logits.argmax(1), "predicted_t": predicted_logits.argmax(1)}
            for name in names:
                update_confusion_matrix(confusion[name], predictions[name].squeeze(0).cpu(), mask)
                update_confusion_matrix(seq_conf[name], predictions[name].squeeze(0).cpu(), mask)
                vc[name].update(mask, predictions[name]); seq_vc[name].update(mask, predictions[name])
            if previous_image is not None:
                flow = raft.current_to_previous(image, previous_image)
                for name in names:
                    mtc_sum[name], mtc_count[name] = _add_tc(previous_predictions[name], predictions[name], flow, mtc_sum[name], mtc_count[name])
                    seq_mtc_sum[name], seq_mtc_count[name] = _add_tc(previous_predictions[name], predictions[name], flow, seq_mtc_sum[name], seq_mtc_count[name])
            previous_image = image
            previous_predictions = predictions
            if index > 0:
                error = state - pending
                previous_state = state
                pending, hidden = predictor.predict_next(previous_state, error, hidden)
            else:
                previous_state = state
        per_sequence[sequence] = {}
        for name in names:
            values = seq_vc[name].values(); iou = compute_iou(seq_conf[name])
            per_sequence[sequence][name] = {
                "mIoU": float(torch.nanmean(iou).item()), "mVC8": values[8], "mVC16": values[16],
                "mTC": seq_mtc_sum[name] / seq_mtc_count[name] if seq_mtc_count[name] else float("nan"),
            }
        per_sequence[sequence]["valid_frame_pairs"] = seq_mtc_count["host"]
    metrics = {}
    for name in names:
        values = vc[name].values(); iou = compute_iou(confusion[name])
        metrics[name] = {"mIoU": float(torch.nanmean(iou).item()), "mVC8": values[8], "mVC16": values[16],
                         "mTC": mtc_sum[name] / mtc_count[name] if mtc_count[name] else float("nan"),
                         "valid_frame_pairs": mtc_count[name]}
    metrics["per_sequence"] = per_sequence
    return metrics


def trainable_parameter_report(encoder, predictor):
    return {
        "temporal_encoder": sum(parameter.numel() for parameter in encoder.parameters()),
        "temporal_predictor": sum(parameter.numel() for parameter in predictor.parameters()),
        "total": sum(parameter.numel() for parameter in encoder.parameters()) + sum(parameter.numel() for parameter in predictor.parameters()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--dynamics-checkpoint", default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--max-train-steps", type=int, default=0)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model, encoder, predictor, source = load_training_components(args)
    assert_freeze_contract(model, encoder, predictor)
    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train); val_groups = sequence_groups(val)
    zero_step = zero_step_check(model, encoder, predictor, next(iter(val_groups.values()))[0])
    print(json.dumps({"zero_step": zero_step, "trainable_parameters": trainable_parameter_report(encoder, predictor)}, sort_keys=True), flush=True)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    raft = FrozenRAFT()
    history = []; best = None; stale = 0
    for epoch in range(1, args.epochs + 1):
        encoder.train(); predictor.train()
        totals = {"frames": 0, "windows": 0, "prediction_loss": 0.0, "anchor_loss": 0.0, "total_loss": 0.0,
                  "pred_mse_sum": 0.0, "copy_mse_sum": 0.0, "state_rms_sum": 0.0, "state_std_sum": 0.0,
                  "true_delta_sq_sum": 0.0, "state_sq_sum": 0.0, "raw_delta_sq_sum": 0.0, "pred_motion_sq_sum": 0.0}
        for samples in train_groups.values():
            row = train_sequence(model, encoder, predictor, samples, optimizer, args.max_train_steps)
            for key in totals:
                totals[key] += row[key]
        for key in ("prediction_loss", "anchor_loss", "state_rms_sum", "state_std_sum"):
            totals[key] /= max(totals["frames"], 1)
        totals["total_loss"] /= max(totals["windows"], 1)
        metrics = evaluate(model, encoder, predictor, val_groups, raft)
        pred = metrics["predicted_t"]
        record = {
            "epoch": epoch, "stage": "T", "train": totals, "metrics": metrics,
            "Rpred": totals["pred_mse_sum"] / max(totals["copy_mse_sum"], 1e-12),
            "pred_motion_ratio": math.sqrt(totals["pred_motion_sq_sum"] / max(totals["frames"], 1)) / math.sqrt(totals["true_delta_sq_sum"] / max(totals["frames"], 1) + 1e-12),
            "true_delta_rms": math.sqrt(totals["true_delta_sq_sum"] / max(totals["frames"], 1)),
            "temporal_to_state_ratio": math.sqrt(totals["true_delta_sq_sum"] / max(totals["frames"], 1)) / math.sqrt(totals["state_sq_sum"] / max(totals["frames"], 1) + 1e-12),
            "dynamic_ratio": math.sqrt(totals["true_delta_sq_sum"] / max(totals["frames"], 1)) / math.sqrt(totals["raw_delta_sq_sum"] / max(totals["frames"] - len(train_groups), 1) + 1e-12),
            "reference_Rpred_Z4": 0.9451,
            "final_output_is_host": True,
        }
        history.append(record); print(json.dumps(record, sort_keys=True), flush=True)
        output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
        payload = {"experiment": "v2_auxiliary_stage_t", "epoch": epoch, "encoder_state_dict": encoder.state_dict(), "predictor_state_dict": predictor.state_dict(), "metrics": record}
        torch.save(payload, output / f"epoch_{epoch:03d}.pt")
        ratio = record["Rpred"]; dynamic_ok = record["dynamic_ratio"] >= 0.5
        if dynamic_ok and (best is None or ratio < best["Rpred"]):
            best = {"epoch": epoch, "Rpred": ratio, "dynamic_ratio": record["dynamic_ratio"], "metrics": metrics}; torch.save(payload, output / "best.pt"); stale = 0
        else:
            stale += 1
        if best is not None and epoch - best["epoch"] >= args.patience:
            break
    result = {
        "experiment": "Predify V2-Auxiliary Stage T", "source_fast_b_checkpoint": args.fast_b_checkpoint,
        "tbptt_steps": TBPTT_STEPS, "lr": LR, "weight_decay": WEIGHT_DECAY, "lambda_anchor": LAMBDA_ANCHOR,
        "trainable_parameters": trainable_parameter_report(encoder, predictor),
        "frozen_modules": ["Host", "Adapter", "Writeback", "Segmentation Decoder"],
        "zero_step": zero_step, "history": history, "best": best,
        "gate": {"Rpred_lt_1": bool(best and best["Rpred"] < 1.0), "Rpred_lt_raw_z4": bool(best and best["Rpred"] < 0.9451),
                 "dynamic_ratio_ge_0.5": bool(best and best["dynamic_ratio"] >= 0.5), "sequences_required": 6},
    }
    result_dir = Path(args.result_output); result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (result_dir / "README.md").write_text("# V2-Auxiliary Stage T\nStage T trains only the 128-channel temporal encoder and causal predictor; final output remains the frozen Host.\n")


if __name__ == "__main__":
    main()
