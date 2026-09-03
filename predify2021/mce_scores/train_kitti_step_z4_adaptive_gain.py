"""Train and evaluate the scalar Z4 Prediction--Observation Adaptive Gain.

Only :class:`Z4AdaptiveGainHead` is trainable.  The Stage-P predictor, Host,
C4 adapter/writeback and decoder are frozen.  Stage-P always consumes the raw
observation and current prediction error; the learned posterior is used only
for the frozen segmentation path.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import VideoConsistency
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
    sequence_groups,
)
from predify2021.mce_scores.evaluate_kitti_step_raft_error_memory import FrozenRAFT
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
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
    measure_sequence,
    z4_predict_next,
)
from predify2021.mce_scores.evaluate_kitti_step_prediction_error_utility_upper_bound import (
    _pair_mtc,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorRegulatedSemanticRestorationPredictor,
    HostFeature,
    UnifiedFeatures,
)
from predify2021.model_factory.z4_adaptive_gain import Z4AdaptiveGainHead

SEED = 0
TBPTT = 16
MAX_EPOCHS = 15
PATIENCE = 3
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
LAMBDA_TC = 1.0
NUM_CLASSES = 19
IGNORE = 255
MTC_GATE = 0.005  # +0.5 percentage points
SEQUENCES = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
STAGE_P_DEFAULT = "/home/lin/predify/experiments/kitti_step_z4_only_stage_p/best.pt"
OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_z4_adaptive_gain"
RESULT_DEFAULT = "results/kitti_step_z4_adaptive_gain"


def load_frozen_components(args):
    model, _ = load_components(
        STATIC_CHECKPOINT_DEFAULT,
        ADAPTER_CHECKPOINT_DEFAULT,
        args.dynamics_checkpoint,
        WRITEBACK_CHECKPOINT_DEFAULT,
    )
    fast = torch.load(args.fast_b_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.output_adapters[3].load_state_dict(
        fast["c4_output_adapter_state_dict"], strict=True
    )
    model.host_conditioned_writebacks["3"].load_state_dict(
        fast["c4_writeback_state_dict"], strict=True
    )
    stage_p = torch.load(args.stage_p_checkpoint, map_location="cpu", weights_only=False)
    predictor = ErrorRegulatedSemanticRestorationPredictor(
        use_error_temporal_stats=True
    ).cuda()
    predictor.load_state_dict(stage_p["model_state_dict"], strict=True)
    model.requires_grad_(False).eval().cuda()
    predictor.requires_grad_(False).eval()
    return model, predictor, stage_p


def assert_freeze_contract(model, predictor, gain_head):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Host/adapter/writeback/decoder must be frozen")
    if any(parameter.requires_grad for parameter in predictor.parameters()):
        raise RuntimeError("Stage-P predictor and legacy semantic modules must be frozen")
    if not all(parameter.requires_grad for parameter in gain_head.parameters()):
        raise RuntimeError("Adaptive Gain Head must be trainable")
    if sum(parameter.numel() for parameter in gain_head.parameters()) != 97:
        raise RuntimeError("Unexpected Adaptive Gain Head parameter count")


def encode(model, sample):
    image = load_image(sample)
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        observation = model.encode_backbone_features(raw)
    return image, observation, raw, tuple(image.shape[-2:])


def gain_logits(model, raw, observation, post_z4, output_size):
    delta = UnifiedFeatures(
        torch.zeros_like(observation.z1),
        torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3),
        post_z4 - observation.z4,
    )
    host_feature = residual_writeback_host_feature(model, raw, delta, output_size)
    return model.decode_from_host_feature(host_feature)


def temporal_loss(current_logits, previous_logits, backward_flow, previous_mask, current_mask):
    height, width = current_logits.shape[-2:]
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
    current_prob = current_logits.softmax(1)
    previous_prob = previous_logits.detach().softmax(1)
    warped_previous = F.grid_sample(
        previous_prob, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    warped_mask = F.grid_sample(
        previous_mask.float()[None, None], grid, mode="nearest",
        padding_mode="zeros", align_corners=True,
    )[0, 0].long()
    confident = F.grid_sample(
        (previous_prob.amax(1, keepdim=True) > 0.70).float(), grid,
        mode="nearest", padding_mode="zeros", align_corners=True,
    )[0, 0] > 0.5
    keep = valid[0] & (warped_mask != IGNORE) & (current_mask != IGNORE) & confident
    if not keep.any():
        return current_logits.sum() * 0.0
    value = F.kl_div(
        current_prob.clamp_min(1e-8).log(), warped_previous.clamp_min(1e-8),
        reduction="none",
    ).sum(1)[0]
    return value[keep].mean()


def train_sequence(model, predictor, gain_head, raft, samples, optimizer, lambda_tc):
    totals = {"windows": 0, "frames": 0, "Lseg": 0.0, "LTC": 0.0, "total": 0.0}
    if len(samples) < 2:
        return totals
    previous_image, observation, raw, output_size = encode(model, samples[0])
    previous_mask = semantic_mask_from_panoptic_png(samples[0]["mask_path"]).cuda()
    with torch.no_grad():
        host = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))
        previous_student_logits = host.detach()
        pending_z4, hidden = z4_predict_next(
            predictor, observation.z4, torch.zeros_like(observation.z4), None
        )
    previous_magnitude = None
    seg_losses, tc_losses = [], []
    for frame_index, sample in enumerate(samples[1:], 1):
        image, observation, raw, output_size = encode(model, sample)
        mask = semantic_mask_from_panoptic_png(sample["mask_path"]).cuda()
        with torch.no_grad():
            error = observation.z4 - pending_z4
            host_logits = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))
            backward_flow = raft.backward_flow(image, previous_image)
        gain, stats, _ = gain_head(observation.z4, pending_z4, previous_magnitude)
        post_z4 = pending_z4 + gain * error
        student_logits = gain_logits(model, raw, observation, post_z4, output_size)
        seg_losses.append(F.cross_entropy(student_logits, mask.unsqueeze(0), ignore_index=IGNORE))
        tc_losses.append(temporal_loss(student_logits, previous_student_logits, backward_flow, previous_mask, mask))
        previous_magnitude = stats[:, 0].detach()
        with torch.no_grad():
            pending_z4, hidden = z4_predict_next(predictor, observation.z4, error, hidden)
        previous_image, previous_mask = image, mask
        previous_student_logits = student_logits.detach()
        totals["frames"] += 1
        window_end = frame_index % TBPTT == 0 or frame_index == len(samples) - 1
        if not window_end:
            continue
        lseg = torch.stack(seg_losses).mean()
        ltc = torch.stack(tc_losses).mean()
        total = lseg + lambda_tc * ltc
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite Adaptive Gain loss")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        totals["windows"] += 1
        totals["Lseg"] += float(lseg.detach().item())
        totals["LTC"] += float(ltc.detach().item())
        totals["total"] += float(total.detach().item())
        seg_losses, tc_losses = [], []
    for key in ("Lseg", "LTC", "total"):
        totals[key] /= max(totals["windows"], 1)
    return totals


def train_epoch(model, predictor, gain_head, raft, groups, optimizer, lambda_tc):
    aggregate = {"sequences": 0, "windows": 0, "frames": 0, "Lseg": 0.0, "LTC": 0.0, "total": 0.0}
    gain_head.train()
    for samples in groups.values():
        row = train_sequence(model, predictor, gain_head, raft, samples, optimizer, lambda_tc)
        aggregate["sequences"] += 1
        aggregate["windows"] += row["windows"]
        aggregate["frames"] += row["frames"]
        for key in ("Lseg", "LTC", "total"):
            aggregate[key] += row[key] * row["windows"]
    for key in ("Lseg", "LTC", "total"):
        aggregate[key] /= max(aggregate["windows"], 1)
    return aggregate


def _pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x = x - x.mean(); y = y - y.mean()
    den = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float((x.mul(y).sum() / den).item()) if den > 0 else float("nan")


def _summarize_gain(records):
    if not records:
        return {"count": 0}
    keys = ("K", "error_magnitude", "relative_error", "error_change", "prediction_observation_discrepancy")
    result = {"count": len(records)}
    for key in keys:
        values = torch.tensor([row[key] for row in records], dtype=torch.float64)
        result[key + "_mean"] = float(values.mean().item())
        result[key + "_std"] = float(values.std(unbiased=False).item())
        result[key + "_min"] = float(values.min().item())
        result[key + "_max"] = float(values.max().item())
        result[key + "_p10"] = float(torch.quantile(values, 0.10).item())
        result[key + "_p50"] = float(torch.quantile(values, 0.50).item())
        result[key + "_p90"] = float(torch.quantile(values, 0.90).item())
    result["corr_K_error_magnitude"] = _pearson(
        [row["K"] for row in records], [row["error_magnitude"] for row in records]
    )
    result["corr_K_relative_error"] = _pearson(
        [row["K"] for row in records], [row["relative_error"] for row in records]
    )
    result["corr_K_error_change"] = _pearson(
        [row["K"] for row in records], [row["error_change"] for row in records]
    )
    return result


@torch.inference_mode()
def evaluate(model, predictor, gain_head, groups, raft, fixed_gain=None):
    names = ("host", "ours")
    confusion = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in names}
    mvc_sums = {name: {8: 0.0, 16: 0.0} for name in names}
    mvc_counts = {name: {8: 0, 16: 0} for name in names}
    mtc_sum = {name: 0.0 for name in names}; mtc_count = {name: 0 for name in names}
    per_sequence = {}; gain_records = {"global": [], "per_sequence": {}}
    gain_head.eval()
    for sequence in SEQUENCES:
        samples = groups[sequence]
        seq_conf = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in names}
        seq_vc = {name: VideoConsistency() for name in names}
        seq_mtc_sum = {name: 0.0 for name in names}; seq_mtc_count = {name: 0 for name in names}
        seq_records = []; previous_image = None; previous_predictions = None; previous_magnitude = None
        image, observation, raw, output_size = encode(model, samples[0])
        host_logits = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))
        mask = semantic_mask_from_panoptic_png(samples[0]["mask_path"])
        predictions = {name: host_logits.argmax(1) for name in names}
        _accumulate(predictions, mask, confusion, seq_conf, seq_vc)
        previous_image = image; previous_predictions = {name: pred.detach() for name, pred in predictions.items()}
        pending_z4, hidden = z4_predict_next(predictor, observation.z4, torch.zeros_like(observation.z4), None)
        for sample in samples[1:]:
            image, observation, raw, output_size = encode(model, sample)
            mask = semantic_mask_from_panoptic_png(sample["mask_path"])
            host_logits = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))
            error, stats = gain_head.statistics(observation.z4, pending_z4, previous_magnitude)
            gain = torch.full_like(error[:, :1, :1, :1], 1.0 if fixed_gain is None else float(fixed_gain))
            if fixed_gain is None:
                gain = gain_head.network(stats).sigmoid().reshape(-1, 1, 1, 1)
            post_z4 = pending_z4 + gain * error
            ours_logits = gain_logits(model, raw, observation, post_z4, output_size)
            predictions = {"host": host_logits.argmax(1), "ours": ours_logits.argmax(1)}
            _accumulate(predictions, mask, confusion, seq_conf, seq_vc)
            gain_row = {
                "K": float(gain.mean().item()),
                "error_magnitude": float(stats[0, 0].item()),
                "relative_error": float(stats[0, 1].item()),
                "error_change": float(stats[0, 2].item()),
                "prediction_observation_discrepancy": float(stats[0, 3].item()),
            }
            seq_records.append(gain_row); gain_records["global"].append(gain_row)
            backward_flow = raft.backward_flow(image, previous_image)
            for name in names:
                score = _pair_mtc(previous_predictions[name], predictions[name], backward_flow)
                if math.isfinite(score):
                    mtc_sum[name] += score; mtc_count[name] += 1
                    seq_mtc_sum[name] += score; seq_mtc_count[name] += 1
            previous_image = image; previous_predictions = {name: pred.detach() for name, pred in predictions.items()}
            previous_magnitude = stats[:, 0].detach()
            pending_z4, hidden = z4_predict_next(predictor, observation.z4, error, hidden)
        gain_records["per_sequence"][sequence] = _summarize_gain(seq_records)
        for name in names:
            stats = seq_vc[name].stats()
            for length in (8, 16):
                mvc_sums[name][length] += stats[length]["sum"]
                mvc_counts[name][length] += stats[length]["count"]
        per_sequence[sequence] = {
            name: {
                "mIoU": float(torch.nanmean(compute_iou(seq_conf[name])).item()),
                "mVC8": seq_vc[name].values()[8], "mVC16": seq_vc[name].values()[16],
                "mTC": seq_mtc_sum[name] / max(seq_mtc_count[name], 1),
                "valid_frame_pairs": seq_mtc_count[name],
            } for name in names
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
    metrics["delta"] = {key: metrics["ours"][key] - metrics["host"][key] for key in ("mIoU", "mVC8", "mVC16", "mTC")}
    metrics["per_sequence"] = per_sequence
    gain_records["global"] = _summarize_gain(gain_records["global"])
    metrics["gain_statistics"] = gain_records
    return metrics


def _accumulate(predictions, mask, confusion, seq_conf, seq_vc):
    for name, prediction in predictions.items():
        value = prediction[0].cpu()
        update_confusion_matrix(confusion[name], value, mask)
        update_confusion_matrix(seq_conf[name], value, mask)
        seq_vc[name].update(mask, value)


@torch.no_grad()
def stage_p_metrics(model, predictor, groups):
    rows = {sequence: measure_sequence(model, predictor, samples) for sequence, samples in groups.items()}
    frames = sum(row["frames"] for row in rows.values())
    pred = sum(row["pred_sum"] for row in rows.values()) / max(frames, 1)
    copy = sum(row["copy_sum"] for row in rows.values()) / max(frames, 1)
    return {
        "mean_pred_mse": pred, "mean_copy_mse": copy, "ratio": pred / max(copy, 1e-12),
        "sequences_better_than_persistence": sum(row["ratio"] < 1.0 for row in rows.values()),
        "sequence_count": len(rows), "effective_frames": frames, "per_sequence": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--stage-p-checkpoint", default=STAGE_P_DEFAULT)
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_DEFAULT)
    parser.add_argument("--dynamics-checkpoint", default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--lambda-tc", type=float, default=LAMBDA_TC)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model, predictor, stage_p_payload = load_frozen_components(args)
    gain_head = Z4AdaptiveGainHead().cuda()
    assert_freeze_contract(model, predictor, gain_head)
    train_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    train_groups = sequence_groups(train_dataset)
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    all_val_groups = sequence_groups(val_dataset)
    val_groups = {sequence: all_val_groups[sequence] for sequence in SEQUENCES}
    raft = FrozenRAFT()

    # Three mandatory pre-training checks.
    host_check = evaluate(model, predictor, gain_head, val_groups, raft, fixed_gain=1.0)
    half_check = evaluate(model, predictor, gain_head, val_groups, raft, fixed_gain=0.5)
    stage_check = stage_p_metrics(model, predictor, val_groups)
    expected_host = {"mIoU": 0.6541093425, "mTC": 0.7078140916, "mVC8": 0.9379491585, "mVC16": 0.9289750719}
    expected_half = {"mIoU": 0.650761, "mTC": 0.719989, "mVC8": 0.942547, "mVC16": 0.934413}
    for key, value in expected_host.items():
        if abs(host_check["ours"][key] - value) > 2e-4:
            raise RuntimeError(f"K=1 Host equivalence failed for {key}: {host_check['ours'][key]}")
    for key, value in expected_half.items():
        if abs(half_check["ours"][key] - value) > 5e-4:
            raise RuntimeError(f"K=0.5 utility equivalence failed for {key}: {half_check['ours'][key]}")
    if abs(stage_check["ratio"] - 0.9451) > 0.01 or stage_check["sequences_better_than_persistence"] < 6:
        raise RuntimeError(f"Stage-P unchanged check failed: {stage_check}")
    checks = {"K1_host_equivalence": host_check, "K05_utility_equivalence": half_check, "stage_p_unchanged": stage_check}
    print(json.dumps({"pretraining_checks": checks}, sort_keys=True), flush=True)

    optimizer = torch.optim.AdamW(gain_head.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    history = []; best = None; bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train = train_epoch(model, predictor, gain_head, raft, train_groups, optimizer, args.lambda_tc)
        metrics = evaluate(model, predictor, gain_head, val_groups, raft)
        record = {"epoch": epoch, "train": train, "val": metrics, "lambda_tc": args.lambda_tc}
        history.append(record)
        torch.save({"experiment": "kitti_step_z4_adaptive_gain", "epoch": epoch, "adaptive_gain_state_dict": gain_head.state_dict(), "stage_p_checkpoint_path": args.stage_p_checkpoint, "metrics": metrics}, output / f"epoch_{epoch:03d}.pt")
        delta_mtc = metrics["delta"]["mTC"]
        eligible = delta_mtc > MTC_GATE
        if eligible and (best is None or (metrics["ours"]["mIoU"], metrics["ours"]["mTC"]) > (best["mIoU"], best["mTC"])):
            best = {"epoch": epoch, "mIoU": metrics["ours"]["mIoU"], "mTC": metrics["ours"]["mTC"], "delta": metrics["delta"]}
            torch.save({"experiment": "kitti_step_z4_adaptive_gain", "epoch": epoch, "adaptive_gain_state_dict": gain_head.state_dict(), "stage_p_checkpoint_path": args.stage_p_checkpoint, "metrics": metrics, "gain_statistics": metrics["gain_statistics"]}, output / "best.pt")
            bad_epochs = 0
        elif eligible:
            bad_epochs += 1
        print(json.dumps({"epoch": epoch, "train": train, "val": {k: metrics[k] for k in ("ours", "delta")}, "gain": metrics["gain_statistics"]["global"], "eligible": eligible}, sort_keys=True), flush=True)
        if best is not None and bad_epochs >= args.patience:
            break

    final_metrics = evaluate(model, predictor, gain_head, val_groups, raft)
    result_dir = Path(args.result_output); result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "training_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    (result_dir / "full9_metrics.json").write_text(json.dumps(final_metrics, indent=2, sort_keys=True) + "\n")
    (result_dir / "gain_statistics.json").write_text(json.dumps(final_metrics["gain_statistics"], indent=2, sort_keys=True) + "\n")
    summary = {
        "experiment": "Predify Z4 Adaptive Gain",
        "stage_p_checkpoint": args.stage_p_checkpoint,
        "stage_p_checkpoint_epoch": stage_p_payload.get("epoch"),
        "trainable_modules": ["Z4AdaptiveGainHead"],
        "trainable_parameter_count": sum(parameter.numel() for parameter in gain_head.parameters()),
        "lambda_tc": args.lambda_tc, "learning_rate": args.lr, "tbptt": TBPTT,
        "max_epochs": args.epochs, "patience": args.patience, "mTC_gate": MTC_GATE,
        "pretraining_checks": checks, "history": history, "best": best,
        "final_full9": final_metrics,
        "status": "GO" if best is not None and best["delta"]["mIoU"] >= 0.0 else "NO-GO",
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (result_dir / "README.md").write_text(
        "# Z4 Adaptive Gain\n\n"
        "Only a frame-wise scalar `Z4AdaptiveGainHead` is trained. Host, C4 "
        "adapter/writeback, decoder, and Stage-P predictor remain frozen. "
        "Stage-P consumes raw `Z4` and `e=Z4-Z4_prediction`; the posterior is "
        "used only for the frozen segmentation path.\n"
    )
    print(json.dumps({"result": str(result_dir / "summary.json"), "best": best}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
