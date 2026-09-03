"""Train and validate the minimal task-aware predictive semantic state (V2.1).

The Host, C4 adapter/writeback and decoder are frozen.  Only the new
predictive semantic encoder, residual ConvGRU predictor and Z4 update head are
optimized.  The causal order is strictly ``Rhat_t -> R_t -> e_t -> Rhat_t+1``.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

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
from predify2021.mce_scores.train_kitti_step_z4_only import FAST_B_DEFAULT
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    HostFeature,
    UnifiedFeatures,
    PredictiveSemanticV2,
)

SEED = 0
NUM_CLASSES = 19
IGNORE = 255
TBPTT = 16
MAX_EPOCHS = 3
PATIENCE = 3
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
LAMBDA_PRED = 0.1
LAMBDA_TC = 0.1
HOST_CONFIDENCE = 0.70
DEV3 = ("0002", "0010", "0018")
FULL9 = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_predictive_semantic_v21"
RESULT_DEFAULT = "results/kitti_step_predictive_semantic_v21"


def encode_host(model, sample):
    image = load_image(sample)
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        observation = model.encode_backbone_features(raw)
    return image, observation, raw, tuple(image.shape[-2:])


def host_logits(model, raw, output_size):
    with torch.no_grad():
        return model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))


def post_logits(model, raw, observation, delta_z4, output_size):
    delta = UnifiedFeatures(
        torch.zeros_like(observation.z1),
        torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3),
        delta_z4,
    )
    host_feature = residual_writeback_host_feature(model, raw, delta, output_size)
    if torch.is_grad_enabled():
        return checkpoint(
            lambda high: model.decode_from_host_feature(
                HostFeature(high, host_feature.low_level, host_feature.output_size)
            ),
            host_feature.tensor,
            use_reentrant=False,
        )
    return model.decode_from_host_feature(host_feature)


def flow_grid(backward_flow, height, width):
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
        (2 * source_x / max(width - 1, 1) - 1, 2 * source_y / max(height - 1, 1) - 1),
        dim=-1,
    )
    return grid, valid


def temporal_loss(current_logits, previous_logits, backward_flow, previous_mask, current_mask):
    height, width = current_logits.shape[-2:]
    grid, valid = flow_grid(backward_flow, height, width)
    current_prob = current_logits.softmax(1)
    previous_prob = previous_logits.detach().softmax(1)
    warped_previous = F.grid_sample(
        previous_prob, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    warped_previous_mask = F.grid_sample(
        previous_mask.float()[None, None], grid, mode="nearest", padding_mode="zeros", align_corners=True
    )[0, 0].long()
    warped_confident = F.grid_sample(
        (previous_prob.amax(1, keepdim=True) > HOST_CONFIDENCE).float(),
        grid, mode="nearest", padding_mode="zeros", align_corners=True,
    )[0, 0] > 0.5
    keep = (
        valid[0] & (warped_previous_mask != IGNORE) & (current_mask != IGNORE)
        & warped_confident
    )
    if not keep.any():
        return current_logits.sum() * 0.0
    per_pixel = F.kl_div(
        current_prob.clamp_min(1e-8).log(), warped_previous.clamp_min(1e-8), reduction="none"
    ).sum(1)[0]
    return per_pixel[keep].mean()


def assert_contract(host, plugin):
    if any(p.requires_grad for p in host.parameters()):
        raise RuntimeError("Host must be frozen")
    if not all(p.requires_grad for p in plugin.parameters()):
        raise RuntimeError("All V2 plugin parameters must be trainable")
    expected = {"encoder", "predictor", "update_head"}
    observed = {name.split(".", 1)[0] for name, p in plugin.named_parameters() if p.requires_grad}
    if observed != expected:
        raise RuntimeError(f"Unexpected V2 trainable modules: {sorted(observed)}")


def load_v2(args):
    host, _ = load_components(
        STATIC_CHECKPOINT_DEFAULT,
        ADAPTER_CHECKPOINT_DEFAULT,
        args.dynamics_checkpoint,
        WRITEBACK_CHECKPOINT_DEFAULT,
    )
    source = torch.load(args.fast_b_checkpoint, map_location="cpu", weights_only=False)
    host.multi_layer_adapter.output_adapters[3].load_state_dict(
        source["c4_output_adapter_state_dict"], strict=True
    )
    host.host_conditioned_writebacks["3"].load_state_dict(
        source["c4_writeback_state_dict"], strict=True
    )
    host.requires_grad_(False).eval().cuda()
    plugin = PredictiveSemanticV2().cuda()
    plugin.train()
    assert_contract(host, plugin)
    return host, plugin, source


def zero_update_sanity(host, plugin, samples):
    """Check exact identity and causal state dimensions before training."""
    sample = samples[0]
    _, observation, raw, output_size = encode_host(host, sample)
    state = plugin.encode(observation.z4)
    predicted, hidden = plugin.predict_next(state.detach(), torch.zeros_like(state), None)
    post, error, delta = plugin.update(observation.z4, state, predicted)
    with torch.no_grad():
        host_out = host_logits(host, raw, output_size)
        zero_out = post_logits(host, raw, observation, delta, output_size)
    if not torch.isfinite(delta).all() or not torch.isfinite(zero_out).all():
        raise FloatingPointError("V2 zero-update sanity produced NaN/Inf")
    max_delta = float(delta.abs().max().item())
    max_logit_diff = float((zero_out - host_out).abs().max().item())
    if max_delta != 0.0 or max_logit_diff > 1e-6:
        raise RuntimeError(
            f"Zero-update identity failed: max_delta={max_delta}, max_logit_diff={max_logit_diff}"
        )
    return {"max_abs_delta_z4": max_delta, "max_abs_logit_difference": max_logit_diff}


def train_sequence(host, plugin, raft, samples, optimizer):
    totals = {"windows": 0, "frames": 0, "Lseg": 0.0, "Lpred": 0.0, "LTC": 0.0, "total": 0.0,
              "delta_abs": 0.0, "delta_rms": 0.0, "delta_rel": 0.0,
              "state_rms": 0.0, "state_std": 0.0, "state_temporal_mse": 0.0,
              "copy_mse": 0.0}
    if len(samples) < 2:
        return totals
    previous_image, observation, raw, output_size = encode_host(host, samples[0])
    previous_mask = semantic_mask_from_panoptic_png(samples[0]["mask_path"]).cuda()
    previous_logits = host_logits(host, raw, output_size).detach()
    state = plugin.encode(observation.z4)
    pending, hidden = plugin.predict_next(state.detach(), torch.zeros_like(state), None)
    previous_state = state.detach()
    losses = {key: [] for key in ("seg", "pred", "tc")}
    stats = {key: [] for key in ("delta_abs", "delta_rms", "delta_rel", "state_rms", "state_std", "state_temporal_mse", "copy_mse")}

    for frame_index, sample in enumerate(samples[1:], 1):
        image, observation, raw, output_size = encode_host(host, sample)
        mask = semantic_mask_from_panoptic_png(sample["mask_path"]).cuda()
        state = plugin.encode(observation.z4)
        post, error, delta = plugin.update(observation.z4, state, pending)
        logits = post_logits(host, raw, observation, delta, output_size)
        hlogits = host_logits(host, raw, output_size)
        losses["seg"].append(F.cross_entropy(logits, mask.unsqueeze(0), ignore_index=IGNORE))
        losses["pred"].append(F.smooth_l1_loss(pending, state.detach()))
        losses["tc"].append(temporal_loss(logits, previous_logits, raft.backward_flow(image, previous_image), previous_mask, mask))
        stats["delta_abs"].append(delta.detach().abs().mean())
        stats["delta_rms"].append(delta.detach().square().mean().sqrt())
        stats["delta_rel"].append(delta.detach().square().mean().sqrt() / (observation.z4.detach().square().mean().sqrt() + 1e-8))
        stats["state_rms"].append(state.detach().square().mean().sqrt())
        stats["state_std"].append(state.detach().std())
        stats["state_temporal_mse"].append(F.mse_loss(state.detach(), previous_state.detach()))
        stats["copy_mse"].append(F.mse_loss(previous_state.detach(), state.detach()))
        # Preserve the predictor input state before consuming this frame.  At
        # a TBPTT boundary it is reused after the optimizer step; reusing
        # next_hidden would consume (R_t,e_t) twice.
        hidden_input = hidden.detach() if hidden is not None else None
        next_pending, next_hidden = plugin.predict_next(state.detach(), error.detach(), hidden)
        previous_image, previous_mask, previous_logits = image, mask, logits.detach()
        previous_state = state.detach()
        hidden = next_hidden
        pending = next_pending
        totals["frames"] += 1
        window_end = frame_index % TBPTT == 0 or frame_index == len(samples) - 1
        if not window_end:
            continue
        lseg, lpred, ltc = (torch.stack(losses[k]).mean() for k in ("seg", "pred", "tc"))
        total = lseg + LAMBDA_PRED * lpred + LAMBDA_TC * ltc
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite V2 objective")
        # Rebuild the first pending prediction after the update with current weights.
        state_boundary, error_boundary, hidden_boundary = state.detach(), error.detach(), hidden_input
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        pending, hidden = plugin.predict_next(state_boundary, error_boundary, hidden_boundary)
        totals["windows"] += 1
        totals["Lseg"] += float(lseg.detach())
        totals["Lpred"] += float(lpred.detach())
        totals["LTC"] += float(ltc.detach())
        totals["total"] += float(total.detach())
        for key in stats:
            totals[key] += float(torch.stack(stats[key]).mean())
        losses = {key: [] for key in losses}
        stats = {key: [] for key in stats}

    for key in ("Lseg", "Lpred", "LTC", "total", "delta_abs", "delta_rms", "delta_rel", "state_rms", "state_std", "state_temporal_mse", "copy_mse"):
        totals[key] /= max(totals["windows"], 1)
    return totals


def train_epoch(host, plugin, raft, groups, optimizer):
    aggregate = {"sequences": 0, "windows": 0, "frames": 0, "Lseg": 0.0, "Lpred": 0.0, "LTC": 0.0, "total": 0.0,
                 "delta_abs": 0.0, "delta_rms": 0.0, "delta_rel": 0.0,
                 "state_rms": 0.0, "state_std": 0.0, "state_temporal_mse": 0.0,
                 "copy_mse": 0.0}
    plugin.train()
    for samples in groups.values():
        row = train_sequence(host, plugin, raft, samples, optimizer)
        aggregate["sequences"] += 1; aggregate["windows"] += row["windows"]; aggregate["frames"] += row["frames"]
        for key in ("Lseg", "Lpred", "LTC", "total", "delta_abs", "delta_rms", "delta_rel", "state_rms", "state_std", "state_temporal_mse", "copy_mse"):
            aggregate[key] += row[key] * row["windows"]
    for key in ("Lseg", "Lpred", "LTC", "total", "delta_abs", "delta_rms", "delta_rel", "state_rms", "state_std", "state_temporal_mse", "copy_mse"):
        aggregate[key] /= max(aggregate["windows"], 1)
    return aggregate


def pair_mtc(previous_prediction, current_prediction, backward_flow):
    height, width = current_prediction.shape[-2:]
    grid, valid = flow_grid(backward_flow, height, width)
    warped = F.grid_sample(previous_prediction.float().unsqueeze(1), grid, mode="nearest", padding_mode="zeros", align_corners=True)[0, 0].long()
    keep = valid[0]; a, b = warped[keep].cpu(), current_prediction[0][keep].cpu()
    if not a.numel():
        return float("nan")
    confusion = torch.bincount(NUM_CLASSES * a + b, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    return float(torch.nanmean(compute_iou(confusion)).item())


@torch.inference_mode()
def evaluate(host, plugin, groups, raft):
    plugin.eval()
    names = ("host", "ours")
    confusion = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in names}
    mvc_sums = {name: {8: 0.0, 16: 0.0} for name in names}; mvc_counts = {name: {8: 0, 16: 0} for name in names}
    mtc_sum = {name: 0.0 for name in names}; mtc_count = {name: 0 for name in names}
    pred_sum = copy_sum = frame_count = 0.0
    state_rms_sum = state_std_sum = state_temporal_sum = 0.0
    state_frame_count = 0
    update_sum = update_rms = update_rel = 0.0; update_frames = 0
    per_sequence = {}
    for sequence in FULL9:
        samples = groups[sequence]
        if len(samples) < 2: continue
        seq_conf = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in names}
        seq_vc = {name: VideoConsistency() for name in names}; seq_mtc_sum = {name: 0.0 for name in names}; seq_mtc_count = {name: 0 for name in names}
        image, observation, raw, output_size = encode_host(host, samples[0]); mask = semantic_mask_from_panoptic_png(samples[0]["mask_path"])
        hlogits = host_logits(host, raw, output_size); predictions = {"host": hlogits.argmax(1), "ours": hlogits.argmax(1)}
        state = plugin.encode(observation.z4); pending, hidden = plugin.predict_next(state.detach(), torch.zeros_like(state), None)
        state_rms_sum += state.square().mean().sqrt().item(); state_std_sum += state.std().item(); state_frame_count += 1
        seq_state_rms_sum = state.square().mean().sqrt().item(); seq_state_std_sum = state.std().item(); seq_state_frame_count = 1
        previous_observation = state; previous_image = image; previous_predictions = {k: v.detach() for k, v in predictions.items()}
        for name, prediction in predictions.items():
            update_confusion_matrix(confusion[name], prediction[0].cpu(), mask); update_confusion_matrix(seq_conf[name], prediction[0].cpu(), mask); seq_vc[name].update(mask, prediction[0].cpu())
        seq_pred_sum = seq_copy_sum = 0.0; seq_frames = 0
        seq_state_temporal_sum = 0.0
        seq_update_sum = seq_update_rms = seq_update_rel = 0.0
        for sample in samples[1:]:
            image, observation, raw, output_size = encode_host(host, sample); mask = semantic_mask_from_panoptic_png(sample["mask_path"])
            hlogits = host_logits(host, raw, output_size); host_prediction = hlogits.argmax(1)
            state = plugin.encode(observation.z4); post, error, delta = plugin.update(observation.z4, state, pending)
            ours_prediction = post_logits(host, raw, observation, delta, output_size).argmax(1)
            predictions = {"host": host_prediction, "ours": ours_prediction}
            for name, prediction in predictions.items():
                update_confusion_matrix(confusion[name], prediction[0].cpu(), mask); update_confusion_matrix(seq_conf[name], prediction[0].cpu(), mask); seq_vc[name].update(mask, prediction[0].cpu())
            score_flow = raft.backward_flow(image, previous_image)
            for name, prediction in predictions.items():
                score = pair_mtc(previous_predictions[name], prediction, score_flow)
                if math.isfinite(score): mtc_sum[name] += score; mtc_count[name] += 1; seq_mtc_sum[name] += score; seq_mtc_count[name] += 1
            pred_value = F.mse_loss(pending, state).item(); copy_value = F.mse_loss(previous_observation, state).item()
            pred_sum += pred_value; copy_sum += copy_value; frame_count += 1; seq_pred_sum += pred_value; seq_copy_sum += copy_value; seq_frames += 1
            state_rms_sum += state.square().mean().sqrt().item(); state_std_sum += state.std().item(); state_frame_count += 1
            state_temporal_sum += copy_value
            seq_state_rms_sum += state.square().mean().sqrt().item(); seq_state_std_sum += state.std().item(); seq_state_temporal_sum += copy_value; seq_state_frame_count += 1
            update_sum += delta.abs().mean().item(); update_rms += delta.square().mean().sqrt().item(); update_rel += (delta.square().mean().sqrt() / (observation.z4.square().mean().sqrt() + 1e-8)).item(); update_frames += 1
            seq_update_sum += delta.abs().mean().item(); seq_update_rms += delta.square().mean().sqrt().item(); seq_update_rel += (delta.square().mean().sqrt() / (observation.z4.square().mean().sqrt() + 1e-8)).item()
            pending, hidden = plugin.predict_next(state.detach(), error.detach(), hidden); previous_observation = state; previous_image = image; previous_predictions = {k: v.detach() for k, v in predictions.items()}
        for name in names:
            stats = seq_vc[name].stats()
            for length in (8, 16): mvc_sums[name][length] += stats[length]["sum"]; mvc_counts[name][length] += stats[length]["count"]
        per_sequence[sequence] = {}
        for name in names:
            per_sequence[sequence][name] = {"mIoU": float(torch.nanmean(compute_iou(seq_conf[name])).item()), "mVC8": seq_vc[name].values()[8], "mVC16": seq_vc[name].values()[16], "mTC": seq_mtc_sum[name] / max(seq_mtc_count[name], 1), "valid_frame_pairs": seq_mtc_count[name]}
        per_sequence[sequence]["ours"].update({"Rpred": seq_pred_sum / max(seq_copy_sum, 1e-12), "update_abs": seq_update_sum / max(seq_frames, 1), "update_rms": seq_update_rms / max(seq_frames, 1), "update_rel": seq_update_rel / max(seq_frames, 1), "state_rms": seq_state_rms_sum / max(seq_state_frame_count, 1), "state_std": seq_state_std_sum / max(seq_state_frame_count, 1), "state_temporal_mse": seq_state_temporal_sum / max(seq_frames, 1), "copy_mse": seq_copy_sum / max(seq_frames, 1)})
    metrics = {name: {"mIoU": float(torch.nanmean(compute_iou(confusion[name])).item()), "mVC8": mvc_sums[name][8] / max(mvc_counts[name][8], 1), "mVC16": mvc_sums[name][16] / max(mvc_counts[name][16], 1), "mTC": mtc_sum[name] / max(mtc_count[name], 1), "valid_frame_pairs": mtc_count[name]} for name in names}
    metrics["ours"].update({"Rpred": pred_sum / max(copy_sum, 1e-12), "pred_mse": pred_sum / max(frame_count, 1), "copy_mse": copy_sum / max(frame_count, 1), "update_abs": update_sum / max(update_frames, 1), "update_rms": update_rms / max(update_frames, 1), "update_rel": update_rel / max(update_frames, 1), "state_rms": state_rms_sum / max(state_frame_count, 1), "state_std": state_std_sum / max(state_frame_count, 1), "state_temporal_mse": state_temporal_sum / max(frame_count, 1)})
    metrics["delta"] = {key: metrics["ours"][key] - metrics["host"][key] for key in ("mIoU", "mVC8", "mVC16", "mTC")}; metrics["per_sequence"] = per_sequence
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_DEFAULT)
    parser.add_argument("--dynamics-checkpoint", default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    host, plugin, source = load_v2(args)
    train_data = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val_data = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups, val_groups = sequence_groups(train_data), sequence_groups(val_data)
    sanity = zero_update_sanity(host, plugin, next(iter(val_groups.values())))
    print(json.dumps({"zero_update_sanity": sanity}, sort_keys=True), flush=True)
    raft = FrozenRAFT()
    zero_metrics = evaluate(host, plugin, val_groups, raft)
    zero_delta = zero_metrics["delta"]
    if abs(zero_delta["mIoU"]) > 1e-4 or abs(zero_delta["mTC"]) > 1e-4:
        raise RuntimeError(
            "Zero-update metric identity failed: "
            f"delta_mIoU={zero_delta['mIoU']}, delta_mTC={zero_delta['mTC']}"
        )
    sanity["full9_metrics"] = {
        "host": zero_metrics["host"],
        "ours": zero_metrics["ours"],
        "delta": zero_delta,
    }
    print(json.dumps({"zero_update_metrics": sanity["full9_metrics"]}, sort_keys=True), flush=True)
    optimizer = torch.optim.AdamW(plugin.trainable_parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = [{"epoch": 0, "train": None, "validation": zero_metrics}]; best = None
    print(json.dumps({"epoch": 0, "validation": zero_metrics}, sort_keys=True), flush=True)
    for epoch in range(1, args.epochs + 1):
        train = train_epoch(host, plugin, raft, train_groups, optimizer)
        metrics = evaluate(host, plugin, val_groups, raft)
        row = {"epoch": epoch, "train": train, "validation": metrics}
        history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
        payload = {"experiment": "predictive_semantic_v21", "epoch": epoch, "model_state_dict": plugin.state_dict(), "metrics": metrics, "config": {"lambda_pred": LAMBDA_PRED, "lambda_tc": LAMBDA_TC, "tbptt": TBPTT, "lr": LEARNING_RATE}}
        torch.save(payload, out / f"epoch_{epoch:03d}.pt")
        score = metrics["ours"]["mIoU"]
        if best is None or score > best["mIoU"] or (score == best["mIoU"] and metrics["ours"]["mTC"] > best["mTC"]):
            best = {"epoch": epoch, "mIoU": score, "mTC": metrics["ours"]["mTC"]}; torch.save(payload, out / "best.pt")
        if epoch - best["epoch"] >= args.patience: break
    result = {"experiment": "Predify V2.1 predictive semantic state", "source_fast_b_checkpoint": args.fast_b_checkpoint, "trainable_modules": ["encoder", "predictor", "update_head"], "frozen": ["Host", "C4 adapter", "C4 writeback", "decoder"], "tbptt": TBPTT, "lambda_pred": LAMBDA_PRED, "lambda_tc": LAMBDA_TC, "lr": LEARNING_RATE, "max_epochs": args.epochs, "patience": args.patience, "zero_update_sanity": sanity, "epoch0_metrics": zero_metrics, "history": history, "best": best}
    result_dir = Path(args.result_output); result_dir.mkdir(parents=True, exist_ok=True); (result_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); (result_dir / "README.md").write_text("# Predify V2.1\nTask-aware predictive semantic state with causal Rhat -> R -> error -> next prediction. Task losses are detached from Predictor inputs.\n")


if __name__ == "__main__":
    main()
