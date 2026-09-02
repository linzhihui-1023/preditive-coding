"""Train the two Semantic V3 experiments (Structure and Full).

V3 keeps the Host, adapters, dynamics, writeback and decoder frozen.  The
semantic state is initialized from frame 0 and only Z4 semantic parameters are
updated.  ``--variant both`` runs both requested experiments in one command;
the two variants are independent and write separate output directories.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.kitti_step_persistent_blur import BLUR_KERNEL_SIZE, warmup_frame_count
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT, ROLE_PREDICTOR_CHECKPOINT_DEFAULT, STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT, error_state, load_components, residual_writeback_host_feature, zero_state,
)
from predify2021.mce_scores.train_kitti_step_error_guided_semantic_restoration import (
    TRAIN_CONDITIONS, TRAIN_PATTERNS, training_assignment, pattern_uses_blur,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorRegulatedSemanticRestorationPredictor,
    HostFeature, UnifiedFeatures,
)

SEED, MAX_EPOCHS, PATIENCE, TBPTT_STEPS, LEARNING_RATE, WEIGHT_DECAY = 0, 15, 3, 8, 1e-4, 0.01
FAST_VALIDATION_SEQUENCES = ("0002", "0010", "0018")
LAMBDA_FEATURE = 1.0
LAMBDA_SEGMENTATION = 1e-4


def slice_state(state, index):
    return UnifiedFeatures(*(value[index:index + 1] for value in state.as_tuple()))


def encode_pair(model, sample, condition, use_blur):
    clean = load_image(sample)
    observed = clean
    if use_blur:
        import torchvision.transforms.functional as TF
        sigma = 2.25 if condition == "Blur-Mid" else 3.0
        observed = TF.gaussian_blur(clean, [BLUR_KERNEL_SIZE] * 2, [sigma] * 2)
    with torch.no_grad():
        raw = model.extract_backbone_features(torch.cat((clean, observed), dim=0))
        states = model.encode_backbone_features(raw)
    return slice_state(states, 1), slice_state(states, 0), type(raw)(*(value[1:2] for value in raw.as_tuple())), type(raw)(*(value[0:1] for value in raw.as_tuple())), tuple(clean.shape[-2:])


def residual_aware_loss(delta, target):
    per_element = F.smooth_l1_loss(delta, target, beta=0.01, reduction="none")
    q = target.detach().abs()
    weights = 1.0 + q.div(q.mean().clamp_min(1e-12)).clamp(0.0, 4.0)
    weights = weights / weights.mean().clamp_min(1e-12)
    return (weights * per_element).mean()


def train_sequence(model, predictor, samples, epoch, sequence_index, optimizer, residual_aware, tbptt_steps, max_steps=0):
    if len(samples) < 2:
        return {"steps": 0, "feature_loss_sum": 0.0, "segmentation_loss_sum": 0.0, "total_loss_sum": 0.0, "segmentation_windows": 0}
    condition, pattern = training_assignment(epoch, sequence_index)
    first, _, _, _, _ = encode_pair(model, samples[0], condition, pattern_uses_blur(pattern, 0, len(samples)))
    pending, h4, h1 = predictor.predict_next(first, zero_state(first), None, None)
    h_sem = predictor.initial_semantic_state(first)
    error_stats = predictor.initial_error_temporal_statistics()
    losses, feature_total, segmentation_total, total_objective, segmentation_windows = [], 0.0, 0.0, 0.0, 0
    steps = min(len(samples) - 1, max_steps) if max_steps else len(samples) - 1
    for offset in range(steps):
        frame = offset + 1
        observation, clean, raw, _, output_size = encode_pair(model, samples[frame], condition, pattern_uses_blur(pattern, frame, len(samples)))
        prediction_error = error_state(observation, pending)
        restored, h_sem, diagnostics = predictor.restore_current(
            observation, pending, h_sem, error_temporal_state=error_stats
        )
        error_stats = diagnostics["error_temporal_state"]
        target = clean.z4.detach()
        if residual_aware:
            loss = residual_aware_loss(restored.z4 - observation.z4, target - observation.z4)
        else:
            loss = F.smooth_l1_loss(restored.z4, target)
        losses.append(loss); feature_total += float(loss.detach().item())
        with torch.no_grad():
            pending, h4, h1 = predictor.predict_next(observation, prediction_error, h4, h1)
        window_end = len(losses) == tbptt_steps or frame == steps
        if window_end:
            feature_window = torch.stack(losses).mean()
            target_mask = semantic_mask_from_panoptic_png(samples[frame]["mask_path"]).unsqueeze(0).to(observation.z4.device)
            segmentation_logits = decode_z4(model, raw, observation, restored, output_size)
            segmentation_loss = F.cross_entropy(segmentation_logits, target_mask, ignore_index=255)
            objective = LAMBDA_FEATURE * feature_window + LAMBDA_SEGMENTATION * segmentation_loss
            optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
            segmentation_total += float(segmentation_loss.detach().item()); total_objective += float(objective.detach().item()); segmentation_windows += 1
            h_sem = h_sem.detach(); error_stats = error_stats.detach() if error_stats is not None else None; losses = []
    return {"steps": len(samples) - 1 if not max_steps else min(len(samples) - 1, max_steps), "feature_loss_sum": feature_total, "segmentation_loss_sum": segmentation_total, "total_loss_sum": total_objective, "segmentation_windows": segmentation_windows}


def train_epoch(model, predictor, groups, epoch, optimizer, residual_aware, tbptt_steps, max_steps=0):
    predictor.train(); steps = feature_total = segmentation_total = total_objective = 0.0; segmentation_windows = 0
    for index, samples in enumerate(groups.values()):
        row = train_sequence(model, predictor, samples, epoch, index, optimizer, residual_aware, tbptt_steps, max_steps)
        steps += row["steps"]; feature_total += row["feature_loss_sum"]; segmentation_total += row["segmentation_loss_sum"]; total_objective += row["total_loss_sum"]; segmentation_windows += row["segmentation_windows"]
    return {"step_count": int(steps), "feature_loss": feature_total / max(steps, 1), "segmentation_loss": segmentation_total / max(segmentation_windows, 1), "total_loss": total_objective / max(segmentation_windows, 1), "segmentation_windows": segmentation_windows}


def decode_z4(model, raw, observation, restored, output_size):
    zero = zero_state(observation)
    delta = UnifiedFeatures(zero.z1, zero.z2, zero.z3, restored.z4 - observation.z4)
    return model.decode_from_host_feature(residual_writeback_host_feature(model, raw, delta, output_size))


def evaluate_condition(model, predictor, groups, condition, residual_aware=False, max_frames=0):
    confusion = {name: torch.zeros((19, 19), dtype=torch.int64) for name in ("blur", "restored", "clean")}
    sums = {"obs": 0.0, "restored": 0.0, "state": 0.0, "steps": 0, "clean_steps": 0}
    predictor.eval()
    with torch.inference_mode():
        for samples in groups.values():
            if len(samples) < 2: continue
            sequence_steps = 0
            onset = warmup_frame_count(len(samples))
            first, _, _, _, _ = encode_pair(model, samples[0], condition, False)
            pending, h4, h1 = predictor.predict_next(first, zero_state(first), None, None)
            h_sem = predictor.initial_semantic_state(first)
            error_stats = predictor.initial_error_temporal_statistics()
            for frame in range(1, len(samples)):
                if max_frames and sequence_steps >= max_frames: break
                obs, clean, raw, clean_raw, output_size = encode_pair(model, samples[frame], condition, frame >= onset)
                error = error_state(obs, pending)
                restored, h_sem, diagnostics = predictor.restore_current(
                    obs, pending, h_sem, error_temporal_state=error_stats
                )
                error_stats = diagnostics["error_temporal_state"]
                if frame >= onset:
                    mask = semantic_mask_from_panoptic_png(samples[frame]["mask_path"])
                    update_confusion_matrix(confusion["blur"], model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size)).argmax(1).squeeze(0).cpu(), mask)
                    update_confusion_matrix(confusion["restored"], decode_z4(model, raw, obs, restored, output_size).argmax(1).squeeze(0).cpu(), mask)
                    sums["obs"] += float((obs.z4.float() - clean.z4.float()).square().mean().item())
                    sums["restored"] += float((restored.z4.float() - clean.z4.float()).square().mean().item())
                    sums["state"] += float((h_sem.float() - clean.z4.float()).square().mean().item())
                    sums["steps"] += 1
                    sequence_steps += 1
                    update_confusion_matrix(confusion["clean"], model.decode_from_host_feature(HostFeature(clean_raw.c4, clean_raw.c1, output_size)).argmax(1).squeeze(0).cpu(), mask)
                pending, h4, h1 = predictor.predict_next(obs, error, h4, h1)
    count = max(sums["steps"], 1)
    metrics = {name: float(torch.nanmean(compute_iou(value)).item()) for name, value in confusion.items()}
    obs, restored, state = sums["obs"] / count, sums["restored"] / count, sums["state"] / count
    return {"effective_frame_count": sums["steps"], "blur_mIoU": metrics["blur"], "restored_mIoU": metrics["restored"], "clean_mIoU": metrics["clean"], "feature_recovery": 1.0 - restored / max(obs, 1e-12), "state_recovery": 1.0 - state / max(obs, 1e-12), "z4_loss": restored}


def smoke_checks(model, predictor, observation, raw, output_size, mask, use_error_temporal_stats=False):
    checks = {}
    checks["frame0_history_exact"] = bool(torch.equal(predictor.initial_semantic_state(observation), observation.z4.detach()))
    h = torch.randn_like(observation.z4); eps1 = torch.randn_like(observation.z4); eps2 = torch.randn_like(observation.z4)
    stats = None
    if use_error_temporal_stats:
        from predify2021.model_factory.deeplabv3plus_resnet50.semantic_recurrent_predictor import update_error_temporal_statistics
        stats = update_error_temporal_statistics(None, eps1).as_features()
    _, d1 = predictor.semantic_state_cell(observation.z4, eps1, h, error_history_statistics=stats); _, d2 = predictor.semantic_state_cell(observation.z4, eps2, h, error_history_statistics=stats)
    checks["error_does_not_change_candidate"] = bool(torch.equal(d1["semantic_candidate"], d2["semantic_candidate"]))
    checks["error_changes_gain"] = bool(not torch.equal(d1["semantic_update_gain"], d2["semantic_update_gain"]))
    _, d3 = predictor.semantic_state_cell(observation.z4 + 1.0, eps1, h, error_history_statistics=stats)
    checks["observation_changes_candidate"] = bool(not torch.equal(d1["semantic_candidate"], d3["semantic_candidate"]))
    candidate = d1["semantic_candidate"]
    output_k0, _ = predictor.semantic_state_cell(
        observation.z4,
        eps1,
        h,
        error_history_statistics=stats,
        update_gain_override=torch.zeros_like(h),
    )
    output_k1, diagnostics_k1 = predictor.semantic_state_cell(
        observation.z4,
        eps1,
        h,
        error_history_statistics=stats,
        update_gain_override=torch.ones_like(h),
    )
    checks["replacement_k0"] = bool(torch.allclose(output_k0, h))
    checks["replacement_k1"] = bool(torch.allclose(output_k1, diagnostics_k1["semantic_candidate"]))
    checks["dynamics_frozen"] = sum(p.numel() for n in predictor.DYNAMICS_MODULES for p in getattr(predictor, n).parameters() if p.requires_grad) == 0
    checks["backbone_frozen"] = all(not parameter.requires_grad for parameter in model.backbone.parameters())
    checks["writeback_frozen"] = all(not parameter.requires_grad for parameter in model.host_conditioned_writebacks.parameters())
    checks["decoder_frozen"] = all(not parameter.requires_grad for parameter in model.decode_head.parameters())
    predictor.zero_grad(set_to_none=True)
    current = UnifiedFeatures(observation.z1, observation.z2, observation.z3, observation.z4 + 0.1 * torch.randn_like(observation.z4))
    semantic_hidden = predictor.initial_semantic_state(observation)
    restored, _, diagnostics = predictor.restore_current(observation, current, semantic_hidden, error_temporal_state=predictor.initial_error_temporal_statistics())
    logits = decode_z4(model, raw, observation, restored, output_size)
    segmentation_loss = F.cross_entropy(logits, mask.unsqueeze(0).to(logits.device), ignore_index=255)
    checks["segmentation_loss_finite"] = bool(torch.isfinite(segmentation_loss).item())
    segmentation_loss.backward()
    checks["semantic_branch_receives_segmentation_gradient"] = any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.detach().abs().sum() > 0 for parameter in predictor.semantic_parameters())
    checks["backbone_has_no_gradient"] = all(parameter.grad is None for parameter in model.backbone.parameters())
    checks["writeback_has_no_gradient"] = all(parameter.grad is None for parameter in model.host_conditioned_writebacks.parameters())
    checks["decoder_has_no_gradient"] = all(parameter.grad is None for parameter in model.decode_head.parameters())
    checks["all_finite"] = all(torch.isfinite(parameter.grad).all() for parameter in predictor.parameters() if parameter.grad is not None)
    predictor.zero_grad(set_to_none=True)
    return checks


def run_variant(args, variant):
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model, source = load_components(STATIC_CHECKPOINT_DEFAULT, ADAPTER_CHECKPOINT_DEFAULT, args.dynamics_checkpoint, WRITEBACK_CHECKPOINT_DEFAULT)
    use_error_temporal_stats = variant == "v3_temporal_stats"
    predictor = ErrorRegulatedSemanticRestorationPredictor(use_error_temporal_stats=use_error_temporal_stats).cuda(); predictor.load_dynamics_from_role_separated_state_dict(source.state_dict()); predictor.freeze_dynamics(); model.requires_grad_(False); model.eval()
    semantic_parameters = [p for p in predictor.semantic_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(semantic_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    train = sequence_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")); val_all = sequence_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")); missing = [sequence for sequence in FAST_VALIDATION_SEQUENCES if sequence not in val_all]
    if missing: raise RuntimeError(f"Missing fixed fast-validation sequences: {missing}")
    val = {sequence: val_all[sequence] for sequence in FAST_VALIDATION_SEQUENCES}
    epochs, train_max, val_max = args.epochs, 0, 0; output = Path(args.output) / variant
    if args.smoke: train, val = dict(list(train.items())[:1]), dict(list(val.items())[:1]); epochs, train_max, val_max, output = 1, 16, 32, output / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    first, _, raw_first, _, output_size = encode_pair(model, next(iter(train.values()))[0], "Blur-Mid", False)
    first_mask = semantic_mask_from_panoptic_png(next(iter(train.values()))[0]["mask_path"])
    checks = smoke_checks(model, predictor, first, raw_first, output_size, first_mask, use_error_temporal_stats); history = []; best = None; stale = 0
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, predictor, train, epoch, optimizer, variant == "v3_full", args.tbptt_steps, train_max)
        conditions = {c: evaluate_condition(model, predictor, val, c, variant == "v3_full", val_max) for c in TRAIN_CONDITIONS}
        val_metrics = {"conditions": conditions, "mean_restored_mIoU": sum(v["restored_mIoU"] for v in conditions.values()) / 2, "mean_loss": sum(v["z4_loss"] for v in conditions.values()) / 2}
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}; history.append(row); print(json.dumps({"variant": variant, **row}, sort_keys=True), flush=True)
        checkpoint = {"model_state_dict": predictor.state_dict(), "source_dynamics_checkpoint": args.dynamics_checkpoint, "epoch": epoch, "use_error_temporal_stats": use_error_temporal_stats}
        torch.save(checkpoint, output / f"epoch_{epoch:03d}.pt")
        if best is None or (val_metrics["mean_restored_mIoU"], -val_metrics["mean_loss"]) > (best["val"]["mean_restored_mIoU"], -best["val"]["mean_loss"]):
            best = row; stale = 0; torch.save({"model_state_dict": predictor.state_dict(), "source_dynamics_checkpoint": args.dynamics_checkpoint, "epoch": epoch, "use_error_temporal_stats": use_error_temporal_stats}, output / "best.pt")
        else: stale += 1
        if stale >= args.patience: break
    summary = {"experiment": f"kitti_step_semantic_{variant}", "variant": variant, "checkpoint": str(output / "best.pt"), "trainable_parameter_count": sum(p.numel() for p in semantic_parameters), "dynamics_trainable_parameter_count": 0, "error_temporal_stats_trainable_parameter_count": sum(p.numel() for p in predictor.semantic_state_cell.history_projection.parameters()) if use_error_temporal_stats else 0, "config": {"epochs": epochs, "patience": args.patience, "tbptt_steps": args.tbptt_steps, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "loss": "ResidualAwareSmoothL1(beta=0.01)" if variant == "v3_full" else "SmoothL1", "fast_validation_sequences": FAST_VALIDATION_SEQUENCES, "use_error_temporal_stats": use_error_temporal_stats}, "smoke_checks": checks, "history": history, "best": best}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); (output / "training_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    return summary


def run_smoke_check(args):
    model, source = load_components(STATIC_CHECKPOINT_DEFAULT, ADAPTER_CHECKPOINT_DEFAULT, args.dynamics_checkpoint, WRITEBACK_CHECKPOINT_DEFAULT)
    predictor = ErrorRegulatedSemanticRestorationPredictor(use_error_temporal_stats=True).cuda()
    predictor.load_dynamics_from_role_separated_state_dict(source.state_dict()); predictor.freeze_dynamics(); model.requires_grad_(False); model.eval(); predictor.eval()
    sample = next(iter(sequence_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")).values()))[0]
    observation, _, raw, _, output_size = encode_pair(model, sample, "Blur-Mid", False)
    checks = smoke_checks(model, predictor, observation, raw, output_size, semantic_mask_from_panoptic_png(sample["mask_path"]), True)
    print(json.dumps({"smoke_checks": checks}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="/home/lin/predify/kitti_step"); parser.add_argument("--output", default="/home/lin/predify/experiments/kitti_step_semantic_v3"); parser.add_argument("--variant", choices=("v3_structure", "v3_temporal_stats", "v3_full", "both"), default="both"); parser.add_argument("--dynamics-checkpoint", default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT); parser.add_argument("--epochs", type=int, default=MAX_EPOCHS); parser.add_argument("--patience", type=int, default=PATIENCE); parser.add_argument("--tbptt-steps", type=int, default=TBPTT_STEPS); parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE); parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY); parser.add_argument("--seed", type=int, default=SEED); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--smoke-check-only", action="store_true"); args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    if args.smoke_check_only:
        run_smoke_check(args); return
    variants = ("v3_structure", "v3_temporal_stats") if args.variant == "both" else (args.variant,)
    result = {variant: run_variant(args, variant) for variant in variants}; print(json.dumps({"variants": list(result)}, indent=2), flush=True)


if __name__ == "__main__": main()
