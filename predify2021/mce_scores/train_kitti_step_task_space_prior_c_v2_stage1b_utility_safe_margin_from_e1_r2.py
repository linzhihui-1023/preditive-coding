"""C-V2 Stage 1B: Transport-Aware Safe-Margin Utility on frozen E1 base.

The validated E1 T+C model remains fully frozen. Only Correction Utility U is
trained. The deployed output is unchanged:

    C_t   = T_t * Warp(C_{t-1}, M_hat_t) + (1-T_t) * I_t
    L_U   = L_host + T_t * DeltaL_transport + U_t * C_t

This version replaces CE-optimal writeback supervision with a safe-margin target.
For the current true class y, define the continuous pairwise logit margin

    m_y(L) = L_y - max_{k != y} L_k.

A single kappa_base is calibrated once from the frozen deployed system on the
training split. On low-resolution pixels where both current Host and warped
previous Host are correct and the warp source is valid, we measure

    |m_y(L_host,t) - m_y(L_prior,t)|.

Each temporal frame contributes its 90th percentile; kappa_base is the median of
those frame-level percentiles. It is therefore a fixed system noise-floor
calibration, not a validation sweep.

The per-pixel transport-aware safety target is

    kappa_t = kappa_base
              + T_t * valid_warp * relu(m_y(L_prior,t) - kappa_base).

U_target is the smallest alpha in [0,1] for which

    m_y(L_transport + alpha*C_t) >= kappa_t.

If the safety margin is unreachable, alpha is chosen to maximize the achievable
true-class margin, equivalently minimizing the hinge margin risk below kappa_t.
The target is built under no_grad from the frozen E1 base. No gradient from U can
change T, C, Motion or the Host.
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
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    sequence_groups,
)
from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import (
    FAST_B_CHECKPOINT_DEFAULT,
    load_fast_b_model,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_only import (
    FULL9,
    IGNORE_LABEL,
    NUM_CLASSES,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_observer import (
    _host_observation,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_residual import (
    OBSERVER_CHECKPOINT_DEFAULT,
    _load_frozen_observer,
    _observe_motion,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask import (
    _initialize_motion,
    _load_frozen_residual,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask_r2 import (
    RESIDUAL_CHECKPOINT_DEFAULT,
    _validate_bounded_motion_checkpoint,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_spatial_mask_semantic_state_r2_main import (
    _update_semantic_state_zero_invalid,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v2_stage1b_utility_from_e1_r2 as base,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_correction_utility import (
    RecurrentCorrectionUtility,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    warp_low_logits,
)

SEED = 0
EPOCHS = 3
TBPTT_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
CALIBRATION_FRAME_QUANTILE = 0.90
UNREACHABLE_SEARCH_STEPS = 12
SLOPE_EPS = 1e-8

BASE_CHECKPOINT_DEFAULT = base.BASE_CHECKPOINT_DEFAULT
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v2_stage1b_utility_safe_margin_from_e1_r2"
)
RESULT_DEFAULT = (
    "results/"
    "kitti_step_task_space_prior_c_v2_stage1b_utility_safe_margin_from_e1_r2"
)


def _lowres_target(current_gt_cpu, low_size):
    return base._lowres_target(current_gt_cpu, low_size)


@torch.no_grad()
def _true_class_margin(logits, target):
    """True-class logit minus the strongest competing-class logit."""
    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected logits [B,C,H,W] and target [B,H,W]")
    safe_target = target.clamp(0, NUM_CLASSES - 1)
    true_logit = logits.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    competitor = logits.clone()
    competitor.scatter_(1, safe_target.unsqueeze(1), float("-inf"))
    max_competitor = competitor.max(dim=1).values
    return true_logit - max_competitor


@torch.no_grad()
def _calibrate_kappa_base(model, observer, residual, groups):
    """Calibrate one fixed safe-margin noise floor from frozen training dynamics."""
    frame_quantiles = []
    stable_pixels = 0
    calibrated_frames = 0

    for samples in groups.values():
        if len(samples) < 3:
            continue
        frame0 = _host_observation(model, samples[0])
        frame1 = _host_observation(model, samples[1])
        pending_motion, motion_hidden = _initialize_motion(
            observer, residual, frame0, frame1
        )
        _, _, previous_low, previous_c1, _ = frame1

        for frame_index in range(2, len(samples)):
            _, _, host_low, current_c1, _ = _host_observation(
                model, samples[frame_index]
            )
            current_gt = semantic_mask_from_panoptic_png(
                samples[frame_index]["mask_path"]
            )
            prior_low, prior_valid = warp_low_logits(previous_low, pending_motion)
            target = _lowres_target(current_gt, tuple(host_low.shape[-2:]))
            valid_label = target != IGNORE_LABEL
            safe_target = target.clamp(0, NUM_CLASSES - 1)

            current_correct = host_low.argmax(1) == safe_target
            prior_correct = prior_low.argmax(1) == safe_target
            stable = valid_label & prior_valid & current_correct & prior_correct

            if bool(stable.any()):
                current_margin = _true_class_margin(host_low.float(), target)
                prior_margin = _true_class_margin(prior_low.float(), target)
                disturbance = (current_margin - prior_margin).abs()[stable]
                if disturbance.numel() > 0:
                    q = torch.quantile(
                        disturbance,
                        torch.tensor(
                            CALIBRATION_FRAME_QUANTILE,
                            device=disturbance.device,
                            dtype=disturbance.dtype,
                        ),
                    )
                    if torch.isfinite(q):
                        frame_quantiles.append(float(q.item()))
                        stable_pixels += int(disturbance.numel())
                        calibrated_frames += 1

            observed_motion = _observe_motion(
                observer, previous_low, previous_c1, host_low, current_c1
            )
            prediction_error = F.softmax(host_low, dim=1) - F.softmax(
                prior_low, dim=1
            )
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion, prediction_error, motion_hidden
            )
            previous_low = host_low.detach()
            previous_c1 = current_c1.detach()
            pending_motion = next_motion.detach()
            motion_hidden = next_motion_hidden.detach()

    if not frame_quantiles:
        raise RuntimeError("Safe-margin calibration found no stable correct temporal pixels")
    values = torch.tensor(frame_quantiles, dtype=torch.float32)
    kappa_base = float(values.median().item())
    if not math.isfinite(kappa_base) or kappa_base <= 0.0:
        raise RuntimeError(f"Invalid calibrated kappa_base={kappa_base}")
    return {
        "kappa_base": kappa_base,
        "frame_quantile": CALIBRATION_FRAME_QUANTILE,
        "aggregation": "median_of_frame_quantiles",
        "calibrated_frames": calibrated_frames,
        "stable_pixels": stable_pixels,
        "frame_quantile_median": kappa_base,
        "frame_quantile_mean": float(values.mean().item()),
    }


@torch.no_grad()
def _transport_aware_kappa(
    prior_logits_low,
    transportability_low,
    prior_valid,
    target,
    kappa_base,
):
    history_margin = _true_class_margin(prior_logits_low.float(), target)
    history_strength = torch.relu(history_margin - float(kappa_base))
    route = transportability_low[:, 0].float() * prior_valid.float()
    kappa = float(kappa_base) + route * history_strength
    return kappa, history_margin


@torch.no_grad()
def _safe_margin_writeback_target(
    transport_base_low,
    semantic_state_low,
    prior_logits_low,
    transportability_low,
    prior_valid,
    current_gt_cpu,
    kappa_base,
):
    """Smallest alpha reaching safe margin; otherwise best achievable alpha."""
    if transport_base_low.shape != semantic_state_low.shape:
        raise ValueError("Transport base and Semantic State must share shape")
    target = _lowres_target(
        current_gt_cpu, tuple(transport_base_low.shape[-2:])
    )
    valid = target != IGNORE_LABEL
    safe_target = target.clamp(0, NUM_CLASSES - 1)

    base_logits = transport_base_low.detach().float()
    state = semantic_state_low.detach().float()
    kappa, history_margin = _transport_aware_kappa(
        prior_logits_low.detach(),
        transportability_low.detach(),
        prior_valid,
        target,
        kappa_base,
    )

    base_true = base_logits.gather(1, safe_target.unsqueeze(1))
    state_true = state.gather(1, safe_target.unsqueeze(1))
    pair_base = base_true - base_logits
    pair_slope = state_true - state

    class_index = torch.arange(
        NUM_CLASSES, device=base_logits.device
    ).view(1, NUM_CLASSES, 1, 1)
    competitor_mask = class_index != safe_target.unsqueeze(1)

    required_lower = torch.zeros_like(kappa)
    allowed_upper = torch.ones_like(kappa)
    infeasible_flat = torch.zeros_like(valid)

    positive = competitor_mask & (pair_slope > SLOPE_EPS)
    negative = competitor_mask & (pair_slope < -SLOPE_EPS)
    flat = competitor_mask & ~(positive | negative)

    raw_bound = (kappa.unsqueeze(1) - pair_base) / torch.where(
        pair_slope.abs() > SLOPE_EPS,
        pair_slope,
        torch.ones_like(pair_slope),
    )

    lower_candidates = torch.where(
        positive,
        raw_bound,
        torch.full_like(raw_bound, float("-inf")),
    )
    upper_candidates = torch.where(
        negative,
        raw_bound,
        torch.full_like(raw_bound, float("inf")),
    )
    required_lower = torch.maximum(
        required_lower, lower_candidates.max(dim=1).values
    )
    allowed_upper = torch.minimum(
        allowed_upper, upper_candidates.min(dim=1).values
    )
    infeasible_flat = (
        flat & (pair_base < kappa.unsqueeze(1))
    ).any(dim=1)

    lower = required_lower.clamp(0.0, 1.0)
    upper = allowed_upper.clamp(0.0, 1.0)
    feasible = valid & ~infeasible_flat & (required_lower <= 1.0) & (
        allowed_upper >= 0.0
    ) & (lower <= upper)

    alpha = torch.zeros_like(kappa)
    alpha[feasible] = lower[feasible]

    unreachable = valid & ~feasible
    if bool(unreachable.any()):
        # m(alpha) is the minimum of affine pairwise margins, hence concave.
        # Ternary search finds the risk-minimizing alpha when kappa is unreachable.
        lo = torch.zeros_like(kappa)
        hi = torch.ones_like(kappa)

        def margin_at(a):
            logits = base_logits + a.unsqueeze(1) * state
            return _true_class_margin(logits, target)

        for _ in range(UNREACHABLE_SEARCH_STEPS):
            a1 = (2.0 * lo + hi) / 3.0
            a2 = (lo + 2.0 * hi) / 3.0
            m1 = margin_at(a1)
            m2 = margin_at(a2)
            move_lo = unreachable & (m1 < m2)
            move_hi = unreachable & ~move_lo
            lo = torch.where(move_lo, a1, lo)
            hi = torch.where(move_hi, a2, hi)

        candidate = 0.5 * (lo + hi)
        margin_zero = margin_at(torch.zeros_like(kappa))
        margin_one = margin_at(torch.ones_like(kappa))
        margin_mid = margin_at(candidate)
        best_alpha = candidate
        best_margin = margin_mid
        use_zero = margin_zero >= best_margin
        best_alpha = torch.where(use_zero, torch.zeros_like(alpha), best_alpha)
        best_margin = torch.where(use_zero, margin_zero, best_margin)
        use_one = margin_one > best_margin
        best_alpha = torch.where(use_one, torch.ones_like(alpha), best_alpha)
        alpha[unreachable] = best_alpha[unreachable]

    base_margin = _true_class_margin(base_logits, target)
    already_safe = valid & (base_margin >= kappa)
    alpha[already_safe] = 0.0

    diagnostics = {
        "target_mean": float(alpha[valid].mean().item()) if bool(valid.any()) else 0.0,
        "already_safe_fraction": float(
            already_safe.sum().item() / max(valid.sum().item(), 1)
        ),
        "feasible_fraction": float(
            feasible.sum().item() / max(valid.sum().item(), 1)
        ),
        "unreachable_fraction": float(
            unreachable.sum().item() / max(valid.sum().item(), 1)
        ),
        "kappa_mean": float(kappa[valid].mean().item()) if bool(valid.any()) else 0.0,
        "history_margin_mean": float(
            history_margin[valid].mean().item()
        ) if bool(valid.any()) else 0.0,
        "base_margin_mean": float(base_margin[valid].mean().item()) if bool(valid.any()) else 0.0,
    }
    return alpha.clamp_(0.0, 1.0), valid, diagnostics


def _utility_loss(
    utility_logits_low,
    transport_base_low,
    semantic_state_low,
    prior_logits_low,
    transportability_low,
    prior_valid,
    current_gt_cpu,
    kappa_base,
):
    target, valid, target_diag = _safe_margin_writeback_target(
        transport_base_low,
        semantic_state_low,
        prior_logits_low,
        transportability_low,
        prior_valid,
        current_gt_cpu,
        kappa_base,
    )
    if not bool(valid.any()):
        return utility_logits_low.sum() * 0.0, target_diag
    loss = F.binary_cross_entropy_with_logits(
        utility_logits_low[:, 0][valid], target[valid]
    )
    return loss, target_diag


def _train_sequence(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    utility_predictor,
    samples,
    optimizer,
    tbptt_steps,
    kappa_base,
):
    if len(samples) < 3:
        return None

    frame0 = _host_observation(model, samples[0])
    frame1 = _host_observation(model, samples[1])
    pending_motion, motion_hidden = _initialize_motion(
        observer, residual, frame0, frame1
    )
    _, _, previous_low, previous_c1, _ = frame1

    transport_hidden = semantic_hidden = mask_hidden = utility_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    buffered_losses = []
    totals = {
        "frames": 0,
        "windows": 0,
        "utility_bce": 0.0,
        "utility_mean": 0.0,
        "utility_target_mean": 0.0,
        "already_safe_fraction": 0.0,
        "feasible_fraction": 0.0,
        "unreachable_fraction": 0.0,
        "kappa_mean": 0.0,
        "history_margin_mean": 0.0,
        "base_margin_mean": 0.0,
        "mask_mean": 0.0,
        "semantic_state_abs": 0.0,
        "total": 0.0,
    }

    for frame_index in range(2, len(samples)):
        _, _, host_low, current_c1, _ = _host_observation(
            model, samples[frame_index]
        )
        current_gt = semantic_mask_from_panoptic_png(
            samples[frame_index]["mask_path"]
        )

        with torch.no_grad():
            prior_low, prior_valid = warp_low_logits(previous_low, pending_motion)
            row = correction(
                current_c1,
                host_low,
                prior_low,
                pending_motion,
                transport_hidden,
                semantic_hidden,
            )
            transport_hidden = row["transport_hidden"]
            semantic_hidden = row["semantic_hidden"]
            delta_transport_low = row["delta_transport"]
            semantic_innovation_low = row["delta_semantic"]

            mask_logits_low, mask_hidden = mask_predictor(
                current_c1,
                host_low,
                prior_low,
                pending_motion,
                mask_hidden,
            )
            transportability_low = torch.sigmoid(mask_logits_low)
            semantic_state_low, _ = _update_semantic_state_zero_invalid(
                semantic_state_low,
                pending_motion,
                transportability_low,
                semantic_innovation_low,
            )
            transport_base_low = (
                host_low + transportability_low * delta_transport_low
            )

        utility_logits_low, utility_hidden = utility_predictor(
            current_c1.detach(),
            host_low.detach(),
            prior_low.detach(),
            pending_motion.detach(),
            transportability_low.detach(),
            delta_transport_low.detach(),
            semantic_state_low.detach(),
            utility_hidden,
        )
        utility_bce, target_diag = _utility_loss(
            utility_logits_low,
            transport_base_low,
            semantic_state_low,
            prior_low,
            transportability_low,
            prior_valid,
            current_gt,
            kappa_base,
        )
        if not torch.isfinite(utility_bce):
            raise FloatingPointError("Non-finite safe-margin utility loss")
        buffered_losses.append(utility_bce)

        with torch.no_grad():
            observed_motion = _observe_motion(
                observer, previous_low, previous_c1, host_low, current_c1
            )
            prediction_error = F.softmax(host_low, dim=1) - F.softmax(
                prior_low, dim=1
            )
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion, prediction_error, motion_hidden
            )

        totals["frames"] += 1
        totals["utility_bce"] += float(utility_bce.detach().item())
        totals["utility_mean"] += float(
            torch.sigmoid(utility_logits_low).mean().detach().item()
        )
        for key in (
            "target_mean",
            "already_safe_fraction",
            "feasible_fraction",
            "unreachable_fraction",
            "kappa_mean",
            "history_margin_mean",
            "base_margin_mean",
        ):
            out_key = "utility_target_mean" if key == "target_mean" else key
            totals[out_key] += target_diag[key]
        totals["mask_mean"] += float(transportability_low.mean().item())
        totals["semantic_state_abs"] += float(
            semantic_state_low.abs().mean().item()
        )

        boundary = (
            len(buffered_losses) == tbptt_steps
            or frame_index == len(samples) - 1
        )
        if boundary:
            window_loss = torch.stack(buffered_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            window_loss.backward()
            optimizer.step()
            totals["windows"] += 1
            totals["total"] += float(window_loss.detach().item())
            buffered_losses = []
            utility_hidden = utility_hidden.detach()

        previous_low = host_low.detach()
        previous_c1 = current_c1.detach()
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()

    frames = max(totals["frames"], 1)
    windows = max(totals["windows"], 1)
    for key in (
        "utility_bce",
        "utility_mean",
        "utility_target_mean",
        "already_safe_fraction",
        "feasible_fraction",
        "unreachable_fraction",
        "kappa_mean",
        "history_margin_mean",
        "base_margin_mean",
        "mask_mean",
        "semantic_state_abs",
    ):
        totals[key] /= frames
    totals["total"] /= windows
    return totals


def _train_epoch(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    utility_predictor,
    groups,
    optimizer,
    tbptt_steps,
    kappa_base,
):
    utility_predictor.train()
    rows = []
    for samples in groups.values():
        row = _train_sequence(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            utility_predictor,
            samples,
            optimizer,
            tbptt_steps,
            kappa_base,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("No valid training sequences")
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--base-checkpoint", default=BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--tbptt-steps", type=int, default=TBPTT_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs <= 0 or args.tbptt_steps <= 0:
        raise ValueError("epochs and tbptt-steps must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    _validate_bounded_motion_checkpoint(args.residual_checkpoint)
    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = _load_frozen_observer(args.observer_checkpoint)
    residual, residual_payload = _load_frozen_residual(
        args.residual_checkpoint, observer
    )
    correction, mask_predictor, base_payload = base._load_frozen_e1_base(
        args.base_checkpoint, observer
    )

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root), "train"
    )
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root), "val"
    )
    train_groups = sequence_groups(train)
    all_val_groups = sequence_groups(val)
    missing = [sequence for sequence in FULL9 if sequence not in all_val_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {sequence: all_val_groups[sequence] for sequence in FULL9}

    calibration = _calibrate_kappa_base(
        model, observer, residual, train_groups
    )
    kappa_base = float(calibration["kappa_base"])
    print(json.dumps({"safe_margin_calibration": calibration}, sort_keys=True), flush=True)

    utility_predictor = RecurrentCorrectionUtility(
        c1_channels=256,
        num_classes=NUM_CLASSES,
        motion_scale=observer.max_displacement_low,
    ).cuda()
    optimizer = torch.optim.AdamW(
        utility_predictor.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    raft = base.FrozenRAFT()

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)
    (result_output / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n"
    )

    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        train_stats = _train_epoch(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            utility_predictor,
            train_groups,
            optimizer,
            args.tbptt_steps,
            kappa_base,
        )
        metrics, diagnostics = base._evaluate(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            utility_predictor,
            val_groups,
            raft,
        )
        delta_vs_host = {
            name: {
                metric: metrics[name][metric] - metrics["host"][metric]
                for metric in ("mIoU", "mTC", "mVC8", "mVC16")
            }
            for name in ("e1_base", "e1_plus_utility")
        }
        delta_utility_vs_base = {
            metric: metrics["e1_plus_utility"][metric]
            - metrics["e1_base"][metric]
            for metric in ("mIoU", "mTC", "mVC8", "mVC16")
        }
        row = {
            "epoch": epoch,
            "kappa_base": kappa_base,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_host": delta_vs_host,
            "delta_utility_vs_e1_base": delta_utility_vs_base,
        }
        history.append(row)

        payload = {
            "experiment": "c_v2_stage1b_utility_safe_margin_from_e1_r2",
            "epoch": epoch,
            "architecture": {
                "base": "frozen validated E1 calibrated Spatial Mask + Semantic State",
                "output": "Host + T*DeltaL_transport + U*C_t",
                "U_role": "transport-aware safe-margin semantic-state writeback",
                "T_and_base_frozen": True,
                "U_does_not_gate_transport": True,
                "U_does_not_gate_state_recurrence": True,
            },
            "training_contract": {
                "only_trainable_module": "RecurrentCorrectionUtility",
                "margin": "true-class logit - strongest competing-class logit",
                "kappa_base": kappa_base,
                "kappa_calibration": calibration,
                "kappa_t": "kappa_base + T*valid_warp*relu(history_margin-kappa_base)",
                "U_target": (
                    "smallest alpha in [0,1] reaching kappa_t; if unreachable, "
                    "alpha maximizing achievable true-class margin"
                ),
                "base_checkpoint": args.base_checkpoint,
                "base_checkpoint_epoch": int(base_payload.get("epoch", -1)),
            },
            "observer_checkpoint": args.observer_checkpoint,
            "residual_checkpoint": args.residual_checkpoint,
            "residual_checkpoint_epoch": residual_payload.get("epoch"),
            "utility_state_dict": utility_predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
        }
        torch.save(payload, output / f"epoch_{epoch:03d}.pt")
        (result_output / f"epoch_{epoch:03d}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )

        key = base._selection_key(metrics)
        if best is None or key > best["selection_key"]:
            best = {
                "epoch": epoch,
                "selection_key": key,
                "metrics": metrics["e1_plus_utility"],
                "delta_vs_host": delta_vs_host["e1_plus_utility"],
                "delta_vs_e1_base": delta_utility_vs_base,
            }
            torch.save(payload, output / "best.pt")
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "experiment": "C-V2 Stage 1B Transport-Aware Safe-Margin Utility on Frozen E1 Base r2",
        "purpose": (
            "Replace confidence-deepening CE utility with minimal semantic writeback "
            "needed to reach a transport-aware temporal safety margin."
        ),
        "base_checkpoint": args.base_checkpoint,
        "calibration": calibration,
        "frozen": [
            "Host",
            "Motion Observer",
            "bounded r=2 Task-Alignment Residual",
            "E1 Transport/Innovation correction",
            "E1 Transportability Mask",
        ],
        "training": {
            "only_trainable": "RecurrentCorrectionUtility",
            "epochs": args.epochs,
            "tbptt_steps": args.tbptt_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "no_kappa_sweep": True,
        },
        "history": history,
        "best": best,
        "decision_rule": (
            "Judge Safe-Margin U directly against the same frozen E1 base. "
            "Do not open kappa, threshold, loss-weight or Motion sweeps."
        ),
    }
    (result_output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "best": best,
                "kappa_base": kappa_base,
                "checkpoint": str(output / "best.pt"),
                "result": str(result_output / "summary.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
