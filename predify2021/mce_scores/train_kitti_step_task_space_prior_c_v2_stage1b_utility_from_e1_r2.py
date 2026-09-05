"""C-V2 Stage 1B: train Correction Utility U on the validated E1 T+C base.

This is the decision-focused utility experiment.  The previously validated E1
Spatial-Mask + carried Semantic-State checkpoint is loaded and frozen.  Only U
is trained, so the run answers one question cleanly: does selective semantic
writeback improve the existing E1 base?

Frozen deployed base:
    C_t = T_t * Warp(C_{t-1}, M_hat_t) + (1-T_t) * I_t
    L_base = L_host + T_t * DeltaL_transport + C_t

Utility candidate:
    L_U = L_host + T_t * DeltaL_transport + U_t * C_t

U is supervised directly as the loss-minimizing writeback coefficient alpha in
[0,1] for the low-resolution semantic state.  For each valid pixel:

    alpha* = argmin_{alpha in [0,1]} CE(L_transport + alpha*C_t, Y)

Cross-entropy is convex along this one-dimensional line, so alpha* is obtained
by endpoint derivative tests plus bisection.  This makes the training target
match the deployed multiplicative role of U without an arbitrary utility-to-
amplitude mapping.
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
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import (
    FAST_B_CHECKPOINT_DEFAULT,
    load_fast_b_model,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_only import (
    FULL9,
    IGNORE_LABEL,
    NUM_CLASSES,
    _pair_mtc,
    _upsample_prior,
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
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_correction_utility import (
    RecurrentCorrectionUtility,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    warp_low_logits,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_role_decoupling import (
    RecurrentTransportabilityMask,
    RoleSeparatedTaskCorrection,
)

SEED = 0
EPOCHS = 3
TBPTT_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
BISECTION_STEPS = 8
BASE_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v2_stage1b_spatial_mask_semantic_state_r2/"
    "epoch_001.pt"
)
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v2_stage1b_utility_from_e1_r2"
)
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v2_stage1b_utility_from_e1_r2"
CANDIDATES = ("host", "e1_base", "e1_plus_utility")


def _load_frozen_e1_base(path, observer):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != "c_v2_stage1b_spatial_mask_semantic_state_r2":
        raise RuntimeError("Base checkpoint is not the calibrated Spatial-Mask Semantic-State model")
    if int(payload.get("epoch", -1)) != 1:
        raise RuntimeError("Utility attribution run requires the validated Epoch-1 base checkpoint")

    correction = RoleSeparatedTaskCorrection(
        c1_channels=256,
        num_classes=NUM_CLASSES,
        motion_scale=observer.max_displacement_low,
    ).cuda()
    mask_predictor = RecurrentTransportabilityMask(
        c1_channels=256,
        num_classes=NUM_CLASSES,
        motion_scale=observer.max_displacement_low,
    ).cuda()
    correction.load_state_dict(payload["correction_state_dict"], strict=True)
    mask_predictor.load_state_dict(payload["mask_state_dict"], strict=True)
    correction.requires_grad_(False).eval()
    mask_predictor.requires_grad_(False).eval()
    return correction, mask_predictor, payload


def _lowres_target(current_gt_cpu, low_size):
    target = current_gt_cpu.cuda(non_blocking=True)[None, None].float()
    target = F.interpolate(target, size=low_size, mode="nearest")[:, 0].long()
    return target


@torch.no_grad()
def _optimal_writeback_target(base_logits_low, semantic_state_low, current_gt_cpu):
    """Return the CE-minimizing alpha in [0,1] for base + alpha*C per pixel."""
    if base_logits_low.shape != semantic_state_low.shape:
        raise ValueError("Base logits and Semantic State must share shape")
    target = _lowres_target(current_gt_cpu, tuple(base_logits_low.shape[-2:]))
    valid = target != IGNORE_LABEL
    safe_target = target.clamp(0, NUM_CLASSES - 1)

    base = base_logits_low.detach().float()
    state = semantic_state_low.detach().float()
    state_true = state.gather(1, safe_target.unsqueeze(1)).squeeze(1)

    def derivative(alpha):
        logits = base + alpha.unsqueeze(1) * state
        probability = F.softmax(logits, dim=1)
        return (probability * state).sum(dim=1) - state_true

    zero = torch.zeros_like(target, dtype=base.dtype)
    one = torch.ones_like(zero)
    d0 = derivative(zero)
    d1 = derivative(one)

    alpha = torch.zeros_like(zero)
    full = valid & (d1 <= 0.0)
    interior = valid & (d0 < 0.0) & (d1 > 0.0)
    alpha[full] = 1.0

    lo = torch.zeros_like(zero)
    hi = torch.ones_like(zero)
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        dm = derivative(mid)
        move_lo = interior & (dm < 0.0)
        move_hi = interior & ~move_lo
        lo = torch.where(move_lo, mid, lo)
        hi = torch.where(move_hi, mid, hi)
    alpha[interior] = 0.5 * (lo[interior] + hi[interior])
    return alpha.clamp_(0.0, 1.0), valid


def _utility_loss(utility_logits_low, base_logits_low, semantic_state_low, current_gt_cpu):
    target, valid = _optimal_writeback_target(
        base_logits_low, semantic_state_low, current_gt_cpu
    )
    if not bool(valid.any()):
        return utility_logits_low.sum() * 0.0, 0.0
    loss = F.binary_cross_entropy_with_logits(
        utility_logits_low[:, 0][valid], target[valid]
    )
    return loss, float(target[valid].mean().item())


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
):
    if len(samples) < 3:
        return None

    frame0 = _host_observation(model, samples[0])
    frame1 = _host_observation(model, samples[1])
    pending_motion, motion_hidden = _initialize_motion(observer, residual, frame0, frame1)
    _, _, previous_low, previous_c1, _ = frame1

    transport_hidden = semantic_hidden = mask_hidden = None
    utility_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    buffered_losses = []
    totals = {
        "frames": 0,
        "windows": 0,
        "utility_bce": 0.0,
        "utility_mean": 0.0,
        "utility_target_mean": 0.0,
        "mask_mean": 0.0,
        "semantic_state_abs": 0.0,
        "total": 0.0,
    }

    for frame_index in range(2, len(samples)):
        _, _, host_low, current_c1, _ = _host_observation(model, samples[frame_index])
        current_gt = semantic_mask_from_panoptic_png(samples[frame_index]["mask_path"])

        with torch.no_grad():
            prior_low, _ = warp_low_logits(previous_low, pending_motion)
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
            transport_base_low = host_low + transportability_low * delta_transport_low

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
        utility_bce, target_mean = _utility_loss(
            utility_logits_low,
            transport_base_low,
            semantic_state_low,
            current_gt,
        )
        if not torch.isfinite(utility_bce):
            raise FloatingPointError("Non-finite utility loss")
        buffered_losses.append(utility_bce)

        with torch.no_grad():
            observed_motion = _observe_motion(
                observer, previous_low, previous_c1, host_low, current_c1
            )
            prediction_error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion, prediction_error, motion_hidden
            )

        totals["frames"] += 1
        totals["utility_bce"] += float(utility_bce.detach().item())
        totals["utility_mean"] += float(torch.sigmoid(utility_logits_low).mean().detach().item())
        totals["utility_target_mean"] += target_mean
        totals["mask_mean"] += float(transportability_low.mean().item())
        totals["semantic_state_abs"] += float(semantic_state_low.abs().mean().item())

        boundary = len(buffered_losses) == tbptt_steps or frame_index == len(samples) - 1
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
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("No valid training sequences")
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


@torch.inference_mode()
def _evaluate(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    utility_predictor,
    groups,
    raft,
):
    utility_predictor.eval()
    confusion = {
        name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
        for name in CANDIDATES
    }
    mtc_sum = {name: 0.0 for name in CANDIDATES}
    mtc_count = {name: 0 for name in CANDIDATES}
    vc_sum = {name: {8: 0.0, 16: 0.0} for name in CANDIDATES}
    vc_count = {name: {8: 0, 16: 0} for name in CANDIDATES}
    diagnostics = {
        "frames_with_temporal_prior": 0,
        "mask_mean": 0.0,
        "utility_mean": 0.0,
        "semantic_state_abs": 0.0,
    }

    for sequence in FULL9:
        previous = None
        pending_motion = motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = utility_hidden = None
        semantic_state_low = None
        previous_predictions = {}
        seq_vc = {name: VideoConsistency() for name in CANDIDATES}

        for sample in groups[sequence]:
            image, host_logits, host_low, current_c1, output_size = _host_observation(model, sample)
            host_pred = host_logits.argmax(1)
            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"])
            previous_image_for_mtc = previous[0] if previous is not None else None

            if previous is None:
                base_pred = utility_pred = host_pred
                semantic_state_low = torch.zeros_like(host_low)
                previous = (image, host_low.detach(), current_c1.detach())
            elif pending_motion is None:
                previous_image, previous_low, previous_c1 = previous
                observed = _observe_motion(observer, previous_low, previous_c1, host_low, current_c1)
                error = F.softmax(host_low, dim=1) - F.softmax(previous_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(observed, error, None)
                base_pred = utility_pred = host_pred
                previous = (image, host_low.detach(), current_c1.detach())
            else:
                previous_image, previous_low, previous_c1 = previous
                prior_low, _ = warp_low_logits(previous_low, pending_motion)
                row = correction(current_c1, host_low, prior_low, pending_motion, transport_hidden, semantic_hidden)
                transport_hidden = row["transport_hidden"]
                semantic_hidden = row["semantic_hidden"]
                delta_transport_low = row["delta_transport"]
                semantic_innovation_low = row["delta_semantic"]
                mask_logits_low, mask_hidden = mask_predictor(
                    current_c1, host_low, prior_low, pending_motion, mask_hidden
                )
                transportability_low = torch.sigmoid(mask_logits_low)
                semantic_state_low, _ = _update_semantic_state_zero_invalid(
                    semantic_state_low,
                    pending_motion,
                    transportability_low,
                    semantic_innovation_low,
                )
                utility_logits_low, utility_hidden = utility_predictor(
                    current_c1,
                    host_low,
                    prior_low,
                    pending_motion,
                    transportability_low,
                    delta_transport_low,
                    semantic_state_low,
                    utility_hidden,
                )
                utility_low = torch.sigmoid(utility_logits_low)
                base_delta_low = transportability_low * delta_transport_low + semantic_state_low
                utility_delta_low = (
                    transportability_low * delta_transport_low + utility_low * semantic_state_low
                )
                base_pred = (host_logits + _upsample_prior(base_delta_low, output_size)).argmax(1)
                utility_pred = (host_logits + _upsample_prior(utility_delta_low, output_size)).argmax(1)

                diagnostics["frames_with_temporal_prior"] += 1
                diagnostics["mask_mean"] += float(transportability_low.mean().item())
                diagnostics["utility_mean"] += float(utility_low.mean().item())
                diagnostics["semantic_state_abs"] += float(semantic_state_low.abs().mean().item())

                observed = _observe_motion(observer, previous_low, previous_c1, host_low, current_c1)
                error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(
                    observed, error, motion_hidden
                )
                previous = (image, host_low.detach(), current_c1.detach())

            predictions = {"host": host_pred, "e1_base": base_pred, "e1_plus_utility": utility_pred}
            for name, prediction in predictions.items():
                pc = prediction.squeeze(0).cpu()
                update_confusion_matrix(confusion[name], pc, gt_cpu)
                seq_vc[name].update(gt_cpu, pc)

            if previous_image_for_mtc is not None:
                teacher_full = raft.current_to_previous(image, previous_image_for_mtc)
                for name, prediction in predictions.items():
                    score = _pair_mtc(previous_predictions[name], prediction, teacher_full)
                    if math.isfinite(score):
                        mtc_sum[name] += score
                        mtc_count[name] += 1

            previous_predictions = {name: prediction.detach() for name, prediction in predictions.items()}

        for name in CANDIDATES:
            stats = seq_vc[name].stats()
            for length in (8, 16):
                vc_sum[name][length] += stats[length]["sum"]
                vc_count[name][length] += stats[length]["count"]

    metrics = {
        name: {
            "mIoU": float(torch.nanmean(compute_iou(confusion[name])).item()),
            "mTC": mtc_sum[name] / mtc_count[name] if mtc_count[name] else float("nan"),
            "mVC8": vc_sum[name][8] / vc_count[name][8] if vc_count[name][8] else float("nan"),
            "mVC16": vc_sum[name][16] / vc_count[name][16] if vc_count[name][16] else float("nan"),
        }
        for name in CANDIDATES
    }
    frames = max(diagnostics["frames_with_temporal_prior"], 1)
    for key in ("mask_mean", "utility_mean", "semantic_state_abs"):
        diagnostics[key] /= frames
    return metrics, diagnostics


def _selection_key(metrics):
    host = metrics["host"]
    candidate = metrics["e1_plus_utility"]
    semantic_non_degraded = candidate["mIoU"] >= host["mIoU"]
    if semantic_non_degraded:
        return (1, candidate["mTC"], candidate["mIoU"])
    return (0, candidate["mIoU"], candidate["mTC"])


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
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    _validate_bounded_motion_checkpoint(args.residual_checkpoint)
    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = _load_frozen_observer(args.observer_checkpoint)
    residual, residual_payload = _load_frozen_residual(args.residual_checkpoint, observer)
    correction, mask_predictor, base_payload = _load_frozen_e1_base(args.base_checkpoint, observer)
    utility_predictor = RecurrentCorrectionUtility(
        c1_channels=256,
        num_classes=NUM_CLASSES,
        motion_scale=observer.max_displacement_low,
    ).cuda()
    optimizer = torch.optim.AdamW(
        utility_predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    raft = FrozenRAFT()

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train)
    all_val_groups = sequence_groups(val)
    missing = [sequence for sequence in FULL9 if sequence not in all_val_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {sequence: all_val_groups[sequence] for sequence in FULL9}

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        train_stats = _train_epoch(
            model, observer, residual, correction, mask_predictor,
            utility_predictor, train_groups, optimizer, args.tbptt_steps
        )
        metrics, diagnostics = _evaluate(
            model, observer, residual, correction, mask_predictor,
            utility_predictor, val_groups, raft
        )
        delta_vs_host = {
            name: {
                metric: metrics[name][metric] - metrics["host"][metric]
                for metric in ("mIoU", "mTC", "mVC8", "mVC16")
            }
            for name in ("e1_base", "e1_plus_utility")
        }
        delta_utility_vs_base = {
            metric: metrics["e1_plus_utility"][metric] - metrics["e1_base"][metric]
            for metric in ("mIoU", "mTC", "mVC8", "mVC16")
        }
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_host": delta_vs_host,
            "delta_utility_vs_e1_base": delta_utility_vs_base,
        }
        history.append(row)
        payload = {
            "experiment": "c_v2_stage1b_utility_from_e1_r2",
            "epoch": epoch,
            "architecture": {
                "base": "frozen validated E1 calibrated Spatial Mask + Semantic State",
                "output": "Host + T*DeltaL_transport + U*C_t",
                "U_role": "loss-minimizing semantic-state writeback coefficient",
                "T_and_base_frozen": True,
                "U_does_not_gate_transport": True,
                "U_does_not_gate_state_recurrence": True,
            },
            "training_contract": {
                "only_trainable_module": "RecurrentCorrectionUtility",
                "U_target": "argmin_alpha CE(L_transport + alpha*C_t,Y), alpha in [0,1]",
                "U_target_solver": f"convex endpoint test + {BISECTION_STEPS}-step bisection",
                "base_checkpoint": args.base_checkpoint,
                "base_checkpoint_epoch": int(base_payload.get("epoch", -1)),
            },
            "observer_checkpoint": args.observer_checkpoint,
            "residual_checkpoint": args.residual_checkpoint,
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

        key = _selection_key(metrics)
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
        "experiment": "C-V2 Stage 1B Utility on Frozen E1 Base r2",
        "purpose": "Isolate whether U improves the already validated E1 T+C base.",
        "base_checkpoint": args.base_checkpoint,
        "frozen": ["Host", "Motion Observer", "bounded r=2 Task-Alignment Residual", "E1 Transport/Innovation correction", "E1 Transportability Mask"],
        "training": {
            "only_trainable": "RecurrentCorrectionUtility",
            "epochs": args.epochs,
            "tbptt_steps": args.tbptt_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "history": history,
        "best": best,
        "decision_rule": (
            "Judge U directly against e1_base in the same evaluation. Keep U only if it "
            "improves the mIoU/mTC tradeoff without reopening T, C, Motion or threshold sweeps."
        ),
    }
    (result_output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"best": best, "checkpoint": str(output / "best.pt"), "result": str(result_output / "summary.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
