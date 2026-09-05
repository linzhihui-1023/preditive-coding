"""C-V2 Stage 1B: simultaneous Role-Supervision vs Spatial-Mask candidates.

This experiment reuses exactly the same frozen Host, Motion Observer and Stage
1B-2 residual-motion predictor for both candidates.  Motion is never updated.
The only difference between the two evaluated candidates is how the two semantic
correction roles are routed at inference:

1) training-only role supervision:
       L_role = L_host + DeltaL_transport + DeltaL_semantic

2) recurrent spatial mask:
       L_mask = L_host + T * DeltaL_transport + (1-T) * DeltaL_semantic

where T is transportability, not a Host/Prior fusion weight.

During training, GT + frozen RAFT define role supervision only.  The mask is
trained directly from that role target.  Final semantic CE never backpropagates
through T, so the mask cannot lower the task loss by collapsing to an all-Host
shortcut.  Motion is frozen and therefore cannot be suppressed by the mask.
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
    RESIDUAL_HIDDEN_CHANNELS,
    _load_frozen_observer,
    _observe_motion,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    MotionResidualPredictor,
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
RESIDUAL_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v2_stage1b_residual/best.pt"
)
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask"
)
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask"
CANDIDATES = ("host", "role_supervision", "spatial_mask")


def _load_frozen_residual(path, observer):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != "c_v2_stage1b2_residual_motion":
        raise RuntimeError("Residual checkpoint is not a Stage 1B-2 checkpoint")
    architecture = payload.get("architecture", {})
    max_residual_low = float(architecture.get("max_residual_low", 16.0))
    residual = MotionResidualPredictor(
        num_classes=NUM_CLASSES,
        hidden_channels=RESIDUAL_HIDDEN_CHANNELS,
        max_observed_displacement_low=observer.max_displacement_low,
        max_residual_displacement_low=max_residual_low,
    ).cuda()
    residual.load_state_dict(payload["residual_state_dict"], strict=True)
    residual.requires_grad_(False).eval()
    return residual, payload


def _warp_label_nearest(previous_gt, backward_flow):
    """Nearest-neighbor warp of previous GT using current->previous full flow."""
    height, width = previous_gt.shape
    if tuple(backward_flow.shape[-2:]) != (height, width):
        raise ValueError("GT and RAFT flow must share full-resolution shape")
    dtype = backward_flow.dtype
    device = backward_flow.device
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    source_x = x.unsqueeze(0) + backward_flow[:, 0]
    source_y = y.unsqueeze(0) + backward_flow[:, 1]
    in_bounds = (
        (source_x >= 0)
        & (source_x <= width - 1)
        & (source_y >= 0)
        & (source_y <= height - 1)
    )
    grid = torch.stack(
        (
            2.0 * source_x / max(width - 1, 1) - 1.0,
            2.0 * source_y / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    )
    sampled = F.grid_sample(
        previous_gt.to(device=device, dtype=torch.float32)[None, None],
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0].long()
    source_valid = in_bounds.squeeze(0) & (sampled != IGNORE_LABEL)
    return sampled, source_valid


def _role_masks(previous_gt, current_gt_cpu, teacher_full):
    """Training-only role target: transportable vs non-transportable semantics."""
    warped_previous_gt, source_valid = _warp_label_nearest(previous_gt, teacher_full)
    current_gt = current_gt_cpu.cuda(non_blocking=True)
    valid = current_gt != IGNORE_LABEL
    transportable = valid & source_valid & (warped_previous_gt == current_gt)
    non_transportable = valid & ~transportable
    if torch.any(transportable & non_transportable):
        raise RuntimeError("Role masks overlap")
    if not torch.equal(transportable | non_transportable, valid):
        raise RuntimeError("Role masks do not partition valid pixels")
    return valid, transportable, non_transportable


def _masked_cross_entropy(logits, target_cpu, mask_gpu):
    target = target_cpu.cuda(non_blocking=True).unsqueeze(0)
    per_pixel = F.cross_entropy(
        logits,
        target,
        ignore_index=IGNORE_LABEL,
        reduction="none",
    )[0]
    if not bool(mask_gpu.any()):
        return logits.sum() * 0.0
    return per_pixel[mask_gpu].mean()


def _masked_zero_residual(delta_logits, mask_gpu):
    expanded = mask_gpu.unsqueeze(0).unsqueeze(0).expand_as(delta_logits)
    if not bool(expanded.any()):
        return delta_logits.sum() * 0.0
    values = delta_logits[expanded]
    return F.smooth_l1_loss(values, torch.zeros_like(values))


def _balanced_transportability_bce(mask_logits_full, transportable, valid):
    """Class-balanced direct role supervision; no downstream CE gradient to mask."""
    target = transportable.float().unsqueeze(0).unsqueeze(0)
    valid4 = valid.unsqueeze(0).unsqueeze(0)
    selected_target = target[valid4]
    selected_logits = mask_logits_full[valid4]
    if selected_target.numel() == 0:
        return mask_logits_full.sum() * 0.0

    positive = selected_target.sum()
    total = torch.tensor(
        float(selected_target.numel()), device=selected_target.device, dtype=selected_target.dtype
    )
    negative = total - positive
    positive_weight = total / (2.0 * positive.clamp_min(1.0))
    negative_weight = total / (2.0 * negative.clamp_min(1.0))
    weights = torch.where(
        selected_target > 0.5,
        positive_weight,
        negative_weight,
    )
    loss = F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_target,
        reduction="none",
    )
    return (loss * weights).mean()


def _detach_hidden(hidden):
    return hidden.detach() if hidden is not None else None


def _initialize_frozen_motion(observer, residual, frame0, frame1):
    """Establish M_hat_2 from frames 0 and 1 using the frozen Stage-1B causal path."""
    _, _, low0, c10, _ = frame0
    _, _, low1, c11, _ = frame1
    with torch.no_grad():
        observed_motion = _observe_motion(observer, low0, c10, low1, c11)
        error = F.softmax(low1, dim=1) - F.softmax(low0, dim=1)
        pending_motion, pending_delta, motion_hidden = residual.predict_next(
            observed_motion, error, None
        )
    return observed_motion.detach(), pending_motion.detach(), pending_delta.detach(), motion_hidden.detach()


def _train_sequence(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    raft,
    samples,
    optimizer,
    tbptt_steps,
):
    if len(samples) < 3:
        return None

    frame0 = _host_observation(model, samples[0])
    frame1 = _host_observation(model, samples[1])
    previous_gt = semantic_mask_from_panoptic_png(samples[1]["mask_path"])
    previous_observed_motion, pending_motion, _, motion_hidden = _initialize_frozen_motion(
        observer, residual, frame0, frame1
    )
    previous_image, _, previous_low, previous_c1, _ = frame1

    transport_hidden = None
    semantic_hidden = None
    mask_hidden = None
    buffered = []
    totals = {
        "windows": 0,
        "frames": 0,
        "transport_ce": 0.0,
        "semantic_ce": 0.0,
        "transport_offrole": 0.0,
        "semantic_offrole": 0.0,
        "mask_bce": 0.0,
        "total": 0.0,
        "transport_fraction": 0.0,
        "mask_mean": 0.0,
        "delta_transport_abs": 0.0,
        "delta_semantic_abs": 0.0,
    }

    for frame_index in range(2, len(samples)):
        current_image, host_logits, host_low, current_c1, output_size = _host_observation(
            model, samples[frame_index]
        )
        current_gt = semantic_mask_from_panoptic_png(samples[frame_index]["mask_path"])

        with torch.no_grad():
            prior_low, _ = warp_low_logits(previous_low, pending_motion)
            teacher_full = raft.current_to_previous(current_image, previous_image)
            valid, transportable, non_transportable = _role_masks(
                previous_gt, current_gt, teacher_full
            )

        correction_row = correction(
            current_c1.detach(),
            host_low.detach(),
            prior_low.detach(),
            pending_motion.detach(),
            transport_hidden,
            semantic_hidden,
        )
        transport_hidden = correction_row["transport_hidden"]
        semantic_hidden = correction_row["semantic_hidden"]
        delta_transport_low = correction_row["delta_transport"]
        delta_semantic_low = correction_row["delta_semantic"]
        delta_transport_full = _upsample_prior(delta_transport_low, output_size)
        delta_semantic_full = _upsample_prior(delta_semantic_low, output_size)

        mask_logits_low, mask_hidden = mask_predictor(
            current_c1.detach(),
            host_low.detach(),
            prior_low.detach(),
            pending_motion.detach(),
            mask_hidden,
        )
        mask_logits_full = _upsample_prior(mask_logits_low, output_size)

        # Each correction head is supervised only for its physical role.
        transport_ce = _masked_cross_entropy(
            host_logits.detach() + delta_transport_full,
            current_gt,
            transportable,
        )
        semantic_ce = _masked_cross_entropy(
            host_logits.detach() + delta_semantic_full,
            current_gt,
            non_transportable,
        )
        # Off-role zero is direct role supervision, not a global sparsity penalty.
        transport_offrole = _masked_zero_residual(
            delta_transport_full, non_transportable
        )
        semantic_offrole = _masked_zero_residual(
            delta_semantic_full, transportable
        )
        mask_bce = _balanced_transportability_bce(
            mask_logits_full, transportable, valid
        )
        total = (
            transport_ce
            + semantic_ce
            + transport_offrole
            + semantic_offrole
            + mask_bce
        )
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite role-vs-mask training loss")
        buffered.append(total)

        with torch.no_grad():
            current_observed_motion = _observe_motion(
                observer, previous_low, previous_c1, host_low, current_c1
            )
            error_t = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
            next_motion, next_delta, next_motion_hidden = residual.predict_next(
                current_observed_motion,
                error_t,
                motion_hidden,
            )

        totals["frames"] += 1
        totals["transport_fraction"] += float(
            transportable.sum().item() / max(valid.sum().item(), 1)
        )
        totals["mask_mean"] += float(torch.sigmoid(mask_logits_full).mean().detach().item())
        totals["delta_transport_abs"] += float(delta_transport_full.abs().mean().detach().item())
        totals["delta_semantic_abs"] += float(delta_semantic_full.abs().mean().detach().item())
        for key, value in (
            ("transport_ce", transport_ce),
            ("semantic_ce", semantic_ce),
            ("transport_offrole", transport_offrole),
            ("semantic_offrole", semantic_offrole),
            ("mask_bce", mask_bce),
        ):
            totals[key] += float(value.detach().item())

        boundary = len(buffered) == tbptt_steps or frame_index == len(samples) - 1
        if boundary:
            window_loss = torch.stack(buffered).mean()
            optimizer.zero_grad(set_to_none=True)
            window_loss.backward()
            optimizer.step()
            totals["windows"] += 1
            totals["total"] += float(window_loss.detach().item())
            buffered = []
            transport_hidden = _detach_hidden(transport_hidden)
            semantic_hidden = _detach_hidden(semantic_hidden)
            mask_hidden = _detach_hidden(mask_hidden)

        previous_image = current_image
        previous_low = host_low.detach()
        previous_c1 = current_c1.detach()
        previous_gt = current_gt
        previous_observed_motion = current_observed_motion.detach()
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()

    frames = max(totals["frames"], 1)
    windows = max(totals["windows"], 1)
    for key in (
        "transport_ce",
        "semantic_ce",
        "transport_offrole",
        "semantic_offrole",
        "mask_bce",
        "transport_fraction",
        "mask_mean",
        "delta_transport_abs",
        "delta_semantic_abs",
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
    raft,
    groups,
    optimizer,
    tbptt_steps,
):
    correction.train()
    mask_predictor.train()
    accumulated = None
    sequence_count = 0
    for samples in groups.values():
        row = _train_sequence(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            raft,
            samples,
            optimizer,
            tbptt_steps,
        )
        if row is None:
            continue
        if accumulated is None:
            accumulated = {key: 0.0 for key in row}
        for key, value in row.items():
            accumulated[key] += float(value)
        sequence_count += 1
    if accumulated is None:
        raise RuntimeError("No valid training sequences")
    for key in accumulated:
        accumulated[key] /= max(sequence_count, 1)
    return accumulated


@torch.inference_mode()
def _evaluate(model, observer, residual, correction, mask_predictor, groups, raft):
    correction.eval()
    mask_predictor.eval()
    confusion = {
        name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
        for name in CANDIDATES
    }
    mtc_sum = {name: 0.0 for name in CANDIDATES}
    mtc_count = {name: 0 for name in CANDIDATES}
    vc_sum = {name: {8: 0.0, 16: 0.0} for name in CANDIDATES}
    vc_count = {name: {8: 0, 16: 0} for name in CANDIDATES}
    diag = {
        "frames_with_temporal_prior": 0,
        "mask_mean": 0.0,
        "delta_transport_abs": 0.0,
        "delta_semantic_abs": 0.0,
    }

    for sequence in FULL9:
        samples = groups[sequence]
        previous = None
        previous_observed_motion = None
        pending_motion = None
        motion_hidden = None
        transport_hidden = None
        semantic_hidden = None
        mask_hidden = None
        previous_predictions = {}
        seq_vc = {name: VideoConsistency() for name in CANDIDATES}

        for index, sample in enumerate(samples):
            image, host_logits, host_low, current_c1, output_size = _host_observation(
                model, sample
            )
            host_pred = host_logits.argmax(1)
            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"])
            teacher_full = None

            if previous is None:
                role_pred = spatial_pred = host_pred
                previous = (image, host_low.detach(), current_c1.detach())
            elif pending_motion is None:
                # The second frame establishes observed motion and M_hat for the
                # following frame; no future prior exists yet.
                previous_image, previous_low, previous_c1 = previous
                current_observed_motion = _observe_motion(
                    observer, previous_low, previous_c1, host_low, current_c1
                )
                error_t = F.softmax(host_low, dim=1) - F.softmax(previous_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(
                    current_observed_motion, error_t, None
                )
                previous_observed_motion = current_observed_motion.detach()
                role_pred = spatial_pred = host_pred
                previous = (image, host_low.detach(), current_c1.detach())
            else:
                previous_image, previous_low, previous_c1 = previous
                prior_low, _ = warp_low_logits(previous_low, pending_motion)
                correction_row = correction(
                    current_c1,
                    host_low,
                    prior_low,
                    pending_motion,
                    transport_hidden,
                    semantic_hidden,
                )
                transport_hidden = correction_row["transport_hidden"]
                semantic_hidden = correction_row["semantic_hidden"]
                delta_transport_full = _upsample_prior(
                    correction_row["delta_transport"], output_size
                )
                delta_semantic_full = _upsample_prior(
                    correction_row["delta_semantic"], output_size
                )
                mask_logits_low, mask_hidden = mask_predictor(
                    current_c1,
                    host_low,
                    prior_low,
                    pending_motion,
                    mask_hidden,
                )
                transportability = torch.sigmoid(
                    _upsample_prior(mask_logits_low, output_size)
                )

                role_logits = (
                    host_logits + delta_transport_full + delta_semantic_full
                )
                spatial_logits = host_logits + (
                    transportability * delta_transport_full
                    + (1.0 - transportability) * delta_semantic_full
                )
                role_pred = role_logits.argmax(1)
                spatial_pred = spatial_logits.argmax(1)

                diag["frames_with_temporal_prior"] += 1
                diag["mask_mean"] += float(transportability.mean().item())
                diag["delta_transport_abs"] += float(delta_transport_full.abs().mean().item())
                diag["delta_semantic_abs"] += float(delta_semantic_full.abs().mean().item())

                current_observed_motion = _observe_motion(
                    observer, previous_low, previous_c1, host_low, current_c1
                )
                error_t = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(
                    current_observed_motion,
                    error_t,
                    motion_hidden,
                )
                previous_observed_motion = current_observed_motion.detach()
                previous = (image, host_low.detach(), current_c1.detach())

            predictions = {
                "host": host_pred,
                "role_supervision": role_pred,
                "spatial_mask": spatial_pred,
            }
            for name, prediction in predictions.items():
                pc = prediction.squeeze(0).cpu()
                update_confusion_matrix(confusion[name], pc, gt_cpu)
                seq_vc[name].update(gt_cpu, pc)

            if index > 0:
                if teacher_full is None:
                    previous_image = samples[index - 1]
                    # mTC needs actual images; use the stored previous frame tensor.
                    stored_previous_image = previous[0] if False else None
                # Recompute teacher only for mTC when the temporal-correction path
                # did not already need it. RAFT remains evaluation support only.
                if teacher_full is None:
                    if index == 1:
                        # At this point previous has already been replaced, so obtain
                        # the preceding image directly through the frozen Host helper.
                        prev_image = _host_observation(model, samples[index - 1])[0]
                    else:
                        prev_image = _host_observation(model, samples[index - 1])[0]
                    teacher_full = raft.current_to_previous(image, prev_image)
                for name, prediction in predictions.items():
                    score = _pair_mtc(previous_predictions[name], prediction, teacher_full)
                    if math.isfinite(score):
                        mtc_sum[name] += score
                        mtc_count[name] += 1

            previous_predictions = {name: pred.detach() for name, pred in predictions.items()}

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
    frames = max(diag["frames_with_temporal_prior"], 1)
    diagnostics = {
        "frames_with_temporal_prior": diag["frames_with_temporal_prior"],
        "mask_mean": diag["mask_mean"] / frames,
        "delta_transport_abs": diag["delta_transport_abs"] / frames,
        "delta_semantic_abs": diag["delta_semantic_abs"] / frames,
    }
    return metrics, diagnostics


def _payload(
    epoch,
    args,
    correction,
    mask_predictor,
    optimizer,
    train_stats,
    metrics,
    diagnostics,
    residual_payload,
):
    return {
        "experiment": "c_v2_stage1b_role_supervision_vs_spatial_mask",
        "epoch": epoch,
        "architecture": {
            "shared_frozen_motion": True,
            "role_supervision_output": "Host + DeltaL_transport + DeltaL_semantic",
            "spatial_mask_output": "Host + T*DeltaL_transport + (1-T)*DeltaL_semantic",
            "T_definition": "transportability; not Host/Prior fusion",
            "mask_semantic_ce_gradient": False,
            "motion_gradient_from_correction": False,
            "current_C1_only_enters_semantic_and_mask_paths": True,
        },
        "training_contract": {
            "role_target": "GT + frozen RAFT current-to-previous correspondence",
            "transport_head": "CE on transportable region; zero residual on non-transportable region",
            "semantic_head": "CE on non-transportable region; zero residual on transportable region",
            "mask_head": "class-balanced direct transportability BCE only",
            "no_global_sparse_lambda": True,
            "no_Host_Prior_soft_fusion": True,
        },
        "frozen": [
            "Host",
            "Adapter",
            "Writeback",
            "Decoder",
            "Motion Observer",
            "Stage 1B-2 Residual Motion Predictor",
        ],
        "observer_checkpoint": args.observer_checkpoint,
        "residual_checkpoint": args.residual_checkpoint,
        "residual_checkpoint_epoch": residual_payload.get("epoch"),
        "train": train_stats,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "correction_state_dict": correction.state_dict(),
        "mask_state_dict": mask_predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=RESIDUAL_CHECKPOINT_DEFAULT)
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

    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = _load_frozen_observer(args.observer_checkpoint)
    residual, residual_payload = _load_frozen_residual(args.residual_checkpoint, observer)
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
    optimizer = torch.optim.AdamW(
        list(correction.parameters()) + list(mask_predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
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
    best = {
        "role_supervision": None,
        "spatial_mask": None,
    }
    for epoch in range(1, args.epochs + 1):
        train_stats = _train_epoch(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            raft,
            train_groups,
            optimizer,
            args.tbptt_steps,
        )
        metrics, diagnostics = _evaluate(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            val_groups,
            raft,
        )
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_host": {
                candidate: {
                    metric: metrics[candidate][metric] - metrics["host"][metric]
                    for metric in ("mIoU", "mTC", "mVC8", "mVC16")
                }
                for candidate in ("role_supervision", "spatial_mask")
            },
        }
        history.append(row)
        payload = _payload(
            epoch,
            args,
            correction,
            mask_predictor,
            optimizer,
            train_stats,
            metrics,
            diagnostics,
            residual_payload,
        )
        torch.save(payload, output / f"epoch_{epoch:03d}.pt")
        (result_output / f"epoch_{epoch:03d}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )

        for candidate in ("role_supervision", "spatial_mask"):
            score = metrics[candidate]["mIoU"]
            if best[candidate] is None or score > best[candidate]["mIoU"]:
                best[candidate] = {
                    "epoch": epoch,
                    "mIoU": score,
                    "metrics": metrics[candidate],
                    "delta_vs_host": row["delta_vs_host"][candidate],
                }
                torch.save(payload, output / f"best_{candidate}.pt")

        print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "experiment": "C-V2 Stage 1B Role Supervision vs Spatial Mask",
        "purpose": (
            "One shared frozen-motion run comparing implicit training-only role separation "
            "against explicit recurrent transportability routing."
        ),
        "training": {
            "epochs": args.epochs,
            "tbptt_steps": args.tbptt_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "frozen_motion": {
            "observer_checkpoint": args.observer_checkpoint,
            "residual_checkpoint": args.residual_checkpoint,
        },
        "history": history,
        "best": best,
        "decision_rule": (
            "Use the existing main metrics. Prefer a candidate only if its semantic gain is not "
            "bought by a material temporal-consistency loss. Do not create threshold sweeps or "
            "additional micro-diagnostics from this run."
        ),
    }
    (result_output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "best": best,
        "role_checkpoint": str(output / "best_role_supervision.pt"),
        "mask_checkpoint": str(output / "best_spatial_mask.pt"),
        "result": str(result_output / "summary.json"),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
