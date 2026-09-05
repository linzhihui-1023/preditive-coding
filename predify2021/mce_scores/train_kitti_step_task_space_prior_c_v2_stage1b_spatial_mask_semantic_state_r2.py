"""C-V2 Stage 1B mainline: calibrated Spatial Mask + carried Semantic Correction State.

This is the successor to spatial_mask_r2. Role-only inference remains retired.
The frozen motion base is unchanged: Stage 1B-1 Motion Observer plus the bounded
+/-2 low-resolution Task-Alignment Residual.

Two structural changes are made together because they define the new routing
semantics:

1) Transportability T is trained with ordinary BCE, not class-balanced BCE.
   Supervision is applied directly at the same low resolution where T is used
   for routing. Full-resolution binary pseudo-labels are area-aggregated into a
   soft local transportability target over valid pixels.

2) Semantic correction is no longer a free per-frame residual. The semantic
   head emits only an innovation I_t. A low-resolution correction state C_t is
   explicitly carried through time:

       C_t^warp = Warp(C_{t-1}, M_hat_t)
       C_t      = T_t * C_t^warp + (1-T_t) * I_t
       L_out    = L_host + T_t * DeltaL_transport + C_t

The final segmentation CE trains the correction heads and the carried state, but
T is detached on that path. Therefore the Mask is still trained only by its
transportability target and cannot trade calibration for single-frame mIoU.
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
    _masked_cross_entropy,
    _masked_zero_residual,
    _role_masks,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask_r2 import (
    RESIDUAL_CHECKPOINT_DEFAULT,
    _validate_bounded_motion_checkpoint,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
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
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v2_stage1b_spatial_mask_semantic_state_r2"
)
RESULT_DEFAULT = (
    "results/kitti_step_task_space_prior_c_v2_stage1b_spatial_mask_semantic_state_r2"
)
CANDIDATES = ("host", "spatial_mask")


def _transportability_bce(mask_logits_low, transportable, valid):
    """Ordinary low-resolution BCE with a valid-pixel area-aggregated soft target."""
    low_size = tuple(mask_logits_low.shape[-2:])
    transportable_full = transportable.float().unsqueeze(0).unsqueeze(0)
    valid_full = valid.float().unsqueeze(0).unsqueeze(0)

    transportable_density = F.interpolate(
        transportable_full, size=low_size, mode="area"
    )
    valid_density = F.interpolate(valid_full, size=low_size, mode="area")
    keep = valid_density > 0.0
    if not bool(keep.any()):
        return mask_logits_low.sum() * 0.0

    target_low = (
        transportable_density / valid_density.clamp_min(1e-6)
    ).clamp_(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(
        mask_logits_low[keep], target_low[keep]
    )


def _update_semantic_state(previous_state, pending_motion, transportability, innovation):
    """Warp the previous correction state and inject innovation only where needed."""
    if previous_state is None:
        previous_state = torch.zeros_like(innovation)
    carried_state, _ = warp_low_logits(previous_state, pending_motion)
    state = transportability * carried_state + (1.0 - transportability) * innovation
    return state, carried_state


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
    pending_motion, motion_hidden = _initialize_motion(observer, residual, frame0, frame1)
    previous_image, _, previous_low, previous_c1, _ = frame1
    previous_gt = semantic_mask_from_panoptic_png(samples[1]["mask_path"])

    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    buffered_losses = []
    totals = {
        "frames": 0,
        "windows": 0,
        "output_ce": 0.0,
        "transport_ce": 0.0,
        "semantic_innovation_ce": 0.0,
        "transport_offrole": 0.0,
        "semantic_offrole": 0.0,
        "mask_bce": 0.0,
        "total": 0.0,
        "transport_fraction": 0.0,
        "mask_mean": 0.0,
        "delta_transport_abs": 0.0,
        "semantic_innovation_abs": 0.0,
        "semantic_state_abs": 0.0,
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

        row = correction(
            current_c1.detach(),
            host_low.detach(),
            prior_low.detach(),
            pending_motion.detach(),
            transport_hidden,
            semantic_hidden,
        )
        transport_hidden = row["transport_hidden"]
        semantic_hidden = row["semantic_hidden"]
        delta_transport_low = row["delta_transport"]
        semantic_innovation_low = row["delta_semantic"]

        mask_logits_low, mask_hidden = mask_predictor(
            current_c1.detach(),
            host_low.detach(),
            prior_low.detach(),
            pending_motion.detach(),
            mask_hidden,
        )
        transportability_low = torch.sigmoid(mask_logits_low)

        # Role supervision is kept as a training contract for the two heads.
        delta_transport_full = _upsample_prior(delta_transport_low, output_size)
        innovation_full = _upsample_prior(semantic_innovation_low, output_size)
        transport_ce = _masked_cross_entropy(
            host_logits.detach() + delta_transport_full,
            current_gt,
            transportable,
        )
        semantic_innovation_ce = _masked_cross_entropy(
            host_logits.detach() + innovation_full,
            current_gt,
            non_transportable,
        )
        transport_offrole = _masked_zero_residual(
            delta_transport_full, non_transportable
        )
        semantic_offrole = _masked_zero_residual(
            innovation_full, transportable
        )
        mask_bce = _transportability_bce(
            mask_logits_low, transportable, valid
        )

        # Final task CE may train correction/state parameters, but not the Mask.
        route = transportability_low.detach()
        semantic_state_low, _ = _update_semantic_state(
            semantic_state_low,
            pending_motion.detach(),
            route,
            semantic_innovation_low,
        )
        output_delta_low = route * delta_transport_low + semantic_state_low
        output_logits = host_logits.detach() + _upsample_prior(output_delta_low, output_size)
        target_gpu = current_gt.cuda(non_blocking=True).unsqueeze(0)
        output_ce = F.cross_entropy(
            output_logits,
            target_gpu,
            ignore_index=IGNORE_LABEL,
        )

        total = (
            output_ce
            + transport_ce
            + semantic_innovation_ce
            + transport_offrole
            + semantic_offrole
            + mask_bce
        )
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite calibrated-mask semantic-state loss")
        buffered_losses.append(total)

        with torch.no_grad():
            observed_motion = _observe_motion(
                observer, previous_low, previous_c1, host_low, current_c1
            )
            prediction_error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion, prediction_error, motion_hidden
            )

        totals["frames"] += 1
        totals["transport_fraction"] += float(
            transportable.sum().item() / max(valid.sum().item(), 1)
        )
        totals["mask_mean"] += float(transportability_low.mean().detach().item())
        totals["delta_transport_abs"] += float(delta_transport_low.abs().mean().detach().item())
        totals["semantic_innovation_abs"] += float(
            semantic_innovation_low.abs().mean().detach().item()
        )
        totals["semantic_state_abs"] += float(semantic_state_low.abs().mean().detach().item())
        for key, value in (
            ("output_ce", output_ce),
            ("transport_ce", transport_ce),
            ("semantic_innovation_ce", semantic_innovation_ce),
            ("transport_offrole", transport_offrole),
            ("semantic_offrole", semantic_offrole),
            ("mask_bce", mask_bce),
        ):
            totals[key] += float(value.detach().item())

        boundary = len(buffered_losses) == tbptt_steps or frame_index == len(samples) - 1
        if boundary:
            window_loss = torch.stack(buffered_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            window_loss.backward()
            optimizer.step()
            totals["windows"] += 1
            totals["total"] += float(window_loss.detach().item())
            buffered_losses = []
            transport_hidden = transport_hidden.detach()
            semantic_hidden = semantic_hidden.detach()
            mask_hidden = mask_hidden.detach()
            semantic_state_low = semantic_state_low.detach()

        previous_image = current_image
        previous_low = host_low.detach()
        previous_c1 = current_c1.detach()
        previous_gt = current_gt
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()

    frames = max(totals["frames"], 1)
    windows = max(totals["windows"], 1)
    for key in (
        "output_ce",
        "transport_ce",
        "semantic_innovation_ce",
        "transport_offrole",
        "semantic_offrole",
        "mask_bce",
        "transport_fraction",
        "mask_mean",
        "delta_transport_abs",
        "semantic_innovation_abs",
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
    raft,
    groups,
    optimizer,
    tbptt_steps,
):
    correction.train()
    mask_predictor.train()
    rows = []
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
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("No valid training sequences")
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }


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
    diagnostics = {
        "frames_with_temporal_prior": 0,
        "mask_mean": 0.0,
        "delta_transport_abs": 0.0,
        "semantic_innovation_abs": 0.0,
        "semantic_state_abs": 0.0,
    }

    for sequence in FULL9:
        previous = None
        pending_motion = motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = None
        previous_predictions = {}
        seq_vc = {name: VideoConsistency() for name in CANDIDATES}

        for sample in groups[sequence]:
            image, host_logits, host_low, current_c1, output_size = _host_observation(
                model, sample
            )
            host_pred = host_logits.argmax(1)
            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"])
            previous_image_for_mtc = previous[0] if previous is not None else None

            if previous is None:
                spatial_pred = host_pred
                semantic_state_low = torch.zeros_like(host_low)
                previous = (image, host_low.detach(), current_c1.detach())
            elif pending_motion is None:
                previous_image, previous_low, previous_c1 = previous
                observed = _observe_motion(
                    observer, previous_low, previous_c1, host_low, current_c1
                )
                error = F.softmax(host_low, dim=1) - F.softmax(previous_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(observed, error, None)
                spatial_pred = host_pred
                previous = (image, host_low.detach(), current_c1.detach())
            else:
                previous_image, previous_low, previous_c1 = previous
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
                semantic_state_low, _ = _update_semantic_state(
                    semantic_state_low,
                    pending_motion,
                    transportability_low,
                    semantic_innovation_low,
                )
                output_delta_low = (
                    transportability_low * delta_transport_low + semantic_state_low
                )
                spatial_pred = (
                    host_logits + _upsample_prior(output_delta_low, output_size)
                ).argmax(1)

                diagnostics["frames_with_temporal_prior"] += 1
                diagnostics["mask_mean"] += float(transportability_low.mean().item())
                diagnostics["delta_transport_abs"] += float(delta_transport_low.abs().mean().item())
                diagnostics["semantic_innovation_abs"] += float(
                    semantic_innovation_low.abs().mean().item()
                )
                diagnostics["semantic_state_abs"] += float(semantic_state_low.abs().mean().item())

                observed = _observe_motion(
                    observer, previous_low, previous_c1, host_low, current_c1
                )
                error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(
                    observed, error, motion_hidden
                )
                previous = (image, host_low.detach(), current_c1.detach())

            predictions = {"host": host_pred, "spatial_mask": spatial_pred}
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

            previous_predictions = {
                name: prediction.detach() for name, prediction in predictions.items()
            }

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
    for key in (
        "mask_mean",
        "delta_transport_abs",
        "semantic_innovation_abs",
        "semantic_state_abs",
    ):
        diagnostics[key] /= frames
    return metrics, diagnostics


def _selection_key(metrics):
    host = metrics["host"]
    spatial = metrics["spatial_mask"]
    semantic_non_degraded = spatial["mIoU"] >= host["mIoU"]
    if semantic_non_degraded:
        return (1, spatial["mTC"], spatial["mIoU"])
    return (0, spatial["mIoU"], spatial["mTC"])


def _checkpoint_payload(
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
        "experiment": "c_v2_stage1b_spatial_mask_semantic_state_r2",
        "epoch": epoch,
        "architecture": {
            "inference_route": "spatial_mask_with_carried_semantic_state",
            "output": "Host + T*DeltaL_transport + C_t",
            "semantic_state": "C_t = T*Warp(C_t-1, M_hat_t) + (1-T)*I_t",
            "semantic_head_role": "innovation I_t only",
            "T_definition": "low-resolution local transportability probability over valid pixels",
            "mask_loss": "ordinary BCE on low-resolution area-aggregated soft target; no class balancing",
            "role_only_inference_retired": True,
            "shared_frozen_motion": True,
            "mask_task_gradient": False,
            "motion_gradient_from_correction": False,
        },
        "training_contract": {
            "role_target": "GT + frozen RAFT current-to-previous correspondence",
            "transport_head": "CE on transportable; zero residual off-role",
            "semantic_innovation_head": "CE on non-transportable; zero residual off-role",
            "mask_head": "ordinary low-resolution BCE transportability only",
            "final_output_ce": "trains correction heads and carried state with T detached",
            "tbptt_semantic_state": "carried across sequence; detached every TBPTT window, never reset within sequence",
        },
        "frozen": [
            "Host",
            "Adapter",
            "Writeback",
            "Decoder",
            "Motion Observer",
            "bounded r=2 Task-Alignment Residual",
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

    _validate_bounded_motion_checkpoint(args.residual_checkpoint)
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
    best = None
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
        delta_vs_host = {
            metric: metrics["spatial_mask"][metric] - metrics["host"][metric]
            for metric in ("mIoU", "mTC", "mVC8", "mVC16")
        }
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_host": delta_vs_host,
        }
        history.append(row)
        payload = _checkpoint_payload(
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

        key = _selection_key(metrics)
        if best is None or key > best["selection_key"]:
            best = {
                "epoch": epoch,
                "selection_key": key,
                "metrics": metrics["spatial_mask"],
                "delta_vs_host": delta_vs_host,
            }
            torch.save(payload, output / "best.pt")
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "experiment": "C-V2 Stage 1B Calibrated Spatial Mask + Semantic Correction State r2",
        "purpose": (
            "Keep Spatial Mask mainline, restore probability meaning of T, and make "
            "Semantic Correction a transported state rather than a free per-frame residual."
        ),
        "training": {
            "epochs": args.epochs,
            "tbptt_steps": args.tbptt_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "mask_loss": "ordinary low-resolution BCE",
            "role_supervision_retained": True,
            "final_output_ce_mask_gradient": False,
        },
        "frozen_motion": {
            "observer_checkpoint": args.observer_checkpoint,
            "residual_checkpoint": args.residual_checkpoint,
        },
        "history": history,
        "best": best,
        "selection_rule": (
            "If Spatial Mask mIoU >= Host, select highest mTC; otherwise select highest mIoU. "
            "Uses existing main metrics only."
        ),
        "decision_scope": (
            "Use this run to decide whether calibrated routing plus carried semantic state "
            "prevents the E1-to-E3 temporal degradation. Do not open threshold/loss sweeps."
        ),
    }
    (result_output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "best": best,
                "checkpoint": str(output / "best.pt"),
                "result": str(result_output / "summary.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
