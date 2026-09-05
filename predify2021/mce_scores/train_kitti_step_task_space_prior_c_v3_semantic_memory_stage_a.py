"""C-V3 Stage A: motion-gated semantic memory + adaptive vector correction.

Stage A is an architecture decision experiment on the validated C-V2 E1 base.
The Host, Motion Observer, bounded motion residual, E1 transport/innovation
correction and E1 transportability mask are loaded and frozen. Only the new
64-D semantic-memory/refinement module is trainable.

Frozen E1 baseline:
    C_t = T_t * Warp(C_{t-1}, M_hat_t) + (1-T_t) * I_t
    L_E1 = L_host + T_t * DeltaL_transport + C_t

C-V3 Stage-A output:
    H_t = Memory(current evidence, Q_mem * Warp(H_{t-1}, M_hat_t))
    DeltaR_t = AdaptiveReadout(H_t, current evidence, C_t)
    L_out = L_E1 + DeltaR_t

The refinement head is zero-initialized, so step zero is exactly E1 Base.
Temporal supervision uses frozen RAFT only during training. The previous output
is a detached teacher and is applied only where RAFT geometry is valid, warped
GT semantics agree, frozen T supports transport, and the previous prediction
has true-class confidence. Temporal loss therefore cannot train T to collapse.
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
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_utility_from_e1_r2 import (
    BASE_CHECKPOINT_DEFAULT,
    _load_frozen_e1_base,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT, flow_grid
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    warp_low_logits,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_memory import (
    MotionGatedSemanticMemoryRefiner,
)


SEED = 0
EPOCHS = 3
TBPTT_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
GRAD_EPS = 1e-12
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v3_semantic_memory_stage_a"
)
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v3_semantic_memory_stage_a"
CANDIDATES = ("host", "e1_base", "c_v3")


@torch.no_grad()
def _frozen_e1_step(
    correction,
    mask_predictor,
    current_c1,
    host_low,
    prior_low,
    pending_motion,
    semantic_state_low,
    transport_hidden,
    semantic_hidden,
    mask_hidden,
):
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
    semantic_state_low, carried_state_low = _update_semantic_state_zero_invalid(
        semantic_state_low,
        pending_motion,
        transportability_low,
        semantic_innovation_low,
    )
    e1_delta_low = transportability_low * delta_transport_low + semantic_state_low
    return {
        "transport_hidden": transport_hidden,
        "semantic_hidden": semantic_hidden,
        "mask_hidden": mask_hidden,
        "delta_transport_low": delta_transport_low,
        "semantic_innovation_low": semantic_innovation_low,
        "transportability_low": transportability_low,
        "semantic_state_low": semantic_state_low,
        "carried_state_low": carried_state_low,
        "e1_delta_low": e1_delta_low,
    }


def _gradient_norm(loss, parameters):
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    total = None
    for grad in grads:
        if grad is None:
            continue
        value = grad.detach().float().square().sum()
        total = value if total is None else total + value
    if total is None:
        return 0.0
    return float(total.sqrt().item())


@torch.no_grad()
def _warped_previous_true_confidence(previous_logits, previous_gt_cpu, grid):
    previous_probability = F.softmax(previous_logits.float(), dim=1)
    previous_gt = previous_gt_cpu.to(previous_logits.device, non_blocking=True)
    valid_previous = previous_gt != IGNORE_LABEL
    safe_previous = previous_gt.clamp(0, NUM_CLASSES - 1)
    confidence = previous_probability.gather(
        1, safe_previous.unsqueeze(0).unsqueeze(0)
    )
    confidence = confidence * valid_previous.unsqueeze(0).unsqueeze(0).to(
        confidence.dtype
    )
    return F.grid_sample(
        confidence,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def _strict_temporal_l1(
    current_logits,
    previous_logits,
    current_gt_cpu,
    previous_gt_cpu,
    backward_flow_full,
    transportability_low,
):
    """Detached-teacher temporal L1 on strict valid semantic correspondences."""
    height, width = current_logits.shape[-2:]
    grid, flow_valid = flow_grid(backward_flow_full, height, width)

    with torch.no_grad():
        previous_probability = F.softmax(previous_logits.detach().float(), dim=1)
        warped_teacher = F.grid_sample(
            previous_probability,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).detach()

        previous_gt = previous_gt_cpu.to(current_logits.device, non_blocking=True)
        current_gt = current_gt_cpu.to(current_logits.device, non_blocking=True)
        previous_valid = previous_gt != IGNORE_LABEL
        current_valid = current_gt != IGNORE_LABEL
        warped_previous_gt = F.grid_sample(
            previous_gt.float().unsqueeze(0).unsqueeze(0),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0).squeeze(0).long()
        semantic_valid = (
            flow_valid.squeeze(0)
            & current_valid
            & (warped_previous_gt == current_gt)
        )

        previous_confidence = _warped_previous_true_confidence(
            previous_logits.detach(), previous_gt_cpu, grid
        )[:, 0]
        transportability_full = F.interpolate(
            transportability_low.detach().float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        weight = (
            semantic_valid.unsqueeze(0).float()
            * transportability_full
            * previous_confidence
        ).detach()

    current_probability = F.softmax(current_logits.float(), dim=1)
    per_pixel = (current_probability - warped_teacher).abs().mean(dim=1)
    weight_sum = weight.sum()
    if not bool(weight_sum > 0):
        return current_logits.sum() * 0.0, {
            "temporal_weight_mean": 0.0,
            "temporal_valid_fraction": 0.0,
            "previous_true_confidence_mean": 0.0,
        }
    loss = (per_pixel * weight).sum() / weight_sum.clamp_min(1e-6)
    valid_count = semantic_valid.sum().item()
    total_count = semantic_valid.numel()
    return loss, {
        "temporal_weight_mean": float(weight.mean().item()),
        "temporal_valid_fraction": float(valid_count / max(total_count, 1)),
        "previous_true_confidence_mean": float(
            previous_confidence[semantic_valid.unsqueeze(0)].mean().item()
        ) if valid_count else 0.0,
    }


@torch.no_grad()
def _zero_step_equality_check(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    groups,
):
    samples = next((value for value in groups.values() if len(value) >= 3), None)
    if samples is None:
        raise RuntimeError("No sequence with at least three frames for zero-step check")
    frame0 = _host_observation(model, samples[0])
    frame1 = _host_observation(model, samples[1])
    pending_motion, _ = _initialize_motion(observer, residual, frame0, frame1)
    _, _, previous_low, _, _ = frame1
    _, _, host_low, current_c1, _ = _host_observation(model, samples[2])
    prior_low, _ = warp_low_logits(previous_low, pending_motion)
    semantic_state_low = torch.zeros_like(previous_low)
    e1 = _frozen_e1_step(
        correction,
        mask_predictor,
        current_c1,
        host_low,
        prior_low,
        pending_motion,
        semantic_state_low,
        None,
        None,
        None,
    )
    row = refiner(
        current_c1,
        host_low,
        prior_low,
        e1["transportability_low"],
        pending_motion,
        e1["semantic_state_low"],
        None,
    )
    refinement_max = float(row["delta_refinement"].abs().max().item())
    e1_low = host_low + e1["e1_delta_low"]
    cv3_low = e1_low + row["delta_refinement"]
    equality_max = float((cv3_low - e1_low).abs().max().item())
    if refinement_max != 0.0 or equality_max != 0.0:
        raise RuntimeError(
            f"C-V3 zero-step equality failed: refinement={refinement_max}, delta={equality_max}"
        )
    return {
        "delta_refinement_abs_max": refinement_max,
        "e1_vs_cv3_abs_max": equality_max,
    }


def _train_sequence(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    raft,
    samples,
    optimizer,
    tbptt_steps,
    temporal_scale_state,
):
    if len(samples) < 3:
        return None

    frame0 = _host_observation(model, samples[0])
    frame1 = _host_observation(model, samples[1])
    pending_motion, motion_hidden = _initialize_motion(observer, residual, frame0, frame1)
    previous_image, previous_host_logits, previous_low, previous_c1, _ = frame1
    previous_gt = semantic_mask_from_panoptic_png(samples[1]["mask_path"])
    previous_output_logits = previous_host_logits.detach()

    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = None
    buffered_seg = []
    buffered_temp = []
    totals = {
        "frames": 0,
        "windows": 0,
        "segmentation_ce": 0.0,
        "temporal_l1": 0.0,
        "total": 0.0,
        "temporal_weight_mean": 0.0,
        "temporal_valid_fraction": 0.0,
        "previous_true_confidence_mean": 0.0,
        "memory_abs": 0.0,
        "warped_memory_abs": 0.0,
        "memory_reliability_mean": 0.0,
        "agreement_mean": 0.0,
        "delta_refinement_abs": 0.0,
        "semantic_state_abs": 0.0,
        "transportability_mean": 0.0,
    }

    trainable = [parameter for parameter in refiner.parameters() if parameter.requires_grad]

    for frame_index in range(2, len(samples)):
        current_image, host_logits, host_low, current_c1, output_size = _host_observation(
            model, samples[frame_index]
        )
        current_gt = semantic_mask_from_panoptic_png(samples[frame_index]["mask_path"])

        with torch.no_grad():
            prior_low, _ = warp_low_logits(previous_low, pending_motion)
            e1 = _frozen_e1_step(
                correction,
                mask_predictor,
                current_c1,
                host_low,
                prior_low,
                pending_motion,
                semantic_state_low,
                transport_hidden,
                semantic_hidden,
                mask_hidden,
            )
            transport_hidden = e1["transport_hidden"]
            semantic_hidden = e1["semantic_hidden"]
            mask_hidden = e1["mask_hidden"]
            semantic_state_low = e1["semantic_state_low"]

        memory_row = refiner(
            current_c1.detach(),
            host_low.detach(),
            prior_low.detach(),
            e1["transportability_low"].detach(),
            pending_motion.detach(),
            semantic_state_low.detach(),
            memory_state,
        )
        memory_state = memory_row["memory"]
        c_v3_delta_low = e1["e1_delta_low"].detach() + memory_row["delta_refinement"]
        current_output_logits = (
            host_logits.detach() + _upsample_prior(c_v3_delta_low, output_size)
        )
        target_gpu = current_gt.to(current_output_logits.device, non_blocking=True).unsqueeze(0)
        segmentation_ce = F.cross_entropy(
            current_output_logits,
            target_gpu,
            ignore_index=IGNORE_LABEL,
        )

        with torch.no_grad():
            teacher_full = raft.current_to_previous(current_image, previous_image)
        temporal_l1, temporal_diag = _strict_temporal_l1(
            current_output_logits,
            previous_output_logits,
            current_gt,
            previous_gt,
            teacher_full,
            e1["transportability_low"],
        )

        if not torch.isfinite(segmentation_ce) or not torch.isfinite(temporal_l1):
            raise FloatingPointError("Non-finite C-V3 Stage-A loss")
        buffered_seg.append(segmentation_ce)
        buffered_temp.append(temporal_l1)

        with torch.no_grad():
            observed_motion = _observe_motion(
                observer, previous_low, previous_c1, host_low, current_c1
            )
            prediction_error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion, prediction_error, motion_hidden
            )

        totals["frames"] += 1
        totals["segmentation_ce"] += float(segmentation_ce.detach().item())
        totals["temporal_l1"] += float(temporal_l1.detach().item())
        for key in (
            "temporal_weight_mean",
            "temporal_valid_fraction",
            "previous_true_confidence_mean",
        ):
            totals[key] += temporal_diag[key]
        totals["memory_abs"] += float(memory_state.abs().mean().detach().item())
        totals["warped_memory_abs"] += float(
            memory_row["warped_memory"].abs().mean().detach().item()
        )
        totals["memory_reliability_mean"] += float(
            memory_row["memory_reliability"].mean().detach().item()
        )
        totals["agreement_mean"] += float(
            memory_row["agreement"].mean().detach().item()
        )
        totals["delta_refinement_abs"] += float(
            memory_row["delta_refinement"].abs().mean().detach().item()
        )
        totals["semantic_state_abs"] += float(semantic_state_low.abs().mean().item())
        totals["transportability_mean"] += float(
            e1["transportability_low"].mean().item()
        )

        boundary = len(buffered_seg) == tbptt_steps or frame_index == len(samples) - 1
        if boundary:
            window_seg = torch.stack(buffered_seg).mean()
            window_temp = torch.stack(buffered_temp).mean()

            if temporal_scale_state["value"] is None and float(window_temp.detach().item()) > 0.0:
                g_seg = _gradient_norm(window_seg, trainable)
                g_temp = _gradient_norm(window_temp, trainable)
                if g_seg > GRAD_EPS and g_temp > GRAD_EPS:
                    temporal_scale_state["value"] = g_seg / g_temp
                    temporal_scale_state["seg_grad_norm"] = g_seg
                    temporal_scale_state["temp_grad_norm"] = g_temp

            lambda_temporal = temporal_scale_state["value"]
            if lambda_temporal is None:
                lambda_temporal = 0.0
            window_loss = window_seg + float(lambda_temporal) * window_temp
            optimizer.zero_grad(set_to_none=True)
            window_loss.backward()
            optimizer.step()

            totals["windows"] += 1
            totals["total"] += float(window_loss.detach().item())
            buffered_seg = []
            buffered_temp = []
            memory_state = memory_state.detach()

        previous_image = current_image
        previous_output_logits = current_output_logits.detach()
        previous_gt = current_gt
        previous_low = host_low.detach()
        previous_c1 = current_c1.detach()
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()

    frames = max(totals["frames"], 1)
    windows = max(totals["windows"], 1)
    for key in (
        "segmentation_ce",
        "temporal_l1",
        "temporal_weight_mean",
        "temporal_valid_fraction",
        "previous_true_confidence_mean",
        "memory_abs",
        "warped_memory_abs",
        "memory_reliability_mean",
        "agreement_mean",
        "delta_refinement_abs",
        "semantic_state_abs",
        "transportability_mean",
    ):
        totals[key] /= frames
    totals["total"] /= windows
    totals["lambda_temporal"] = (
        float(temporal_scale_state["value"])
        if temporal_scale_state["value"] is not None
        else 0.0
    )
    return totals


def _train_epoch(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    raft,
    groups,
    optimizer,
    tbptt_steps,
    temporal_scale_state,
):
    refiner.train()
    correction.eval()
    mask_predictor.eval()
    rows = []
    for samples in groups.values():
        row = _train_sequence(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            raft,
            samples,
            optimizer,
            tbptt_steps,
            temporal_scale_state,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("No valid training sequences")
    averaged = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }
    averaged["lambda_temporal"] = (
        float(temporal_scale_state["value"])
        if temporal_scale_state["value"] is not None
        else 0.0
    )
    return averaged


@torch.inference_mode()
def _evaluate(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    groups,
    raft,
):
    refiner.eval()
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
        "transportability_mean": 0.0,
        "semantic_state_abs": 0.0,
        "memory_abs": 0.0,
        "warped_memory_abs": 0.0,
        "memory_reliability_mean": 0.0,
        "agreement_mean": 0.0,
        "delta_refinement_abs": 0.0,
    }

    for sequence in FULL9:
        previous = None
        pending_motion = motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = None
        memory_state = None
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
                e1_pred = c_v3_pred = host_pred
                semantic_state_low = torch.zeros_like(host_low)
                previous = (image, host_low.detach(), current_c1.detach())
            elif pending_motion is None:
                previous_image, previous_low, previous_c1 = previous
                observed = _observe_motion(
                    observer, previous_low, previous_c1, host_low, current_c1
                )
                error = F.softmax(host_low, dim=1) - F.softmax(previous_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(
                    observed, error, None
                )
                e1_pred = c_v3_pred = host_pred
                previous = (image, host_low.detach(), current_c1.detach())
            else:
                previous_image, previous_low, previous_c1 = previous
                prior_low, _ = warp_low_logits(previous_low, pending_motion)
                e1 = _frozen_e1_step(
                    correction,
                    mask_predictor,
                    current_c1,
                    host_low,
                    prior_low,
                    pending_motion,
                    semantic_state_low,
                    transport_hidden,
                    semantic_hidden,
                    mask_hidden,
                )
                transport_hidden = e1["transport_hidden"]
                semantic_hidden = e1["semantic_hidden"]
                mask_hidden = e1["mask_hidden"]
                semantic_state_low = e1["semantic_state_low"]

                memory_row = refiner(
                    current_c1,
                    host_low,
                    prior_low,
                    e1["transportability_low"],
                    pending_motion,
                    semantic_state_low,
                    memory_state,
                )
                memory_state = memory_row["memory"]
                e1_delta_low = e1["e1_delta_low"]
                c_v3_delta_low = e1_delta_low + memory_row["delta_refinement"]
                e1_pred = (
                    host_logits + _upsample_prior(e1_delta_low, output_size)
                ).argmax(1)
                c_v3_pred = (
                    host_logits + _upsample_prior(c_v3_delta_low, output_size)
                ).argmax(1)

                diagnostics["frames_with_temporal_prior"] += 1
                diagnostics["transportability_mean"] += float(
                    e1["transportability_low"].mean().item()
                )
                diagnostics["semantic_state_abs"] += float(
                    semantic_state_low.abs().mean().item()
                )
                diagnostics["memory_abs"] += float(memory_state.abs().mean().item())
                diagnostics["warped_memory_abs"] += float(
                    memory_row["warped_memory"].abs().mean().item()
                )
                diagnostics["memory_reliability_mean"] += float(
                    memory_row["memory_reliability"].mean().item()
                )
                diagnostics["agreement_mean"] += float(
                    memory_row["agreement"].mean().item()
                )
                diagnostics["delta_refinement_abs"] += float(
                    memory_row["delta_refinement"].abs().mean().item()
                )

                observed = _observe_motion(
                    observer, previous_low, previous_c1, host_low, current_c1
                )
                error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(
                    observed, error, motion_hidden
                )
                previous = (image, host_low.detach(), current_c1.detach())

            predictions = {
                "host": host_pred,
                "e1_base": e1_pred,
                "c_v3": c_v3_pred,
            }
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
        "transportability_mean",
        "semantic_state_abs",
        "memory_abs",
        "warped_memory_abs",
        "memory_reliability_mean",
        "agreement_mean",
        "delta_refinement_abs",
    ):
        diagnostics[key] /= frames
    return metrics, diagnostics


def _selection_key(metrics):
    base = metrics["e1_base"]
    candidate = metrics["c_v3"]
    e1_non_degraded = candidate["mIoU"] >= base["mIoU"]
    if e1_non_degraded:
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
    correction, mask_predictor, base_payload = _load_frozen_e1_base(
        args.base_checkpoint, observer
    )
    refiner = MotionGatedSemanticMemoryRefiner(
        c1_channels=256,
        num_classes=NUM_CLASSES,
    ).cuda()

    if any(parameter.requires_grad for parameter in correction.parameters()):
        raise RuntimeError("E1 correction must remain frozen in C-V3 Stage A")
    if any(parameter.requires_grad for parameter in mask_predictor.parameters()):
        raise RuntimeError("E1 transportability mask must remain frozen in C-V3 Stage A")
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
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

    zero_step = _zero_step_equality_check(
        model,
        observer,
        residual,
        correction,
        mask_predictor,
        refiner,
        train_groups,
    )

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)
    temporal_scale_state = {
        "value": None,
        "seg_grad_norm": None,
        "temp_grad_norm": None,
    }

    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        train_stats = _train_epoch(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            raft,
            train_groups,
            optimizer,
            args.tbptt_steps,
            temporal_scale_state,
        )
        metrics, diagnostics = _evaluate(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            val_groups,
            raft,
        )
        delta_vs_host = {
            name: {
                metric: metrics[name][metric] - metrics["host"][metric]
                for metric in ("mIoU", "mTC", "mVC8", "mVC16")
            }
            for name in ("e1_base", "c_v3")
        }
        delta_vs_e1 = {
            metric: metrics["c_v3"][metric] - metrics["e1_base"][metric]
            for metric in ("mIoU", "mTC", "mVC8", "mVC16")
        }
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_host": delta_vs_host,
            "delta_c_v3_vs_e1_base": delta_vs_e1,
            "temporal_scale_calibration": temporal_scale_state.copy(),
        }
        history.append(row)

        key = _selection_key(metrics)
        payload = {
            "experiment": "c_v3_semantic_memory_stage_a",
            "epoch": epoch,
            "architecture": {
                "base": "frozen validated E1 calibrated Spatial Mask + Semantic State",
                "memory": "64-D motion-compensated reliability-gated temporal semantic evidence",
                "memory_reliability": "T * valid_warp * (1 - 0.5*L1(P_host,P_prior))",
                "current_anchor": "projected C1 + P_host + prediction_error + T",
                "readout": "multi-scale d=1/d=2 adaptive vector refinement",
                "output": "Host + T*DeltaL_transport + C_t + DeltaR_t",
                "zero_initialized_refinement": True,
            },
            "training_contract": {
                "only_trainable": "MotionGatedSemanticMemoryRefiner",
                "segmentation_loss": "full-resolution CE",
                "temporal_loss": "probability L1 to detached previous-output RAFT warp",
                "temporal_mask": "RAFT-valid * GT-semantic-consistent * stopgrad(T) * warped previous true-class confidence",
                "temporal_scale": "single first-valid-window gradient-norm calibration; no lambda sweep",
                "tbptt": args.tbptt_steps,
                "zero_step_equality": zero_step,
            },
            "frozen": [
                "Host",
                "Motion Observer",
                "bounded r=2 Task-Alignment Residual",
                "E1 Transport/Innovation correction",
                "E1 Transportability Mask",
                "E1 Semantic Correction State generator",
            ],
            "base_checkpoint": args.base_checkpoint,
            "base_checkpoint_epoch": int(base_payload.get("epoch", -1)),
            "observer_checkpoint": args.observer_checkpoint,
            "residual_checkpoint": args.residual_checkpoint,
            "residual_checkpoint_epoch": residual_payload.get("epoch"),
            "temporal_scale_calibration": temporal_scale_state.copy(),
            "refiner_state_dict": refiner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
        }
        torch.save(payload, output / f"epoch_{epoch:03d}.pt")
        (result_output / f"epoch_{epoch:03d}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )

        if best is None or key > best["selection_key"]:
            best = {
                "epoch": epoch,
                "selection_key": key,
                "metrics": metrics["c_v3"],
                "delta_vs_host": delta_vs_host["c_v3"],
                "delta_vs_e1_base": delta_vs_e1,
            }
            torch.save(payload, output / "best.pt")
        print(json.dumps(row, sort_keys=True), flush=True)

    stage_a_go = bool(
        best["delta_vs_host"]["mIoU"] >= 0.005
        and best["delta_vs_host"]["mTC"] >= 0.010
    )
    summary = {
        "experiment": "C-V3 Stage A Motion-Gated Semantic Memory",
        "purpose": (
            "Test whether high-dimensional motion-compensated semantic evidence plus "
            "adaptive vector readout breaks the C-V2 scalar/direct-correction ceiling."
        ),
        "base_checkpoint": args.base_checkpoint,
        "zero_step_equality": zero_step,
        "temporal_scale_calibration": temporal_scale_state,
        "training": {
            "only_trainable": "MotionGatedSemanticMemoryRefiner",
            "epochs": args.epochs,
            "tbptt_steps": args.tbptt_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "no_lambda_sweep": True,
            "motion_frozen": True,
            "T_frozen": True,
        },
        "history": history,
        "best": best,
        "stage_a_gate": {
            "go": stage_a_go,
            "criterion": "best Full9 delta_vs_host mIoU >= +0.5 pp AND mTC >= +1.0 pp",
            "if_no_go": "Do not proceed to Stage B motion unfreezing or parameter sweeps.",
        },
    }
    (result_output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "best": best,
                "stage_a_go": stage_a_go,
                "checkpoint": str(output / "best.pt"),
                "result": str(result_output / "summary.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
