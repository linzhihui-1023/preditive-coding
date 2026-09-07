"""C-V8 Multi-Hypothesis Prediction-Error Direct Correction.

中文：C-V8 多假设预测误差直接修正。

Frozen foundation:
  Host -> Motion -> E1 -> C-V3 -> K=4 one-warp historical hypotheses.

C-V8 removes candidate utility classification, abstention thresholds and hard
Current/history selection. Probability-space prediction errors drive a
motion-aligned recurrent Error State. Centered-logit prediction errors provide
the actual correction directions:

    e^P_k = P_current - P_history_k
    e^L_k = C(L_current) - C(L_history_k)
    DeltaL_k = -G_k * e^L_k
    L_C-V8 = L_C-V3 + sum_k DeltaL_k

Each candidate correction is upsampled independently, multiplied by that
candidate's full-resolution validity mask, and only then summed. Training
optimizes final segmentation CE plus the validated C-V3 strict RAFT temporal
loss. RAFT is absent at inference.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import load_fast_b_model
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_residual import (
    _load_frozen_observer,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask import (
    _load_frozen_residual,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask_r2 import (
    _validate_bounded_motion_checkpoint,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_utility_from_e1_r2 import (
    _load_frozen_e1_base,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v3_temporal_semantic_prediction import (
    _load_frozen_cv3_refiner,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_multihypothesis_error_correction import (
    MultiHypothesisErrorDirectCorrection,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)

from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v3_semantic_memory_stage_a as c_v3_train,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v6_error_centric_multihypothesis as c_v6,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v7_semantic_first as c_v7_protocol,
)


EXPERIMENT = "c_v8_multihypothesis_direct_error_correction"
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v8_multihypothesis_direct_error_correction"
)
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v8_multihypothesis_direct_error_correction"
DEV_MIOU_GAIN_TARGET = 0.005
CANDIDATES = (
    "host",
    "e1_base",
    "c_v3_base",
    "c_v8",
    "multiframe_semantic_oracle",
    "multiframe_semantic_temporal_oracle",
)


def _delta_metrics(candidate, reference):
    return {
        key: candidate[key] - reference[key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }


def _selection_key(metrics):
    candidate = metrics["c_v8"]
    return (candidate["mIoU"], candidate["mTC"])


def _center_logits(logits):
    """Remove the per-pixel additive logit gauge（逐像素类别均值中心化）."""
    return logits - logits.mean(dim=1, keepdim=True)


def _age_contribution(values):
    total = float(sum(values))
    if total <= 0.0:
        return [0.0 for _ in values]
    return [float(value / total) for value in values]


def _zero_step_check(module):
    """Verify zero-init direct correction exactly preserves C-V3."""
    device = next(module.parameters()).device
    n, c, h, w = 1, module.num_classes, 5, 7
    probability_errors = [
        torch.randn(n, c, h, w, device=device)
        for _ in range(module.history_length)
    ]
    correction_errors = [
        torch.randn(n, c, h, w, device=device)
        for _ in range(module.history_length)
    ]
    dynamics = torch.randn(n, c, h, w, device=device)
    current_probability = F.softmax(torch.randn(n, c, h, w, device=device), dim=1)
    current_margin = torch.rand(n, 1, h, w, device=device)
    transportability = torch.rand(n, 1, h, w, device=device)
    reliability = torch.rand(n, 1, h, w, device=device)
    validities = [
        torch.ones(n, 1, h, w, device=device)
        for _ in range(module.history_length)
    ]
    with torch.no_grad():
        row = module(
            probability_errors,
            correction_errors,
            dynamics,
            current_probability,
            current_margin,
            transportability,
            reliability,
            validities,
            None,
        )
    delta_max = float(row["delta_logits"].abs().max().item())
    if delta_max != 0.0:
        raise RuntimeError(f"C-V8 zero-step correction is not zero: {delta_max}")
    return {"delta_logits_abs_max": delta_max, "c_v8_equals_c_v3": True}


def _centered_logit_errors(
    current_low_logits,
    candidate_rows,
    history_validities,
    history_length,
):
    """Build strict-validity-gated centered-logit correction directions."""
    low_size = tuple(current_low_logits.shape[-2:])
    current_centered = _center_logits(current_low_logits.detach())
    errors = []
    for index in range(history_length):
        validity = history_validities[index].detach().to(current_low_logits.dtype)
        if index < len(candidate_rows):
            history_low_logits = F.interpolate(
                candidate_rows[index]["logits"].detach(),
                size=low_size,
                mode="bilinear",
                align_corners=False,
            )
            history_centered = _center_logits(history_low_logits)
            errors.append((current_centered - history_centered) * validity)
        else:
            errors.append(torch.zeros_like(current_centered))
    return errors


def _full_resolution_candidate_corrections(
    candidate_corrections_low,
    candidate_rows,
    history_length,
    output_size,
):
    """Upsample -> candidate-specific full validity -> sum（逐候选严格有效）."""
    full_terms = []
    for index in range(history_length):
        term = F.interpolate(
            candidate_corrections_low[:, index],
            size=tuple(output_size),
            mode="bilinear",
            align_corners=False,
        )
        if index < len(candidate_rows):
            valid_full = candidate_rows[index]["valid_full"].detach()
            if valid_full.ndim == 3:
                valid_full = valid_full.unsqueeze(1)
            term = term * valid_full.to(term.dtype)
        else:
            term = torch.zeros_like(term)
        full_terms.append(term)
    stacked = torch.stack(full_terms, dim=1)
    return stacked, stacked.sum(dim=1)


def _correction_evidence(
    module,
    dynamics,
    c_v3_logits,
    candidate_rows,
    pending_motion,
    transportability_low,
    memory_reliability_low,
    error_state,
    dynamics_state,
):
    low_size = tuple(transportability_low.shape[-2:])
    current_low_logits = F.interpolate(
        c_v3_logits.detach(),
        size=low_size,
        mode="bilinear",
        align_corners=False,
    )
    current_probability = F.softmax(current_low_logits, dim=1)
    history_probabilities, low_path_validities = c_v5._pad_history_for_controller(
        current_probability,
        candidate_rows,
        module.history_length,
    )
    history_validities = c_v6._strict_history_validities(
        current_probability,
        candidate_rows,
        low_path_validities,
        module.history_length,
    )
    multi = c_v6.build_multihypothesis_error_evidence(
        current_probability,
        history_probabilities,
        history_validities,
    )
    prediction_errors = multi["prediction_errors"]
    correction_errors = _centered_logit_errors(
        current_low_logits,
        candidate_rows,
        history_validities,
        module.history_length,
    )
    primary_error = prediction_errors[0]
    history1_valid = history_validities[0].detach()

    if dynamics_state is None:
        warped_dynamics = None
    else:
        warped_dynamics, _ = c_v6._warp_low_state_zero_invalid(
            dynamics_state,
            pending_motion,
        )
        warped_dynamics = (
            warped_dynamics
            * history1_valid
            * transportability_low.detach().clamp(0.0, 1.0)
        )
    dynamics_state = dynamics.step(primary_error, warped_dynamics).detach()

    previous_error_state = None
    if error_state is not None:
        warped_error_state, _ = c_v6._warp_low_state_zero_invalid(
            error_state,
            pending_motion,
        )
        any_valid = torch.stack(history_validities, dim=0).amax(dim=0).detach()
        history_support = (
            transportability_low.detach().clamp(0.0, 1.0)
            * memory_reliability_low.detach().clamp(0.0, 1.0)
            * any_valid
        )
        previous_error_state = warped_error_state * history_support

    row = module(
        prediction_errors,
        correction_errors,
        dynamics_state,
        current_probability,
        multi["current_margin"],
        transportability_low.detach(),
        memory_reliability_low.detach(),
        [validity.detach() for validity in history_validities],
        previous_error_state,
    )
    candidate_corrections_full, delta_full = _full_resolution_candidate_corrections(
        row["candidate_corrections_low"],
        candidate_rows,
        module.history_length,
        c_v3_logits.shape[-2:],
    )
    return {
        "row": row,
        "delta_logits_low": row["delta_logits"],
        "delta_logits_full": delta_full,
        "candidate_corrections_full": candidate_corrections_full,
        "prediction_errors": prediction_errors,
        "correction_errors": correction_errors,
        "prediction_error": primary_error,
        "dynamics_state": dynamics_state,
        "history_validities": history_validities,
    }


def _new_age_totals(history_length):
    return {
        "gain_abs_by_age": [0.0] * history_length,
        "correction_abs_by_age": [0.0] * history_length,
    }


def _add_age_frame(totals, evidence):
    gains = evidence["row"]["candidate_gains"]
    corrections = evidence["candidate_corrections_full"]
    for index in range(gains.shape[1]):
        totals["gain_abs_by_age"][index] += float(
            gains[:, index].abs().mean().detach().item()
        )
        totals["correction_abs_by_age"][index] += float(
            corrections[:, index].abs().mean().detach().item()
        )


def _finalize_age_totals(totals, frames):
    frames = max(int(frames), 1)
    gain = [value / frames for value in totals["gain_abs_by_age"]]
    correction = [value / frames for value in totals["correction_abs_by_age"]]
    return {
        "gain_abs_by_age": gain,
        "correction_abs_by_age": correction,
        "correction_contribution_by_age": _age_contribution(correction),
    }


def _train_sequence(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    module,
    dynamics,
    raft,
    samples,
    optimizer,
    tbptt_steps,
    temporal_scale_state,
):
    if len(samples) < 3:
        return None

    frame0 = c_v5._host_observation(model, samples[0])
    frame1 = c_v5._host_observation(model, samples[1])
    pending_motion, motion_hidden = c_v5._initialize_motion(
        observer,
        residual,
        frame0,
        frame1,
    )
    previous_image, previous_host_logits, previous_low, previous_c1, _ = frame1
    previous_gt = c_v5.semantic_mask_from_panoptic_png(samples[1]["mask_path"])
    previous_output_logits = previous_host_logits.detach()

    raw_history = [previous_host_logits.detach()]
    motion_history = []
    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = None
    error_state = None
    dynamics_state = None

    buffered_seg = []
    buffered_temp = []
    frames_in_window = 0
    totals = {
        "frames": 0,
        "windows": 0,
        "optimizer_steps": 0,
        "segmentation_ce": 0.0,
        "temporal_l1": 0.0,
        "total_loss": 0.0,
        "prediction_error_abs": 0.0,
        "correction_error_abs": 0.0,
        "dynamics_error_abs": 0.0,
        "error_state_abs": 0.0,
        "delta_logits_abs": 0.0,
        "any_history_valid_mean": 0.0,
        "temporal_weight_mean": 0.0,
        "temporal_valid_fraction": 0.0,
        "previous_true_confidence_mean": 0.0,
    }
    totals.update(_new_age_totals(module.history_length))
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]

    for frame_index in range(2, len(samples)):
        current_image, host_logits, host_low, current_c1, output_size = c_v5._host_observation(
            model,
            samples[frame_index],
        )
        current_gt = c_v5.semantic_mask_from_panoptic_png(samples[frame_index]["mask_path"])

        with torch.no_grad():
            prior_low, _ = c_v5.warp_low_logits(previous_low, pending_motion)
            e1 = c_v5._frozen_e1_step(
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
            memory_row, memory_state, _, c_v3_logits = c_v5._frozen_cv3_step(
                refiner,
                current_c1,
                host_low,
                prior_low,
                e1,
                pending_motion,
                memory_state,
                output_size,
                host_logits,
            )
            candidate_rows = c_v5._build_history_candidates(
                raw_history,
                motion_history,
                pending_motion,
                module.history_length,
            )

        evidence = _correction_evidence(
            module,
            dynamics,
            c_v3_logits,
            candidate_rows,
            pending_motion,
            e1["transportability_low"],
            memory_row["memory_reliability"],
            error_state,
            dynamics_state,
        )
        error_state = evidence["row"]["error_state"]
        dynamics_state = evidence["dynamics_state"]
        current_output_logits = c_v3_logits.detach() + evidence["delta_logits_full"]
        target_gpu = current_gt.to(current_output_logits.device, non_blocking=True).unsqueeze(0)
        segmentation_ce = F.cross_entropy(
            current_output_logits,
            target_gpu,
            ignore_index=c_v5.IGNORE_LABEL,
        )

        with torch.no_grad():
            teacher_full = raft.current_to_previous(current_image, previous_image)
        temporal_l1, temporal_diag = c_v3_train._strict_temporal_l1(
            current_output_logits,
            previous_output_logits,
            current_gt,
            previous_gt,
            teacher_full,
            e1["transportability_low"],
        )
        if not torch.isfinite(segmentation_ce) or not torch.isfinite(temporal_l1):
            raise FloatingPointError("Non-finite C-V8 training loss")

        buffered_seg.append(segmentation_ce)
        buffered_temp.append(temporal_l1)
        frames_in_window += 1

        with torch.no_grad():
            observed_motion = c_v5._observe_motion(
                observer,
                previous_low,
                previous_c1,
                host_low,
                current_c1,
            )
            motion_error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion,
                motion_error,
                motion_hidden,
            )

        totals["frames"] += 1
        totals["segmentation_ce"] += float(segmentation_ce.detach().item())
        totals["temporal_l1"] += float(temporal_l1.detach().item())
        totals["prediction_error_abs"] += float(
            torch.stack([error.abs().mean() for error in evidence["prediction_errors"]]).mean().detach().item()
        )
        totals["correction_error_abs"] += float(
            torch.stack([error.abs().mean() for error in evidence["correction_errors"]]).mean().detach().item()
        )
        totals["dynamics_error_abs"] += float(dynamics_state.abs().mean().item())
        totals["error_state_abs"] += float(error_state.abs().mean().detach().item())
        totals["delta_logits_abs"] += float(
            evidence["delta_logits_full"].abs().mean().detach().item()
        )
        totals["any_history_valid_mean"] += float(
            evidence["row"]["any_history_valid"].mean().item()
        )
        _add_age_frame(totals, evidence)
        for key in (
            "temporal_weight_mean",
            "temporal_valid_fraction",
            "previous_true_confidence_mean",
        ):
            totals[key] += temporal_diag[key]

        boundary = frames_in_window >= tbptt_steps or frame_index == len(samples) - 1
        if boundary:
            window_seg = torch.stack(buffered_seg).mean()
            window_temp = torch.stack(buffered_temp).mean()
            if temporal_scale_state["value"] is None and float(window_temp.detach().item()) > 0.0:
                g_seg = c_v3_train._gradient_norm(window_seg, trainable)
                g_temp = c_v3_train._gradient_norm(window_temp, trainable)
                if g_seg > c_v3_train.GRAD_EPS and g_temp > c_v3_train.GRAD_EPS:
                    temporal_scale_state["value"] = g_seg / g_temp
                    temporal_scale_state["seg_grad_norm"] = g_seg
                    temporal_scale_state["temp_grad_norm"] = g_temp
            lambda_temporal = temporal_scale_state["value"] or 0.0
            window_loss = window_seg + float(lambda_temporal) * window_temp
            optimizer.zero_grad(set_to_none=True)
            window_loss.backward()
            optimizer.step()
            totals["optimizer_steps"] += 1
            totals["windows"] += 1
            totals["total_loss"] += float(window_loss.detach().item())

            buffered_seg = []
            buffered_temp = []
            frames_in_window = 0
            if error_state is not None:
                error_state = error_state.detach()
            if memory_state is not None:
                memory_state = memory_state.detach()
            if dynamics_state is not None:
                dynamics_state = dynamics_state.detach()

        raw_history.insert(0, c_v3_logits.detach())
        raw_history = raw_history[: module.history_length]
        motion_history.insert(0, pending_motion.detach())
        motion_history = motion_history[: max(module.history_length - 1, 0)]
        previous_image = current_image
        previous_low = host_low.detach()
        previous_c1 = current_c1.detach()
        previous_gt = current_gt
        previous_output_logits = current_output_logits.detach()
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()

    frames = max(totals["frames"], 1)
    windows = max(totals["windows"], 1)
    for key in (
        "segmentation_ce",
        "temporal_l1",
        "prediction_error_abs",
        "correction_error_abs",
        "dynamics_error_abs",
        "error_state_abs",
        "delta_logits_abs",
        "any_history_valid_mean",
        "temporal_weight_mean",
        "temporal_valid_fraction",
        "previous_true_confidence_mean",
    ):
        totals[key] /= frames
    age_stats = _finalize_age_totals(totals, frames)
    totals.update(age_stats)
    totals["total_loss"] /= windows
    totals["lambda_temporal"] = temporal_scale_state["value"]
    return totals


def _train_epoch(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    module,
    dynamics,
    raft,
    groups,
    optimizer,
    tbptt_steps,
    temporal_scale_state,
):
    module.train()
    refiner.eval()
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
            module,
            dynamics,
            raft,
            samples,
            optimizer,
            tbptt_steps,
            temporal_scale_state,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("No valid C-V8 training sequences")

    frame_total = max(sum(row["frames"] for row in rows), 1)
    window_total = max(sum(row["windows"] for row in rows), 1)
    frame_keys = (
        "segmentation_ce",
        "temporal_l1",
        "prediction_error_abs",
        "correction_error_abs",
        "dynamics_error_abs",
        "error_state_abs",
        "delta_logits_abs",
        "any_history_valid_mean",
        "temporal_weight_mean",
        "temporal_valid_fraction",
        "previous_true_confidence_mean",
    )
    result = {
        "frames": sum(row["frames"] for row in rows),
        "windows": sum(row["windows"] for row in rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "total_loss": sum(row["total_loss"] * row["windows"] for row in rows) / window_total,
        "lambda_temporal": temporal_scale_state["value"],
    }
    for key in frame_keys:
        result[key] = sum(row[key] * row["frames"] for row in rows) / frame_total
    for key in ("gain_abs_by_age", "correction_abs_by_age"):
        result[key] = [
            sum(row[key][index] * row["frames"] for row in rows) / frame_total
            for index in range(module.history_length)
        ]
    result["correction_contribution_by_age"] = _age_contribution(
        result["correction_abs_by_age"]
    )
    return result


@torch.inference_mode()
def _evaluate(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    module,
    dynamics,
    groups,
    raft,
):
    module.eval()
    refiner.eval()
    correction.eval()
    mask_predictor.eval()

    confusion = {
        name: torch.zeros((c_v5.NUM_CLASSES, c_v5.NUM_CLASSES), dtype=torch.int64)
        for name in CANDIDATES
    }
    mtc_sum = {name: 0.0 for name in CANDIDATES}
    mtc_count = {name: 0 for name in CANDIDATES}
    vc_sum = {name: {8: 0.0, 16: 0.0} for name in CANDIDATES}
    vc_count = {name: {8: 0, 16: 0} for name in CANDIDATES}
    target_totals = c_v5._new_target_totals(module.history_length)
    diagnostic_frames = 0
    correction_abs = 0.0
    error_state_abs = 0.0
    prediction_error_abs = 0.0
    correction_error_abs = 0.0
    dynamics_error_abs = 0.0
    age_totals = _new_age_totals(module.history_length)

    for sequence, samples in groups.items():
        previous = None
        previous_cv3_logits = None
        pending_motion = motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = None
        memory_state = None
        error_state = None
        dynamics_state = None
        raw_history = []
        motion_history = []
        previous_predictions = {}
        seq_vc = {name: c_v5.VideoConsistency() for name in CANDIDATES}

        for sample in samples:
            image, host_logits, host_low, current_c1, output_size = c_v5._host_observation(
                model,
                sample,
            )
            host_pred = host_logits.argmax(1)
            gt_cpu = c_v5.semantic_mask_from_panoptic_png(sample["mask_path"])
            previous_image_for_mtc = previous[0] if previous is not None else None
            teacher_full = (
                raft.current_to_previous(image, previous_image_for_mtc)
                if previous_image_for_mtc is not None
                else None
            )

            if previous is None:
                e1_pred = c_v3_pred = c_v8_pred = host_pred
                semantic_oracle_pred = temporal_oracle_pred = host_pred
                semantic_state_low = torch.zeros_like(host_low)
                previous_cv3_logits = host_logits.detach()
                raw_history = [host_logits.detach()]
                previous = (image, host_low.detach(), current_c1.detach())
            elif pending_motion is None:
                previous_image, previous_low, previous_c1 = previous
                observed = c_v5._observe_motion(
                    observer,
                    previous_low,
                    previous_c1,
                    host_low,
                    current_c1,
                )
                error = F.softmax(host_low, dim=1) - F.softmax(previous_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(
                    observed,
                    error,
                    None,
                )
                e1_pred = c_v3_pred = c_v8_pred = host_pred
                semantic_oracle_pred = temporal_oracle_pred = host_pred
                previous_cv3_logits = host_logits.detach()
                raw_history = [host_logits.detach()]
                previous = (image, host_low.detach(), current_c1.detach())
            else:
                previous_image, previous_low, previous_c1 = previous
                prior_low, _ = c_v5.warp_low_logits(previous_low, pending_motion)
                e1 = c_v5._frozen_e1_step(
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
                memory_row, memory_state, e1_logits, c_v3_logits = c_v5._frozen_cv3_step(
                    refiner,
                    current_c1,
                    host_low,
                    prior_low,
                    e1,
                    pending_motion,
                    memory_state,
                    output_size,
                    host_logits,
                )
                candidate_rows = c_v5._build_history_candidates(
                    raw_history,
                    motion_history,
                    pending_motion,
                    module.history_length,
                )
                evidence = _correction_evidence(
                    module,
                    dynamics,
                    c_v3_logits,
                    candidate_rows,
                    pending_motion,
                    e1["transportability_low"],
                    memory_row["memory_reliability"],
                    error_state,
                    dynamics_state,
                )
                error_state = evidence["row"]["error_state"]
                dynamics_state = evidence["dynamics_state"]
                c_v8_logits = c_v3_logits + evidence["delta_logits_full"]

                (
                    semantic_target,
                    temporal_target,
                    _,
                    _,
                    target_diag,
                ) = c_v5._build_multiframe_target(
                    c_v3_logits,
                    candidate_rows,
                    previous_cv3_logits,
                    gt_cpu,
                    teacher_full,
                )
                c_v5._add_target_totals(target_totals, target_diag)
                current_pred, history_preds = c_v5._candidate_predictions(
                    c_v3_logits,
                    candidate_rows,
                )
                e1_pred = e1_logits.argmax(1)
                c_v3_pred = current_pred
                c_v8_pred = c_v8_logits.argmax(1)
                semantic_oracle_pred = c_v5._apply_selection(
                    current_pred,
                    history_preds,
                    semantic_target,
                )
                temporal_oracle_pred = c_v5._apply_selection(
                    current_pred,
                    history_preds,
                    temporal_target,
                )

                diagnostic_frames += 1
                correction_abs += float(evidence["delta_logits_full"].abs().mean().item())
                error_state_abs += float(error_state.abs().mean().item())
                prediction_error_abs += float(
                    torch.stack([error.abs().mean() for error in evidence["prediction_errors"]]).mean().item()
                )
                correction_error_abs += float(
                    torch.stack([error.abs().mean() for error in evidence["correction_errors"]]).mean().item()
                )
                dynamics_error_abs += float(dynamics_state.abs().mean().item())
                _add_age_frame(age_totals, evidence)

                observed = c_v5._observe_motion(
                    observer,
                    previous_low,
                    previous_c1,
                    host_low,
                    current_c1,
                )
                motion_error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
                next_motion, _, next_motion_hidden = residual.predict_next(
                    observed,
                    motion_error,
                    motion_hidden,
                )
                raw_history.insert(0, c_v3_logits.detach())
                raw_history = raw_history[: module.history_length]
                motion_history.insert(0, pending_motion.detach())
                motion_history = motion_history[: max(module.history_length - 1, 0)]
                previous_cv3_logits = c_v3_logits.detach()
                previous = (image, host_low.detach(), current_c1.detach())
                pending_motion = next_motion.detach()
                motion_hidden = next_motion_hidden.detach()

            predictions = {
                "host": host_pred,
                "e1_base": e1_pred,
                "c_v3_base": c_v3_pred,
                "c_v8": c_v8_pred,
                "multiframe_semantic_oracle": semantic_oracle_pred,
                "multiframe_semantic_temporal_oracle": temporal_oracle_pred,
            }
            for name, prediction in predictions.items():
                pred_cpu = prediction.squeeze(0).cpu()
                c_v5.update_confusion_matrix(confusion[name], pred_cpu, gt_cpu)
                seq_vc[name].update(gt_cpu, pred_cpu)

            if teacher_full is not None:
                for name, prediction in predictions.items():
                    if name in previous_predictions:
                        score = c_v5._pair_mtc(
                            previous_predictions[name],
                            prediction,
                            teacher_full,
                        )
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
            "mIoU": float(torch.nanmean(c_v5.compute_iou(confusion[name])).item()),
            "mTC": mtc_sum[name] / mtc_count[name] if mtc_count[name] else float("nan"),
            "mVC8": vc_sum[name][8] / vc_count[name][8] if vc_count[name][8] else float("nan"),
            "mVC16": vc_sum[name][16] / vc_count[name][16] if vc_count[name][16] else float("nan"),
        }
        for name in CANDIDATES
    }
    denom = max(diagnostic_frames, 1)
    age_stats = _finalize_age_totals(age_totals, denom)
    diagnostics = {
        "target_distribution": c_v5._target_rates(target_totals),
        "history_length": module.history_length,
        "history_source": "detached frozen C-V3 logits used only to form prediction errors",
        "raw_history_probability_in_correction": False,
        "raw_history_logits_in_correction_module": False,
        "hard_candidate_selection": False,
        "utility_estimator": False,
        "abstention_threshold": False,
        "correction_type": "class-wise gain along centered-logit prediction-error direction",
        "state_error_space": "probability",
        "correction_error_space": "centered logits",
        "full_resolution_validity": "candidate-specific mask after per-candidate upsampling, before summation",
        "prediction_error_reference": "strict-validity-gated t-1..t-K frozen C-V3 hypotheses",
        "error_state_motion_aligned": True,
        "error_state_history_support": "T * Q_mem * any_history_valid",
        "dynamics_error_reference": "motion-compensated strict-validity-gated t-1 probability error",
        "correction_abs": correction_abs / denom,
        "error_state_abs": error_state_abs / denom,
        "prediction_error_abs": prediction_error_abs / denom,
        "correction_error_abs": correction_error_abs / denom,
        "dynamics_error_abs": dynamics_error_abs / denom,
        "gain_abs_by_age": age_stats["gain_abs_by_age"],
        "correction_abs_by_age": age_stats["correction_abs_by_age"],
        "correction_contribution_by_age": age_stats["correction_contribution_by_age"],
        "raft_inference": False,
    }
    return metrics, diagnostics


def _resolve_protocol_groups(root, protocol):
    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(root), "val")
    if protocol == "dev3":
        train_groups = c_v7_protocol._dev3_sequence_groups(train, sequence_groups)
        val_groups = c_v7_protocol._dev3_sequence_groups(val, sequence_groups)
        definition = c_v7_protocol._protocol_definition("dev3")
    else:
        train_groups = sequence_groups(train)
        all_val_groups = sequence_groups(val)
        missing = [sequence for sequence in c_v5.FULL9 if sequence not in all_val_groups]
        if missing:
            raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
        val_groups = {sequence: all_val_groups[sequence] for sequence in c_v5.FULL9}
        definition = c_v7_protocol._protocol_definition("full9")
    return train_groups, val_groups, definition


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=c_v5.FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=c_v5.OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=c_v5.RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--base-checkpoint", default=c_v5.BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v3-checkpoint", default=c_v5.C_V3_CHECKPOINT_DEFAULT)
    parser.add_argument("--protocol", choices=("dev3", "full9"), default="dev3")
    parser.add_argument("--output", default=None)
    parser.add_argument("--result-output", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--tbptt-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=c_v5.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=c_v5.WEIGHT_DECAY)
    parser.add_argument("--dynamics-tau-e", type=float, default=c_v5.DYNAMICS_TAU_E)
    parser.add_argument("--dynamics-k-e", type=float, default=c_v5.DYNAMICS_K_E)
    parser.add_argument("--dynamics-dt", type=float, default=c_v5.DYNAMICS_DT)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--seed", type=int, default=c_v5.SEED)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.output is None:
        args.output = OUTPUT_DEFAULT + ("_dev3" if args.protocol == "dev3" else "")
    if args.result_output is None:
        args.result_output = RESULT_DEFAULT + ("_dev3" if args.protocol == "dev3" else "")

    _validate_bounded_motion_checkpoint(args.residual_checkpoint)
    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = _load_frozen_observer(args.observer_checkpoint)
    residual, residual_payload = _load_frozen_residual(args.residual_checkpoint, observer)
    correction, mask_predictor, base_payload = _load_frozen_e1_base(
        args.base_checkpoint,
        observer,
    )
    refiner, cv3_payload = _load_frozen_cv3_refiner(args.c_v3_checkpoint)

    module = MultiHypothesisErrorDirectCorrection(
        num_classes=c_v5.NUM_CLASSES,
        history_length=c_v5.HISTORY_LENGTH,
        hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
    ).cuda()
    dynamics = EulerDynamicsError(
        tau_e=args.dynamics_tau_e,
        k_e=args.dynamics_k_e,
        dt=args.dynamics_dt,
    )
    zero_step = _zero_step_check(module)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    raft = FrozenRAFT()
    train_groups, val_groups, protocol_definition = _resolve_protocol_groups(
        args.root,
        args.protocol,
    )

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    oracle_metrics, oracle_diagnostics = _evaluate(
        model,
        observer,
        residual,
        correction,
        mask_predictor,
        refiner,
        module,
        dynamics,
        val_groups,
        raft,
    )
    if oracle_metrics["c_v8"]["mIoU"] != oracle_metrics["c_v3_base"]["mIoU"]:
        raise RuntimeError("C-V8 pipeline zero-step mIoU is not exactly C-V3")
    oracle_precheck = {
        "metrics": oracle_metrics,
        "diagnostics": oracle_diagnostics,
        "delta_multiframe_oracle_vs_c_v3": _delta_metrics(
            oracle_metrics["multiframe_semantic_temporal_oracle"],
            oracle_metrics["c_v3_base"],
        ),
        "protocol": args.protocol,
        "protocol_definition": protocol_definition,
    }
    with (result_output / "oracle_precheck.json").open("w") as handle:
        json.dump(oracle_precheck, handle, indent=2)
    print(json.dumps({"oracle_precheck": oracle_precheck}, indent=2), flush=True)
    if args.oracle_only:
        return

    history = []
    best = None
    temporal_scale_state = {
        "value": None,
        "seg_grad_norm": None,
        "temp_grad_norm": None,
    }
    for epoch in range(1, args.epochs + 1):
        if args.protocol == "dev3":
            epoch_groups, sampling = c_v7_protocol._dev3_epoch_groups(
                train_groups,
                epoch - 1,
            )
        else:
            epoch_groups = train_groups
            sampling = None

        train_stats = _train_epoch(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            module,
            dynamics,
            raft,
            epoch_groups,
            optimizer,
            args.tbptt_steps,
            temporal_scale_state,
        )
        if sampling is not None:
            train_stats["dev3_sampling"] = sampling
        metrics, diagnostics = _evaluate(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            module,
            dynamics,
            val_groups,
            raft,
        )
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_c_v3_base": {
                name: _delta_metrics(metrics[name], metrics["c_v3_base"])
                for name in (
                    "c_v8",
                    "multiframe_semantic_oracle",
                    "multiframe_semantic_temporal_oracle",
                )
            },
            "selection_key": list(_selection_key(metrics)),
            "protocol": args.protocol,
            "protocol_definition": protocol_definition,
        }
        history.append(row)
        with (result_output / f"epoch_{epoch:03d}.json").open("w") as handle:
            json.dump(row, handle, indent=2)

        if best is None or tuple(row["selection_key"]) > tuple(best["selection_key"]):
            best = row
            torch.save(
                {
                    "experiment": EXPERIMENT,
                    "epoch": epoch,
                    "module_state_dict": module.state_dict(),
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "train": train_stats,
                    "dynamics": dynamics.config(),
                    "protocol": args.protocol,
                    "protocol_definition": protocol_definition,
                    "architecture": {
                        "history_length": module.history_length,
                        "history_source": "detached frozen C-V3 logits",
                        "history_semantic_interface": "prediction errors only",
                        "state_error_space": "probability",
                        "correction_error_space": "centered logits",
                        "correction_formula": "DeltaL_k = -G_k * centered_logit_error_k",
                        "full_resolution_validity": "per-candidate after upsampling, before summation",
                        "hard_candidate_selection": False,
                        "utility_estimator": False,
                        "abstention_threshold": False,
                        "error_state": "motion-aligned recurrent multi-hypothesis Error State",
                        "error_state_history_support": "T * Q_mem * any_history_valid",
                        "dynamics_error": "explicit Euler state from t-1 probability error",
                        "zero_initialized_gain_head": True,
                        "training_loss": "final segmentation CE + gradient-balanced strict temporal L1",
                        "raft_inference": False,
                    },
                },
                output / "best.pt",
            )
        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V8 produced no epoch result")

    base = best["metrics"]["c_v3_base"]
    candidate = best["metrics"]["c_v8"]
    delta = _delta_metrics(candidate, base)
    summary = {
        "experiment": "C-V8 Multi-Hypothesis Prediction-Error Direct Correction",
        "oracle_precheck": oracle_precheck,
        "best": best,
        "history": history,
        "zero_step": zero_step,
        "dynamics": dynamics.config(),
        "protocol": args.protocol,
        "protocol_definition": protocol_definition,
        "frozen_checkpoints": {
            "fast_b": args.fast_b_checkpoint,
            "observer": args.observer_checkpoint,
            "residual": args.residual_checkpoint,
            "e1_base": args.base_checkpoint,
            "c_v3_base": args.c_v3_checkpoint,
            "c_v3_checkpoint_epoch": cv3_payload.get("epoch"),
            "residual_experiment": residual_payload.get("experiment"),
            "e1_experiment": base_payload.get("experiment"),
        },
        "architecture": {
            "history_length": module.history_length,
            "history_source": "detached frozen C-V3 logits used only to form prediction errors",
            "history_logits_resampling": "one final warp per candidate",
            "controller_output_feedback": False,
            "state_prediction_error": "strict-validity-gated probability error",
            "correction_prediction_error": "strict-validity-gated centered-logit error",
            "correction_formula": "DeltaL_k = -G_k * e_logit_k; DeltaL=sum_k DeltaL_k",
            "full_resolution_validity": "candidate-specific mask after per-candidate upsampling",
            "error_state_motion_aligned": True,
            "error_state_history_support": "T * Q_mem * any_history_valid",
            "direct_correction_channels": c_v5.NUM_CLASSES,
            "hard_candidate_selection": False,
            "utility_estimator": False,
            "abstention_threshold": False,
            "training_loss": "final segmentation CE + gradient-balanced strict temporal L1",
            "raft_inference": False,
        },
        "selection_rule": "maximize validation C-V8 mIoU, then mTC",
        "development_gate": {
            "required_delta_mIoU_vs_c_v3": DEV_MIOU_GAIN_TARGET,
            "required_delta_mTC_vs_c_v3": 0.0,
            "passed": (
                delta["mIoU"] >= DEV_MIOU_GAIN_TARGET
                and delta["mTC"] >= 0.0
            ),
        },
        "delta_vs_c_v3_base": delta,
    }
    with (result_output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
