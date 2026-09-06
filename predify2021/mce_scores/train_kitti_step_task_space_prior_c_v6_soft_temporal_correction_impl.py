"""Corrected C-V6 implementation.

C-V6: Set-based Deep Semantic Evidence + Soft Stateful Correction
中文：集合式深历史语义证据 + 软有状态时序修正。

This implementation fixes the code-review blockers in the first C-V6 draft:
1. History transport is strictly probability-space:
      softmax(raw frozen C-V3 logits) -> one final warp -> renormalize.
2. Protection supervision is applied only when Current is GT-correct and no
   valid history candidate is GT-correct. A single bad history no longer closes
   the soft gate when other history candidates are correct.
3. History attention is directly supervised wherever at least one valid history
   candidate is GT-correct, not only on Current-wrong rescue pixels.
4. Persistence propagates only a correction that the previous C-V6 output
   actually made GT-correct; an unsuccessful previous correction is never used
   as the persistence teacher.
5. Gate initialization is numerically C-V3-equivalent (bias=-20), with a
   pre-training hard-prediction equivalence check on Dev3.
6. At TBPTT boundaries the trainable correction state is recomputed once with
   updated weights before being carried into the next window.

K=4 remains fixed. No K sweep, loss-weight sweep, recursive semantic state, or
C-V6-output feedback is introduced.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

import predify2021.mce_scores.train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as v5
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_soft_temporal_correction import (
    SetBasedSoftTemporalCorrector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)


SEED = 0
EPOCHS = 3
TBPTT_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
HISTORY_LENGTH = 4
ATTENTION_HIDDEN_CHANNELS = 16
CORRECTION_HIDDEN_CHANNELS = 32
GATE_INIT_BIAS = -20.0

DYNAMICS_TAU_E = 4.0
DYNAMICS_K_E = 1.0
DYNAMICS_DT = 1.0

LOSS_SEG_WEIGHT = 1.0
LOSS_ATTN_WEIGHT = 1.0
LOSS_GATE_WEIGHT = 1.0
LOSS_PERSIST_WEIGHT = 1.0

CV4_E2_MIOU_REFERENCE = 0.6637739071008685
CV4_E3_MTC_REFERENCE = 0.712733
MTC_MILESTONE = 0.72
MTC_TARGET_LOW = 0.74
MTC_TARGET_HIGH = 0.76
ZERO_STEP_SEQUENCES = ("0002", "0010", "0018")

OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v6_soft_temporal_correction"
)
RESULT_DEFAULT = (
    "results/kitti_step_task_space_prior_c_v6_soft_temporal_correction"
)
CANDIDATES = ("host", "e1_base", "c_v3_base", "c_v6")


def _renormalize_probability(probability, validity=None):
    probability = probability.float().clamp_min(0.0)
    if validity is not None:
        probability = probability * validity.to(probability.dtype)
    denominator = probability.sum(dim=1, keepdim=True)
    normalized = probability / denominator.clamp_min(1.0e-8)
    if validity is not None:
        normalized = normalized * validity.to(normalized.dtype)
    return normalized


def _resize_probability(probability, size):
    if tuple(probability.shape[-2:]) == tuple(size):
        return _renormalize_probability(probability)
    resized = F.interpolate(
        probability.float(),
        size=tuple(size),
        mode="bilinear",
        align_corners=False,
    )
    return _renormalize_probability(resized)


def _warp_full_probability_zero_invalid(raw_logits, backward_flow_low):
    """Softmax before the one semantic warp; never softmax warped logits."""
    raw_probability = F.softmax(raw_logits.detach().float(), dim=1)
    full_size = tuple(raw_probability.shape[-2:])
    flow_full = v5._upsample_predicted_flow(backward_flow_low, full_size)
    grid, valid = v5.flow_grid(flow_full, full_size[0], full_size[1])
    warped = F.grid_sample(
        raw_probability,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    valid_mask = valid.bool().unsqueeze(1)
    warped = _renormalize_probability(warped, valid_mask)
    return warped, valid.bool()


def _build_probability_history_candidates(
    raw_history,
    motion_history,
    pending_motion,
    history_length,
):
    """Build K independent frozen C-V3 probability candidates with one warp each."""
    rows = []
    available = min(len(raw_history), history_length)
    for age in range(1, available + 1):
        if age > 1 and len(motion_history) < age - 1:
            break
        accumulated_flow, accumulated_valid_low = v5._accumulated_flow_for_age(
            pending_motion,
            motion_history,
            age,
        )
        probability, valid_full = _warp_full_probability_zero_invalid(
            raw_history[age - 1],
            accumulated_flow,
        )
        low_path_full = F.interpolate(
            accumulated_valid_low.float().unsqueeze(1),
            size=tuple(valid_full.shape[-2:]),
            mode="nearest",
        )[:, 0] > 0.5
        valid_full = valid_full & low_path_full
        probability = _renormalize_probability(
            probability,
            valid_full.unsqueeze(1),
        )
        rows.append(
            {
                "age": age,
                "probability": probability,
                "valid_full": valid_full,
                "valid_low_path": accumulated_valid_low.bool(),
                "flow": accumulated_flow,
            }
        )
    return rows


def _pad_history_low(current_probability, candidate_rows, history_length):
    probabilities = []
    validities = []
    low_size = tuple(current_probability.shape[-2:])
    for index in range(history_length):
        if index < len(candidate_rows):
            row = candidate_rows[index]
            probability = _resize_probability(row["probability"], low_size)
            validity = row["valid_low_path"].float().unsqueeze(1)
            if tuple(validity.shape[-2:]) != low_size:
                validity = F.interpolate(validity, size=low_size, mode="nearest")
            validity = validity.to(current_probability.dtype)
            probability = _renormalize_probability(probability, validity)
            probabilities.append(probability)
            validities.append(validity)
        else:
            probabilities.append(current_probability)
            validities.append(torch.zeros_like(current_probability[:, :1]))
    return probabilities, validities


def _low_correction_evidence(
    corrector,
    dynamics,
    c_v3_logits,
    candidate_rows,
    pending_motion,
    transportability_low,
    memory_reliability_low,
    correction_hidden,
    dynamics_state,
):
    low_size = tuple(transportability_low.shape[-2:])
    current_full_probability = F.softmax(c_v3_logits.detach().float(), dim=1)
    current_probability = _resize_probability(current_full_probability, low_size)
    history_probabilities, history_validities = _pad_history_low(
        current_probability,
        candidate_rows,
        corrector.history_length,
    )

    history1_probability = history_probabilities[0]
    history1_validity = history_validities[0]
    prediction_error = (
        current_probability - history1_probability
    ) * history1_validity

    if dynamics_state is None:
        warped_dynamics = None
    else:
        warped_dynamics, _ = v5._warp_low_state_zero_invalid(
            dynamics_state,
            pending_motion,
        )
        warped_dynamics = (
            warped_dynamics
            * history1_validity
            * transportability_low.detach().clamp(0.0, 1.0)
        )

    next_dynamics_state = dynamics.step(
        prediction_error,
        warped_dynamics,
    ).detach()

    row = corrector(
        current_probability.detach(),
        [probability.detach() for probability in history_probabilities],
        [validity.detach() for validity in history_validities],
        prediction_error.detach(),
        next_dynamics_state.detach(),
        transportability_low.detach(),
        memory_reliability_low.detach(),
        correction_hidden,
    )
    return {
        "row": row,
        "prediction_error": prediction_error,
        "dynamics_state": next_dynamics_state,
    }


def _full_resolution_soft_output(c_v3_logits, candidate_rows, corrector_row):
    full_size = tuple(c_v3_logits.shape[-2:])
    current_probability = F.softmax(c_v3_logits.detach().float(), dim=1)

    score_full = F.interpolate(
        corrector_row["history_scores"],
        size=full_size,
        mode="bilinear",
        align_corners=False,
    )
    full_validities = []
    history_probabilities = []
    for index in range(corrector_row["history_scores"].shape[1]):
        if index < len(candidate_rows):
            row = candidate_rows[index]
            validity = row["valid_full"].bool().unsqueeze(1)
            full_validities.append(validity)
            history_probabilities.append(row["probability"].detach().float())
        else:
            full_validities.append(
                torch.zeros_like(current_probability[:, :1], dtype=torch.bool)
            )
            history_probabilities.append(current_probability)

    valid_tensor = torch.cat(full_validities, dim=1)
    masked_scores = score_full.masked_fill(~valid_tensor, -1.0e4)
    attention = torch.softmax(masked_scores, dim=1) * valid_tensor.to(score_full.dtype)
    attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

    history_probability = torch.zeros_like(current_probability)
    for index, probability in enumerate(history_probabilities):
        history_probability = (
            history_probability + attention[:, index : index + 1] * probability
        )
    history_available = valid_tensor.any(dim=1, keepdim=True)
    history_probability = _renormalize_probability(
        history_probability,
        history_available,
    )

    lambda_logit = F.interpolate(
        corrector_row["lambda_logit"],
        size=full_size,
        mode="bilinear",
        align_corners=False,
    )
    lambda_probability = (
        torch.sigmoid(lambda_logit) * history_available.to(lambda_logit.dtype)
    )
    output_probability = (
        (1.0 - lambda_probability) * current_probability
        + lambda_probability * history_probability
    )
    output_probability = _renormalize_probability(output_probability)
    return {
        "current_probability": current_probability,
        "history_probability": history_probability,
        "output_probability": output_probability,
        "attention": attention,
        "lambda_logit": lambda_logit,
        "lambda_probability": lambda_probability,
        "history_validities": valid_tensor,
    }


def _semantic_targets(full, candidate_rows, gt_cpu):
    gt = gt_cpu.to(full["current_probability"].device, non_blocking=True)
    valid_gt = gt != v5.IGNORE_LABEL
    current_pred = full["current_probability"].argmax(dim=1)[0]

    candidate_correct = []
    candidate_available = []
    for row in candidate_rows:
        pred = row["probability"].argmax(dim=1)[0]
        valid = row["valid_full"][0].bool() & valid_gt
        candidate_correct.append(valid & (pred == gt))
        candidate_available.append(valid)
    while len(candidate_correct) < HISTORY_LENGTH:
        candidate_correct.append(torch.zeros_like(valid_gt))
        candidate_available.append(torch.zeros_like(valid_gt))

    correct_tensor = torch.stack(candidate_correct, dim=0)
    available_tensor = torch.stack(candidate_available, dim=0)
    any_history_correct = correct_tensor.any(dim=0)
    any_history_available = available_tensor.any(dim=0)

    current_correct = valid_gt & (current_pred == gt)
    rescue = valid_gt & ~current_correct & any_history_correct
    protection = current_correct & any_history_available & ~any_history_correct
    gate_supervised = rescue | protection
    gate_target = rescue.float()

    attention_supervised = valid_gt & any_history_correct
    attention_target = correct_tensor.float()
    attention_target = attention_target / attention_target.sum(
        dim=0,
        keepdim=True,
    ).clamp_min(1.0)

    return {
        "gt": gt,
        "valid_gt": valid_gt,
        "current_pred": current_pred,
        "current_correct": current_correct,
        "rescue": rescue,
        "protection": protection,
        "gate_supervised": gate_supervised,
        "gate_target": gate_target,
        "attention_supervised": attention_supervised,
        "attention_target": attention_target,
        "any_history_correct": any_history_correct,
        "any_history_available": any_history_available,
    }


def _segmentation_loss(output_probability, gt):
    return F.nll_loss(
        output_probability.clamp_min(1.0e-8).log(),
        gt.unsqueeze(0),
        ignore_index=v5.IGNORE_LABEL,
        reduction="mean",
    )


def _attention_loss(attention, attention_target, supervised):
    if not bool(supervised.any()):
        return attention.sum() * 0.0
    log_attention = attention.clamp_min(1.0e-8).log()[0]
    per_pixel = -(attention_target * log_attention).sum(dim=0)
    return per_pixel[supervised].mean()


def _balanced_gate_loss(lambda_logit, gate_target, gate_supervised):
    if not bool(gate_supervised.any()):
        return lambda_logit.sum() * 0.0
    logits = lambda_logit[0, 0][gate_supervised]
    target = gate_target[gate_supervised]
    positive = target.sum()
    total = target.new_tensor(float(target.numel()))
    negative = total - positive

    if positive.item() <= 0.0 or negative.item() <= 0.0:
        return F.binary_cross_entropy_with_logits(logits, target)

    positive_weight = total / (2.0 * positive)
    negative_weight = total / (2.0 * negative)
    weights = torch.where(target > 0.5, positive_weight, negative_weight)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weights).mean()


def _warp_boolean_mask(mask, grid):
    return F.grid_sample(
        mask.float()[None, None],
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0] > 0.5


def _strict_persistence_loss(
    current_correction,
    previous_correction,
    current_gt_cpu,
    previous_gt_cpu,
    current_rescue,
    previous_rescue,
    previous_output_correct,
    teacher_full,
):
    if (
        previous_correction is None
        or previous_gt_cpu is None
        or previous_rescue is None
        or previous_output_correct is None
        or teacher_full is None
    ):
        return current_correction.sum() * 0.0, 0

    height, width = current_correction.shape[-2:]
    grid, flow_valid = v5.flow_grid(teacher_full, height, width)
    warped_previous_correction = F.grid_sample(
        previous_correction.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    warped_previous_gt = F.grid_sample(
        previous_gt_cpu.to(
            device=current_correction.device,
            dtype=torch.float32,
        )[None, None],
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0].long()
    warped_previous_rescue = _warp_boolean_mask(
        previous_rescue.to(current_correction.device),
        grid,
    )
    warped_previous_output_correct = _warp_boolean_mask(
        previous_output_correct.to(current_correction.device),
        grid,
    )

    current_gt = current_gt_cpu.to(current_correction.device, non_blocking=True)
    stable_semantic = (
        flow_valid[0]
        & (current_gt != v5.IGNORE_LABEL)
        & (warped_previous_gt == current_gt)
    )
    persist_mask = (
        stable_semantic
        & current_rescue
        & warped_previous_rescue
        & warped_previous_output_correct
    )
    count = int(persist_mask.sum().item())
    if count == 0:
        return current_correction.sum() * 0.0, 0

    difference = (
        current_correction.float() - warped_previous_correction
    ).abs().mean(dim=1)[0]
    return difference[persist_mask].mean(), count


def _train_sequence(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    corrector,
    dynamics,
    raft,
    samples,
    optimizer,
    tbptt_steps,
):
    if len(samples) < 4:
        return None

    frame0 = v5._host_observation(model, samples[0])
    frame1 = v5._host_observation(model, samples[1])
    pending_motion, motion_hidden = v5._initialize_motion(
        observer,
        residual,
        frame0,
        frame1,
    )
    previous_image, _, previous_low, previous_c1, _ = frame1

    raw_history = []
    motion_history = []
    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = None
    correction_hidden = None
    dynamics_state = None

    previous_correction = None
    previous_rescue = None
    previous_output_correct = None
    previous_gt = None

    buffered_losses = []
    frames_in_window = 0
    totals = {
        "frames": 0,
        "supervised_frames": 0,
        "optimizer_steps": 0,
        "seg_loss": 0.0,
        "attention_loss": 0.0,
        "gate_loss": 0.0,
        "persistence_loss": 0.0,
        "total_loss": 0.0,
        "attention_supervised_pixels": 0,
        "rescue_pixels": 0,
        "protection_pixels": 0,
        "gate_supervised_pixels": 0,
        "persistence_pixels": 0,
        "lambda_mean": 0.0,
        "lambda_on_rescue": 0.0,
        "lambda_on_protection": 0.0,
        "lambda_rescue_frames": 0,
        "lambda_protection_frames": 0,
        "prediction_error_abs": 0.0,
        "dynamics_error_abs": 0.0,
        "attention_mass_by_age": [0.0] * HISTORY_LENGTH,
        "attention_age_frames": [0] * HISTORY_LENGTH,
    }

    for frame_index in range(2, len(samples)):
        current_image, host_logits, host_low, current_c1, output_size = v5._host_observation(
            model,
            samples[frame_index],
        )
        current_gt = v5.semantic_mask_from_panoptic_png(
            samples[frame_index]["mask_path"]
        )

        with torch.no_grad():
            prior_low, _ = v5.warp_low_logits(previous_low, pending_motion)
            e1 = v5._frozen_e1_step(
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

            memory_row, memory_state, _, c_v3_logits = v5._frozen_cv3_step(
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
            observed_motion = v5._observe_motion(
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

        if not raw_history:
            raw_history = [c_v3_logits.detach()]
            previous_correction = torch.zeros_like(
                F.softmax(c_v3_logits.detach().float(), dim=1)
            )
            previous_rescue = torch.zeros_like(current_gt, dtype=torch.bool)
            previous_output_correct = (
                c_v3_logits.argmax(1)[0].detach().cpu() == current_gt
            ) & (current_gt != v5.IGNORE_LABEL)
            previous_gt = current_gt
            previous_image = current_image
            previous_low = host_low.detach()
            previous_c1 = current_c1.detach()
            pending_motion = next_motion.detach()
            motion_hidden = next_motion_hidden.detach()
            totals["frames"] += 1
            continue

        with torch.no_grad():
            candidate_rows = _build_probability_history_candidates(
                raw_history,
                motion_history,
                pending_motion,
                HISTORY_LENGTH,
            )

        correction_hidden_before = correction_hidden
        dynamics_state_before = dynamics_state
        evidence = _low_correction_evidence(
            corrector,
            dynamics,
            c_v3_logits,
            candidate_rows,
            pending_motion,
            e1["transportability_low"],
            memory_row["memory_reliability"],
            correction_hidden_before,
            dynamics_state_before,
        )
        correction_hidden = evidence["row"]["correction_hidden"]
        dynamics_state = evidence["dynamics_state"]
        full = _full_resolution_soft_output(
            c_v3_logits,
            candidate_rows,
            evidence["row"],
        )

        with torch.no_grad():
            targets = _semantic_targets(full, candidate_rows, current_gt)
            teacher_full = raft.current_to_previous(current_image, previous_image)

        seg_loss = _segmentation_loss(full["output_probability"], targets["gt"])
        attention_loss = _attention_loss(
            full["attention"],
            targets["attention_target"],
            targets["attention_supervised"],
        )
        gate_loss = _balanced_gate_loss(
            full["lambda_logit"],
            targets["gate_target"],
            targets["gate_supervised"],
        )
        current_correction = (
            full["output_probability"] - full["current_probability"]
        )
        persistence_loss, persistence_pixels = _strict_persistence_loss(
            current_correction,
            previous_correction,
            current_gt,
            previous_gt,
            targets["rescue"],
            previous_rescue,
            previous_output_correct,
            teacher_full,
        )
        total_loss = (
            LOSS_SEG_WEIGHT * seg_loss
            + LOSS_ATTN_WEIGHT * attention_loss
            + LOSS_GATE_WEIGHT * gate_loss
            + LOSS_PERSIST_WEIGHT * persistence_loss
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite C-V6 training loss")
        buffered_losses.append(total_loss)

        with torch.no_grad():
            current_output_correct = (
                full["output_probability"].argmax(1)[0] == targets["gt"]
            ) & targets["valid_gt"]
            rescue_count = int(targets["rescue"].sum().item())
            protection_count = int(targets["protection"].sum().item())
            totals["frames"] += 1
            totals["supervised_frames"] += 1
            totals["seg_loss"] += float(seg_loss.detach().item())
            totals["attention_loss"] += float(attention_loss.detach().item())
            totals["gate_loss"] += float(gate_loss.detach().item())
            totals["persistence_loss"] += float(persistence_loss.detach().item())
            totals["total_loss"] += float(total_loss.detach().item())
            totals["attention_supervised_pixels"] += int(
                targets["attention_supervised"].sum().item()
            )
            totals["rescue_pixels"] += rescue_count
            totals["protection_pixels"] += protection_count
            totals["gate_supervised_pixels"] += int(
                targets["gate_supervised"].sum().item()
            )
            totals["persistence_pixels"] += persistence_pixels
            totals["lambda_mean"] += float(full["lambda_probability"].mean().item())
            if rescue_count:
                totals["lambda_on_rescue"] += float(
                    full["lambda_probability"][0, 0][targets["rescue"]].mean().item()
                )
                totals["lambda_rescue_frames"] += 1
            if protection_count:
                totals["lambda_on_protection"] += float(
                    full["lambda_probability"][0, 0][targets["protection"]].mean().item()
                )
                totals["lambda_protection_frames"] += 1
            totals["prediction_error_abs"] += float(
                evidence["prediction_error"].abs().mean().item()
            )
            totals["dynamics_error_abs"] += float(
                dynamics_state.abs().mean().item()
            )
            for age in range(HISTORY_LENGTH):
                valid = full["history_validities"][0, age]
                if bool(valid.any()):
                    totals["attention_mass_by_age"][age] += float(
                        full["attention"][0, age][valid].mean().item()
                    )
                    totals["attention_age_frames"][age] += 1

        frames_in_window += 1
        boundary = frames_in_window >= tbptt_steps or frame_index == len(samples) - 1
        if boundary:
            window_loss = torch.stack(buffered_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            window_loss.backward()
            optimizer.step()
            totals["optimizer_steps"] += 1
            buffered_losses = []
            frames_in_window = 0

            if frame_index != len(samples) - 1:
                boundary_hidden = (
                    correction_hidden_before.detach()
                    if correction_hidden_before is not None
                    else None
                )
                boundary_dynamics = (
                    dynamics_state_before.detach()
                    if dynamics_state_before is not None
                    else None
                )
                with torch.no_grad():
                    refreshed = _low_correction_evidence(
                        corrector,
                        dynamics,
                        c_v3_logits,
                        candidate_rows,
                        pending_motion,
                        e1["transportability_low"],
                        memory_row["memory_reliability"],
                        boundary_hidden,
                        boundary_dynamics,
                    )
                    correction_hidden = refreshed["row"]["correction_hidden"].detach()
                    dynamics_state = refreshed["dynamics_state"].detach()
                    refreshed_full = _full_resolution_soft_output(
                        c_v3_logits,
                        candidate_rows,
                        refreshed["row"],
                    )
                    current_correction = (
                        refreshed_full["output_probability"]
                        - refreshed_full["current_probability"]
                    ).detach()
                    current_output_correct = (
                        refreshed_full["output_probability"].argmax(1)[0]
                        == targets["gt"]
                    ) & targets["valid_gt"]
            else:
                correction_hidden = (
                    correction_hidden.detach() if correction_hidden is not None else None
                )
                dynamics_state = (
                    dynamics_state.detach() if dynamics_state is not None else None
                )
            if memory_state is not None:
                memory_state = memory_state.detach()

        raw_history.insert(0, c_v3_logits.detach())
        raw_history = raw_history[:HISTORY_LENGTH]
        motion_history.insert(0, pending_motion.detach())
        motion_history = motion_history[: HISTORY_LENGTH - 1]

        previous_correction = current_correction.detach()
        previous_rescue = targets["rescue"].detach().cpu()
        previous_output_correct = current_output_correct.detach().cpu()
        previous_gt = current_gt
        previous_image = current_image
        previous_low = host_low.detach()
        previous_c1 = current_c1.detach()
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()

    supervised_frames = max(totals["supervised_frames"], 1)
    for key in (
        "seg_loss",
        "attention_loss",
        "gate_loss",
        "persistence_loss",
        "total_loss",
        "lambda_mean",
        "prediction_error_abs",
        "dynamics_error_abs",
    ):
        totals[key] /= supervised_frames
    totals["lambda_on_rescue"] /= max(totals["lambda_rescue_frames"], 1)
    totals["lambda_on_protection"] /= max(totals["lambda_protection_frames"], 1)
    for age in range(HISTORY_LENGTH):
        totals["attention_mass_by_age"][age] /= max(
            totals["attention_age_frames"][age],
            1,
        )
    return totals


def _train_epoch(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    corrector,
    dynamics,
    raft,
    groups,
    optimizer,
    tbptt_steps,
):
    corrector.train()
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
            corrector,
            dynamics,
            raft,
            samples,
            optimizer,
            tbptt_steps,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("No valid C-V6 training sequences")

    frame_total = max(sum(row["supervised_frames"] for row in rows), 1)
    result = {
        "frames": sum(row["frames"] for row in rows),
        "supervised_frames": sum(row["supervised_frames"] for row in rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "attention_supervised_pixels": sum(
            row["attention_supervised_pixels"] for row in rows
        ),
        "rescue_pixels": sum(row["rescue_pixels"] for row in rows),
        "protection_pixels": sum(row["protection_pixels"] for row in rows),
        "gate_supervised_pixels": sum(row["gate_supervised_pixels"] for row in rows),
        "persistence_pixels": sum(row["persistence_pixels"] for row in rows),
    }
    for key in (
        "seg_loss",
        "attention_loss",
        "gate_loss",
        "persistence_loss",
        "total_loss",
        "lambda_mean",
        "prediction_error_abs",
        "dynamics_error_abs",
    ):
        result[key] = sum(
            row[key] * row["supervised_frames"] for row in rows
        ) / frame_total

    rescue_frames = max(sum(row["lambda_rescue_frames"] for row in rows), 1)
    protection_frames = max(sum(row["lambda_protection_frames"] for row in rows), 1)
    result["lambda_on_rescue"] = sum(
        row["lambda_on_rescue"] * row["lambda_rescue_frames"] for row in rows
    ) / rescue_frames
    result["lambda_on_protection"] = sum(
        row["lambda_on_protection"] * row["lambda_protection_frames"] for row in rows
    ) / protection_frames
    result["lambda_rescue_frames"] = sum(row["lambda_rescue_frames"] for row in rows)
    result["lambda_protection_frames"] = sum(
        row["lambda_protection_frames"] for row in rows
    )

    result["attention_mass_by_age"] = []
    for age in range(HISTORY_LENGTH):
        age_frames = max(sum(row["attention_age_frames"][age] for row in rows), 1)
        result["attention_mass_by_age"].append(
            sum(
                row["attention_mass_by_age"][age] * row["attention_age_frames"][age]
                for row in rows
            ) / age_frames
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
    corrector,
    dynamics,
    groups,
    raft,
):
    corrector.eval()
    refiner.eval()
    correction.eval()
    mask_predictor.eval()

    confusion = {
        name: torch.zeros((v5.NUM_CLASSES, v5.NUM_CLASSES), dtype=torch.int64)
        for name in CANDIDATES
    }
    mtc_sum = {name: 0.0 for name in CANDIDATES}
    mtc_count = {name: 0 for name in CANDIDATES}
    vc_sum = {name: {8: 0.0, 16: 0.0} for name in CANDIDATES}
    vc_count = {name: {8: 0, 16: 0} for name in CANDIDATES}

    diagnostics = {
        "decision_frames": 0,
        "c_v6_vs_c_v3_disagreement_pixels": 0,
        "lambda_sum": 0.0,
        "lambda_rescue_sum": 0.0,
        "lambda_rescue_frames": 0,
        "lambda_protection_sum": 0.0,
        "lambda_protection_frames": 0,
        "attention_supervised_pixels": 0,
        "rescue_pixels": 0,
        "protection_pixels": 0,
        "attention_mass_by_age": [0.0] * HISTORY_LENGTH,
        "attention_age_frames": [0] * HISTORY_LENGTH,
    }

    for sequence, samples in groups.items():
        previous = None
        pending_motion = motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = None
        memory_state = None
        correction_hidden = None
        dynamics_state = None
        raw_history = []
        motion_history = []
        previous_predictions = {}
        seq_vc = {name: v5.VideoConsistency() for name in CANDIDATES}

        for sample in samples:
            image, host_logits, host_low, current_c1, output_size = v5._host_observation(
                model,
                sample,
            )
            host_pred = host_logits.argmax(1)
            gt_cpu = v5.semantic_mask_from_panoptic_png(sample["mask_path"])
            previous_image_for_mtc = previous[0] if previous is not None else None
            teacher_full = (
                raft.current_to_previous(image, previous_image_for_mtc)
                if previous_image_for_mtc is not None
                else None
            )

            if previous is None:
                e1_pred = c_v3_pred = c_v6_pred = host_pred
                semantic_state_low = torch.zeros_like(host_low)
                previous = (image, host_low.detach(), current_c1.detach())

            elif pending_motion is None:
                previous_image, previous_low, previous_c1 = previous
                observed = v5._observe_motion(
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
                e1_pred = c_v3_pred = c_v6_pred = host_pred
                previous = (image, host_low.detach(), current_c1.detach())

            else:
                previous_image, previous_low, previous_c1 = previous
                prior_low, _ = v5.warp_low_logits(previous_low, pending_motion)
                e1 = v5._frozen_e1_step(
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
                memory_row, memory_state, e1_logits, c_v3_logits = v5._frozen_cv3_step(
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

                observed = v5._observe_motion(
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

                e1_pred = e1_logits.argmax(1)
                c_v3_pred = c_v3_logits.argmax(1)
                if not raw_history:
                    c_v6_pred = c_v3_pred
                    raw_history = [c_v3_logits.detach()]
                else:
                    candidate_rows = _build_probability_history_candidates(
                        raw_history,
                        motion_history,
                        pending_motion,
                        HISTORY_LENGTH,
                    )
                    evidence = _low_correction_evidence(
                        corrector,
                        dynamics,
                        c_v3_logits,
                        candidate_rows,
                        pending_motion,
                        e1["transportability_low"],
                        memory_row["memory_reliability"],
                        correction_hidden,
                        dynamics_state,
                    )
                    correction_hidden = evidence["row"]["correction_hidden"]
                    dynamics_state = evidence["dynamics_state"]
                    full = _full_resolution_soft_output(
                        c_v3_logits,
                        candidate_rows,
                        evidence["row"],
                    )
                    c_v6_pred = full["output_probability"].argmax(1)
                    targets = _semantic_targets(full, candidate_rows, gt_cpu)

                    diagnostics["decision_frames"] += 1
                    diagnostics["c_v6_vs_c_v3_disagreement_pixels"] += int(
                        (c_v6_pred != c_v3_pred).sum().item()
                    )
                    diagnostics["lambda_sum"] += float(
                        full["lambda_probability"].mean().item()
                    )
                    diagnostics["attention_supervised_pixels"] += int(
                        targets["attention_supervised"].sum().item()
                    )
                    rescue_count = int(targets["rescue"].sum().item())
                    protection_count = int(targets["protection"].sum().item())
                    diagnostics["rescue_pixels"] += rescue_count
                    diagnostics["protection_pixels"] += protection_count
                    if rescue_count:
                        diagnostics["lambda_rescue_sum"] += float(
                            full["lambda_probability"][0, 0][targets["rescue"]].mean().item()
                        )
                        diagnostics["lambda_rescue_frames"] += 1
                    if protection_count:
                        diagnostics["lambda_protection_sum"] += float(
                            full["lambda_probability"][0, 0][targets["protection"]].mean().item()
                        )
                        diagnostics["lambda_protection_frames"] += 1
                    for age in range(HISTORY_LENGTH):
                        valid = full["history_validities"][0, age]
                        if bool(valid.any()):
                            diagnostics["attention_mass_by_age"][age] += float(
                                full["attention"][0, age][valid].mean().item()
                            )
                            diagnostics["attention_age_frames"][age] += 1

                    raw_history.insert(0, c_v3_logits.detach())
                    raw_history = raw_history[:HISTORY_LENGTH]
                    motion_history.insert(0, pending_motion.detach())
                    motion_history = motion_history[: HISTORY_LENGTH - 1]

                previous = (image, host_low.detach(), current_c1.detach())
                pending_motion = next_motion.detach()
                motion_hidden = next_motion_hidden.detach()

            predictions = {
                "host": host_pred,
                "e1_base": e1_pred,
                "c_v3_base": c_v3_pred,
                "c_v6": c_v6_pred,
            }
            for name, prediction in predictions.items():
                pred_cpu = prediction.squeeze(0).cpu()
                v5.update_confusion_matrix(confusion[name], pred_cpu, gt_cpu)
                seq_vc[name].update(gt_cpu, pred_cpu)

            if teacher_full is not None:
                for name, prediction in predictions.items():
                    if name in previous_predictions:
                        score = v5._pair_mtc(
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
            "mIoU": float(torch.nanmean(v5.compute_iou(confusion[name])).item()),
            "mTC": mtc_sum[name] / mtc_count[name] if mtc_count[name] else float("nan"),
            "mVC8": vc_sum[name][8] / vc_count[name][8] if vc_count[name][8] else float("nan"),
            "mVC16": vc_sum[name][16] / vc_count[name][16] if vc_count[name][16] else float("nan"),
        }
        for name in CANDIDATES
    }
    decision_frames = max(diagnostics["decision_frames"], 1)
    diagnostics_out = {
        "decision_frames": diagnostics["decision_frames"],
        "c_v6_vs_c_v3_disagreement_pixels": diagnostics[
            "c_v6_vs_c_v3_disagreement_pixels"
        ],
        "lambda_mean": diagnostics["lambda_sum"] / decision_frames,
        "lambda_on_rescue": diagnostics["lambda_rescue_sum"]
        / max(diagnostics["lambda_rescue_frames"], 1),
        "lambda_on_protection": diagnostics["lambda_protection_sum"]
        / max(diagnostics["lambda_protection_frames"], 1),
        "attention_supervised_pixels": diagnostics["attention_supervised_pixels"],
        "rescue_pixels": diagnostics["rescue_pixels"],
        "protection_pixels": diagnostics["protection_pixels"],
        "attention_mass_by_age": [
            diagnostics["attention_mass_by_age"][age]
            / max(diagnostics["attention_age_frames"][age], 1)
            for age in range(HISTORY_LENGTH)
        ],
        "history_length": HISTORY_LENGTH,
        "history_source": "detached raw frozen C-V3 logits only",
        "history_transport": (
            "softmax(raw C-V3 logits) -> one final probability warp -> renormalize"
        ),
        "controller_output_feedback": False,
        "semantic_state_recursive": False,
        "prediction_error_reference": "t-1 frozen C-V3 only",
        "dynamics_error_reference": "t-1 prediction error only",
        "raft_inference": False,
    }
    return metrics, diagnostics_out


def _delta_metrics(candidate, reference):
    return {
        key: candidate[key] - reference[key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }


def _selection_key(metrics):
    candidate = metrics["c_v6"]
    preserved = candidate["mIoU"] >= CV4_E2_MIOU_REFERENCE
    if preserved:
        return (1, candidate["mTC"], candidate["mIoU"])
    return (0, candidate["mIoU"], candidate["mTC"])


def _zero_step_equivalence_check(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    corrector,
    dynamics,
    val_groups,
    raft,
):
    check_groups = {
        sequence: val_groups[sequence]
        for sequence in ZERO_STEP_SEQUENCES
        if sequence in val_groups
    }
    if not check_groups:
        raise RuntimeError("No Dev3 sequence available for C-V6 zero-step check")
    metrics, diagnostics = _evaluate(
        model,
        observer,
        residual,
        correction,
        mask_predictor,
        refiner,
        corrector,
        dynamics,
        check_groups,
        raft,
    )
    mismatch = diagnostics["c_v6_vs_c_v3_disagreement_pixels"]
    return {
        "sequences": list(check_groups),
        "c_v6_vs_c_v3_disagreement_pixels": mismatch,
        "passed": mismatch == 0,
        "c_v3_base": metrics["c_v3_base"],
        "c_v6": metrics["c_v6"],
        "gate_init_bias": GATE_INIT_BIAS,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=v5.FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=v5.OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=v5.RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--base-checkpoint", default=v5.BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v3-checkpoint", default=v5.C_V3_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--tbptt-steps", type=int, default=TBPTT_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-zero-step-check", action="store_true")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs <= 0 or args.tbptt_steps <= 0:
        raise ValueError("epochs and tbptt-steps must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    v5._validate_bounded_motion_checkpoint(args.residual_checkpoint)
    model = v5.load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = v5._load_frozen_observer(args.observer_checkpoint)
    residual, residual_payload = v5._load_frozen_residual(
        args.residual_checkpoint,
        observer,
    )
    correction, mask_predictor, base_payload = v5._load_frozen_e1_base(
        args.base_checkpoint,
        observer,
    )
    refiner, cv3_payload = v5._load_frozen_cv3_refiner(args.c_v3_checkpoint)

    corrector = SetBasedSoftTemporalCorrector(
        num_classes=v5.NUM_CLASSES,
        history_length=HISTORY_LENGTH,
        attention_hidden_channels=ATTENTION_HIDDEN_CHANNELS,
        correction_hidden_channels=CORRECTION_HIDDEN_CHANNELS,
        gate_init_bias=GATE_INIT_BIAS,
    ).cuda()
    dynamics = EulerDynamicsError(
        tau_e=DYNAMICS_TAU_E,
        k_e=DYNAMICS_K_E,
        dt=DYNAMICS_DT,
    )

    for name, module in (
        ("Host", model),
        ("Motion Observer", observer),
        ("Motion Residual", residual),
        ("E1 correction", correction),
        ("E1 transportability mask", mask_predictor),
        ("C-V3 refiner", refiner),
    ):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError(f"{name} must remain frozen in C-V6")

    raft = v5.FrozenRAFT()
    train = v5.KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root),
        "train",
    )
    val = v5.KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root),
        "val",
    )
    train_groups = v5.sequence_groups(train)
    all_val_groups = v5.sequence_groups(val)
    missing = [sequence for sequence in v5.FULL9 if sequence not in all_val_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {sequence: all_val_groups[sequence] for sequence in v5.FULL9}

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    zero_step = None
    if not args.skip_zero_step_check:
        zero_step = _zero_step_equivalence_check(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            corrector,
            dynamics,
            val_groups,
            raft,
        )
        with (result_output / "zero_step_check.json").open("w") as handle:
            json.dump(zero_step, handle, indent=2)
        if not zero_step["passed"]:
            raise RuntimeError(
                "C-V6 zero-step check failed: untrained hard prediction differs from C-V3"
            )

    optimizer = torch.optim.AdamW(
        corrector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
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
            refiner,
            corrector,
            dynamics,
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
            refiner,
            corrector,
            dynamics,
            val_groups,
            raft,
        )
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_c_v3_base": _delta_metrics(
                metrics["c_v6"],
                metrics["c_v3_base"],
            ),
            "delta_vs_host": _delta_metrics(metrics["c_v6"], metrics["host"]),
            "selection_key": list(_selection_key(metrics)),
        }
        history.append(row)
        with (result_output / f"epoch_{epoch:03d}.json").open("w") as handle:
            json.dump(row, handle, indent=2)

        if best is None or tuple(row["selection_key"]) > tuple(best["selection_key"]):
            best = row
            torch.save(
                {
                    "experiment": "c_v6_soft_temporal_correction",
                    "implementation": "probability_warp_supervision_fix",
                    "epoch": epoch,
                    "corrector_state_dict": corrector.state_dict(),
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "train": train_stats,
                    "dynamics": dynamics.config(),
                    "architecture": {
                        "history_length": HISTORY_LENGTH,
                        "history_source": "detached raw frozen C-V3 logits",
                        "history_transport": (
                            "softmax(raw logits) -> one probability warp -> renormalize"
                        ),
                        "history_attention": "shared pixel-wise learned attention",
                        "recent_history_priority": False,
                        "soft_output": "(1-lambda)*P_current + lambda*P_history",
                        "controller_output_feedback": False,
                        "semantic_state_recursive": False,
                        "prediction_error_reference": "t-1 frozen C-V3 only",
                        "attention_hidden_channels": ATTENTION_HIDDEN_CHANNELS,
                        "correction_hidden_channels": CORRECTION_HIDDEN_CHANNELS,
                        "gate_init_bias": GATE_INIT_BIAS,
                    },
                    "loss_weights": {
                        "seg": LOSS_SEG_WEIGHT,
                        "attention": LOSS_ATTN_WEIGHT,
                        "gate": LOSS_GATE_WEIGHT,
                        "persistence": LOSS_PERSIST_WEIGHT,
                    },
                },
                output / "best.pt",
            )
        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V6 produced no epoch result")

    candidate = best["metrics"]["c_v6"]
    summary = {
        "experiment": "C-V6 Set-based Deep Semantic Evidence + Soft Stateful Correction",
        "implementation": "probability_warp_supervision_fix",
        "source_decision_evidence": {
            "c_v5_oracle_commit": "ff404d4",
            "decision": (
                "K=4 adds strong semantic rescue but does not improve hard-selection mTC; "
                "deep history is therefore retained as soft semantic evidence rather than "
                "direct final-output candidates."
            ),
        },
        "zero_step_check": zero_step,
        "best": best,
        "history": history,
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
        "selection_rule": {
            "hard_constraint": f"C-V6 mIoU >= {CV4_E2_MIOU_REFERENCE}",
            "objective_after_constraint": "maximize C-V6 mTC, then mIoU",
        },
        "targets": {
            "mIoU_floor": CV4_E2_MIOU_REFERENCE,
            "mIoU_preserved": candidate["mIoU"] >= CV4_E2_MIOU_REFERENCE,
            "mTC_previous_best_reference": CV4_E3_MTC_REFERENCE,
            "mTC_milestone_72": MTC_MILESTONE,
            "mTC_reached_72": candidate["mTC"] >= MTC_MILESTONE,
            "mTC_target_low": MTC_TARGET_LOW,
            "mTC_target_high": MTC_TARGET_HIGH,
        },
        "architecture": {
            "history_length": HISTORY_LENGTH,
            "history_attention": "candidate-preserving pixel-wise soft attention",
            "history_probability_warp": True,
            "soft_gate": True,
            "recursive_semantic_state": False,
            "controller_output_feedback": False,
            "raft_inference": False,
        },
        "losses": {
            "segmentation": "full-resolution final-output NLL",
            "attention": (
                "uniform mass over every GT-correct valid history wherever one exists"
            ),
            "gate": (
                "balanced Rescue=1; Protection=0 only when Current is correct and no history is correct"
            ),
            "persistence": (
                "RAFT+GT stable correspondence; consecutive rescue plus previous C-V6 actually correct"
            ),
            "weights": [1.0, 1.0, 1.0, 1.0],
        },
        "epochs": args.epochs,
        "tbptt_steps": args.tbptt_steps,
    }
    with (result_output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
