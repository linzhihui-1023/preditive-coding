"""C-V4 corrected main: full-resolution semantic-safe stateful hysteresis.

This entrypoint supersedes the first C-V4 draft for execution.

Corrections relative to the draft:
1. Current/history semantic correctness is supervised at the same full resolution
   used by mIoU/mTC, not against nearest-downsampled GT.
2. Historical semantic logits are motion-warped at full resolution. The final
   decision never upsamples a low-resolution class map by nearest neighbor.
3. Dynamics Error is motion-compensated and receives zero prediction error where
   predicted history is invalid.
4. BCE is aggregated per supervised conflict pixel over each TBPTT window rather
   than averaging per-frame means.
5. The Controller output remains non-autoregressive: only frozen C-V3 output is
   propagated as the next semantic history.

Semantic correctness has priority. Temporal preference is supervised only when
both current and history candidates are GT-wrong, using RAFT-aligned previous
frozen C-V3 prediction as the training-only temporal reference.
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
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_utility_from_e1_r2 import (
    BASE_CHECKPOINT_DEFAULT,
    _load_frozen_e1_base,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v3_semantic_memory_stage_a import (
    _frozen_e1_step,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v3_temporal_semantic_prediction import (
    C_V3_CHECKPOINT_DEFAULT,
    _frozen_cv3_step,
    _load_frozen_cv3_refiner,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT, flow_grid
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    low_flow_grid,
    warp_low_logits,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
    StatefulSemanticHysteresisController,
)

SEED = 0
EPOCHS = 3
TBPTT_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
CONTROLLER_HIDDEN_CHANNELS = 32

# Fixed first-pass time scale in frame units. e_t is also supplied explicitly,
# so the Controller sees both fast and slow error evidence. No sweep is built in.
DYNAMICS_TAU_E = 4.0
DYNAMICS_K_E = 1.0
DYNAMICS_DT = 1.0

MIOU_REFERENCE = 0.662706834793679
MTC_TARGET_LOW = 0.74
MTC_TARGET_HIGH = 0.76

OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v4_stateful_semantic_hysteresis_main"
)
RESULT_DEFAULT = (
    "results/kitti_step_task_space_prior_c_v4_stateful_semantic_hysteresis_main"
)
CANDIDATES = (
    "host",
    "e1_base",
    "c_v3_base",
    "hysteresis",
    "semantic_oracle",
    "semantic_temporal_oracle",
)


def _upsample_predicted_flow(backward_flow_low, full_size):
    """Convert low-resolution pixel displacement to full-resolution pixels."""
    full_h, full_w = full_size
    low_h, low_w = backward_flow_low.shape[-2:]
    flow = F.interpolate(
        backward_flow_low.float(),
        size=full_size,
        mode="bilinear",
        align_corners=True,
    ).clone()
    flow[:, 0] *= (full_w - 1) / max(low_w - 1, 1)
    flow[:, 1] *= (full_h - 1) / max(low_h - 1, 1)
    return flow.to(backward_flow_low.dtype)


def _warp_full_logits_zero_invalid(previous_logits, backward_flow_low):
    """Warp previous full-resolution logits with predicted motion; zero invalid."""
    full_size = tuple(previous_logits.shape[-2:])
    flow_full = _upsample_predicted_flow(backward_flow_low, full_size)
    grid, valid = flow_grid(flow_full, full_size[0], full_size[1])
    warped = F.grid_sample(
        previous_logits.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    warped = warped * valid.unsqueeze(1).to(warped.dtype)
    return warped.to(previous_logits.dtype), valid


def _warp_low_state_zero_invalid(previous_state, backward_flow_low):
    """Motion-compensate an explicit low-resolution state with zero invalid."""
    if previous_state is None:
        return None, None
    grid, valid = low_flow_grid(backward_flow_low)
    warped = F.grid_sample(
        previous_state.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    warped = warped * valid.unsqueeze(1).to(warped.dtype)
    return warped.to(previous_state.dtype), valid


def _warp_previous_prediction_with_raft(previous_logits, teacher_full):
    """Training/evaluation-only temporal reference for mTC-consistent targets."""
    height, width = previous_logits.shape[-2:]
    grid, valid = flow_grid(teacher_full, height, width)
    previous_prediction = previous_logits.argmax(dim=1).float().unsqueeze(1)
    warped = F.grid_sample(
        previous_prediction,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0].long()
    return warped, valid


def _build_full_resolution_target(
    current_logits,
    history_logits,
    history_valid,
    previous_cv3_logits,
    current_gt_cpu,
    teacher_full,
):
    """Semantic-first, temporal-second target at final evaluation resolution."""
    current_pred = current_logits.argmax(dim=1)[0]
    history_pred = history_logits.argmax(dim=1)[0]
    gt = current_gt_cpu.to(current_logits.device, non_blocking=True)
    valid_gt = gt != IGNORE_LABEL
    valid_history = history_valid[0].bool()
    conflict = valid_gt & valid_history & (current_pred != history_pred)

    current_correct = current_pred == gt
    history_correct = history_pred == gt
    exactly_one_correct = current_correct ^ history_correct
    both_wrong = ~current_correct & ~history_correct

    teacher_previous, teacher_valid = _warp_previous_prediction_with_raft(
        previous_cv3_logits,
        teacher_full,
    )
    teacher_previous = teacher_previous[0]
    teacher_valid = teacher_valid[0].bool()
    current_temporal_match = current_pred == teacher_previous
    history_temporal_match = history_pred == teacher_previous
    temporal_discriminable = (
        teacher_valid & (current_temporal_match ^ history_temporal_match)
    )

    supervised = conflict & (
        exactly_one_correct | (both_wrong & temporal_discriminable)
    )
    keep_history = supervised & (
        (history_correct & ~current_correct)
        | (
            both_wrong
            & history_temporal_match
            & ~current_temporal_match
        )
    )
    semantic_keep = conflict & history_correct & ~current_correct

    row = {
        "valid_pixels": int(valid_gt.sum().item()),
        "conflict_pixels": int(conflict.sum().item()),
        "supervised_conflict_pixels": int(supervised.sum().item()),
        "keep_target_pixels": int(keep_history.sum().item()),
        "semantic_keep_pixels": int(semantic_keep.sum().item()),
        "current_correct_history_wrong": int(
            (conflict & current_correct & ~history_correct).sum().item()
        ),
        "current_wrong_history_correct": int(
            (conflict & ~current_correct & history_correct).sum().item()
        ),
        "both_wrong_history_temporal_better": int(
            (
                conflict
                & both_wrong
                & teacher_valid
                & history_temporal_match
                & ~current_temporal_match
            ).sum().item()
        ),
        "both_wrong_current_temporal_better": int(
            (
                conflict
                & both_wrong
                & teacher_valid
                & current_temporal_match
                & ~history_temporal_match
            ).sum().item()
        ),
        "both_wrong_temporal_tie_or_invalid": int(
            (conflict & both_wrong & ~temporal_discriminable).sum().item()
        ),
    }
    return keep_history, semantic_keep, supervised, conflict, row


def _new_target_totals():
    return {
        "valid_pixels": 0,
        "conflict_pixels": 0,
        "supervised_conflict_pixels": 0,
        "keep_target_pixels": 0,
        "semantic_keep_pixels": 0,
        "current_correct_history_wrong": 0,
        "current_wrong_history_correct": 0,
        "both_wrong_history_temporal_better": 0,
        "both_wrong_current_temporal_better": 0,
        "both_wrong_temporal_tie_or_invalid": 0,
    }


def _add_target_totals(total, row):
    for key in total:
        total[key] += int(row[key])


def _target_rates(total):
    valid = max(total["valid_pixels"], 1)
    conflict = max(total["conflict_pixels"], 1)
    supervised = max(total["supervised_conflict_pixels"], 1)
    return {
        **total,
        "conflict_fraction_of_valid": total["conflict_pixels"] / valid,
        "supervised_fraction_of_conflicts": (
            total["supervised_conflict_pixels"] / conflict
        ),
        "keep_target_fraction_of_supervised": (
            total["keep_target_pixels"] / supervised
        ),
        "semantic_keep_fraction_of_conflicts": (
            total["semantic_keep_pixels"] / conflict
        ),
        "current_correct_history_wrong_fraction_of_conflicts": (
            total["current_correct_history_wrong"] / conflict
        ),
        "current_wrong_history_correct_fraction_of_conflicts": (
            total["current_wrong_history_correct"] / conflict
        ),
        "both_wrong_history_temporal_better_fraction_of_conflicts": (
            total["both_wrong_history_temporal_better"] / conflict
        ),
        "both_wrong_current_temporal_better_fraction_of_conflicts": (
            total["both_wrong_current_temporal_better"] / conflict
        ),
        "both_wrong_temporal_tie_or_invalid_fraction_of_conflicts": (
            total["both_wrong_temporal_tie_or_invalid"] / conflict
        ),
    }


def _controller_evidence(
    controller,
    dynamics,
    c_v3_logits,
    history_logits,
    history_valid_full,
    pending_motion,
    transportability_low,
    memory_reliability_low,
    controller_hidden,
    dynamics_state,
):
    low_size = tuple(transportability_low.shape[-2:])
    current_low = F.interpolate(
        c_v3_logits.detach(),
        size=low_size,
        mode="bilinear",
        align_corners=False,
    )
    history_low = F.interpolate(
        history_logits.detach(),
        size=low_size,
        mode="bilinear",
        align_corners=False,
    )
    current_probability = F.softmax(current_low, dim=1)
    history_probability = F.softmax(history_low, dim=1)

    _, low_valid = low_flow_grid(pending_motion)
    history_valid_low = low_valid.unsqueeze(1).to(current_probability.dtype)
    prediction_error = (
        current_probability - history_probability
    ) * history_valid_low

    if dynamics_state is None:
        warped_dynamics = None
    else:
        warped_dynamics, _ = _warp_low_state_zero_invalid(
            dynamics_state,
            pending_motion,
        )
    dynamics_state = dynamics.step(
        prediction_error,
        warped_dynamics,
    ).detach()

    row = controller(
        current_probability.detach(),
        history_probability.detach(),
        prediction_error.detach(),
        dynamics_state.detach(),
        transportability_low.detach(),
        memory_reliability_low.detach(),
        history_valid_low.detach(),
        controller_hidden,
    )
    keep_logit_full = F.interpolate(
        row["keep_logit"],
        size=tuple(c_v3_logits.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )[:, 0]

    valid_from_low = F.interpolate(
        history_valid_low,
        size=tuple(c_v3_logits.shape[-2:]),
        mode="nearest",
    )[:, 0] > 0.5
    # Full-resolution warp validity is authoritative for final decisions.
    valid_full = valid_from_low & history_valid_full.bool()

    return {
        "row": row,
        "keep_logit_full": keep_logit_full,
        "prediction_error": prediction_error,
        "dynamics_state": dynamics_state,
        "history_valid_low": history_valid_low,
        "history_valid_full": valid_full,
    }


def _apply_keep(current_prediction, history_prediction, keep_mask):
    output = current_prediction.clone()
    output[0][keep_mask] = history_prediction[0][keep_mask]
    return output


def _zero_step_check(controller):
    weight_max = float(controller.keep_head.weight.detach().abs().max().item())
    bias_max = float(controller.keep_head.bias.detach().abs().max().item())
    if weight_max != 0.0 or bias_max != 0.0:
        raise RuntimeError("C-V4 final keep head must be exactly zero initialized")
    return {
        "keep_head_weight_abs_max": weight_max,
        "keep_head_bias_abs_max": bias_max,
        "hard_rule": "keep_logit_full > 0",
        "zero_step_behavior": "all final pixels use frozen C-V3",
    }


def _train_sequence(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    controller,
    dynamics,
    raft,
    samples,
    optimizer,
    tbptt_steps,
):
    if len(samples) < 3:
        return None

    frame0 = _host_observation(model, samples[0])
    frame1 = _host_observation(model, samples[1])
    pending_motion, motion_hidden = _initialize_motion(
        observer, residual, frame0, frame1
    )
    previous_image, previous_host_logits, previous_low, previous_c1, _ = frame1

    previous_cv3_logits = previous_host_logits.detach()
    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = None
    controller_hidden = None
    dynamics_state = None

    loss_sums = []
    loss_counts = []
    frames_in_window = 0
    totals = {
        "frames": 0,
        "supervised_frames": 0,
        "optimizer_steps": 0,
        "controller_bce_per_pixel": 0.0,
        "supervised_pixels": 0,
        "hard_keep_fraction_of_conflicts": 0.0,
        "keep_probability_on_conflicts": 0.0,
        "prediction_error_abs": 0.0,
        "dynamics_error_abs": 0.0,
    }
    target_totals = _new_target_totals()

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

            memory_row, memory_state, e1_logits, c_v3_logits = _frozen_cv3_step(
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
            history_logits, history_valid_full = _warp_full_logits_zero_invalid(
                previous_cv3_logits,
                pending_motion,
            )

        evidence = _controller_evidence(
            controller,
            dynamics,
            c_v3_logits,
            history_logits,
            history_valid_full,
            pending_motion,
            e1["transportability_low"],
            memory_row["memory_reliability"],
            controller_hidden,
            dynamics_state,
        )
        controller_hidden = evidence["row"]["hidden"]
        dynamics_state = evidence["dynamics_state"]

        with torch.no_grad():
            teacher_full = raft.current_to_previous(current_image, previous_image)
            (
                keep_target,
                _,
                supervised,
                conflict,
                target_diag,
            ) = _build_full_resolution_target(
                c_v3_logits,
                history_logits,
                evidence["history_valid_full"],
                previous_cv3_logits,
                current_gt,
                teacher_full,
            )
            _add_target_totals(target_totals, target_diag)

        selected = supervised.unsqueeze(0)
        if bool(selected.any()):
            target = keep_target.float().unsqueeze(0)
            pixel_loss_sum = F.binary_cross_entropy_with_logits(
                evidence["keep_logit_full"][selected],
                target[selected],
                reduction="sum",
            )
            count = int(selected.sum().item())
            loss_sums.append(pixel_loss_sum)
            loss_counts.append(count)
            totals["supervised_frames"] += 1
            totals["supervised_pixels"] += count
            totals["controller_bce_per_pixel"] += float(pixel_loss_sum.detach().item())

        with torch.no_grad():
            conflict_count = int(conflict.sum().item())
            hard_keep = (evidence["keep_logit_full"][0] > 0) & conflict
            keep_probability = torch.sigmoid(evidence["keep_logit_full"][0])
            totals["hard_keep_fraction_of_conflicts"] += float(
                hard_keep.sum().item() / max(conflict_count, 1)
            )
            totals["keep_probability_on_conflicts"] += float(
                keep_probability[conflict].mean().item()
                if conflict_count else 0.0
            )
            totals["prediction_error_abs"] += float(
                evidence["prediction_error"].abs().mean().item()
            )
            totals["dynamics_error_abs"] += float(
                dynamics_state.abs().mean().item()
            )

            observed_motion = _observe_motion(
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
        frames_in_window += 1
        boundary = frames_in_window >= tbptt_steps or frame_index == len(samples) - 1
        if boundary:
            if loss_sums:
                total_count = max(sum(loss_counts), 1)
                window_loss = torch.stack(loss_sums).sum() / float(total_count)
                optimizer.zero_grad(set_to_none=True)
                window_loss.backward()
                optimizer.step()
                totals["optimizer_steps"] += 1
            loss_sums = []
            loss_counts = []
            frames_in_window = 0
            if controller_hidden is not None:
                controller_hidden = controller_hidden.detach()
            if memory_state is not None:
                memory_state = memory_state.detach()
            if dynamics_state is not None:
                dynamics_state = dynamics_state.detach()

        previous_image = current_image
        previous_low = host_low.detach()
        previous_c1 = current_c1.detach()
        previous_cv3_logits = c_v3_logits.detach()  # never Controller output
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()

    frames = max(totals["frames"], 1)
    totals["controller_bce_per_pixel"] /= max(totals["supervised_pixels"], 1)
    for key in (
        "hard_keep_fraction_of_conflicts",
        "keep_probability_on_conflicts",
        "prediction_error_abs",
        "dynamics_error_abs",
    ):
        totals[key] /= frames
    totals["targets"] = _target_rates(target_totals)
    return totals


def _train_epoch(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    controller,
    dynamics,
    raft,
    groups,
    optimizer,
    tbptt_steps,
):
    controller.train()
    refiner.eval()
    correction.eval()
    mask_predictor.eval()
    rows = []
    target_totals = _new_target_totals()

    for samples in groups.values():
        row = _train_sequence(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            controller,
            dynamics,
            raft,
            samples,
            optimizer,
            tbptt_steps,
        )
        if row is not None:
            rows.append(row)
            for key in target_totals:
                target_totals[key] += row["targets"][key]

    if not rows:
        raise RuntimeError("No valid C-V4 training sequences")

    frame_total = max(sum(row["frames"] for row in rows), 1)
    supervised_pixel_total = max(sum(row["supervised_pixels"] for row in rows), 1)
    result = {
        "frames": sum(row["frames"] for row in rows),
        "supervised_frames": sum(row["supervised_frames"] for row in rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "supervised_pixels": sum(row["supervised_pixels"] for row in rows),
        "controller_bce_per_pixel": (
            sum(
                row["controller_bce_per_pixel"] * row["supervised_pixels"]
                for row in rows
            )
            / supervised_pixel_total
        ),
    }
    for key in (
        "hard_keep_fraction_of_conflicts",
        "keep_probability_on_conflicts",
        "prediction_error_abs",
        "dynamics_error_abs",
    ):
        result[key] = (
            sum(row[key] * row["frames"] for row in rows) / frame_total
        )
    result["targets"] = _target_rates(target_totals)
    return result


@torch.inference_mode()
def _evaluate(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    controller,
    dynamics,
    groups,
    raft,
):
    controller.eval()
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
    target_totals = _new_target_totals()
    diagnostics = {
        "decision_frames": 0,
        "conflict_pixels": 0,
        "valid_pixels": 0,
        "hard_keep_pixels": 0,
        "keep_probability_sum_on_conflicts": 0.0,
        "prediction_error_abs_sum": 0.0,
        "dynamics_error_abs_sum": 0.0,
        "history_valid_pixels": 0,
        "history_total_pixels": 0,
    }

    for sequence in FULL9:
        previous = None
        previous_cv3_logits = None
        pending_motion = motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = None
        memory_state = None
        controller_hidden = None
        dynamics_state = None
        previous_predictions = {}
        seq_vc = {name: VideoConsistency() for name in CANDIDATES}

        for sample in groups[sequence]:
            image, host_logits, host_low, current_c1, output_size = _host_observation(
                model, sample
            )
            host_pred = host_logits.argmax(1)
            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"])
            previous_image_for_mtc = previous[0] if previous is not None else None
            teacher_full = (
                raft.current_to_previous(image, previous_image_for_mtc)
                if previous_image_for_mtc is not None
                else None
            )

            if previous is None:
                e1_pred = c_v3_pred = hysteresis_pred = host_pred
                semantic_oracle_pred = temporal_oracle_pred = host_pred
                semantic_state_low = torch.zeros_like(host_low)
                previous_cv3_logits = host_logits.detach()
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
                e1_pred = c_v3_pred = hysteresis_pred = host_pred
                semantic_oracle_pred = temporal_oracle_pred = host_pred
                previous_cv3_logits = host_logits.detach()
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

                memory_row, memory_state, e1_logits, c_v3_logits = _frozen_cv3_step(
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
                history_logits, history_valid_full = _warp_full_logits_zero_invalid(
                    previous_cv3_logits,
                    pending_motion,
                )
                evidence = _controller_evidence(
                    controller,
                    dynamics,
                    c_v3_logits,
                    history_logits,
                    history_valid_full,
                    pending_motion,
                    e1["transportability_low"],
                    memory_row["memory_reliability"],
                    controller_hidden,
                    dynamics_state,
                )
                controller_hidden = evidence["row"]["hidden"]
                dynamics_state = evidence["dynamics_state"]

                (
                    oracle_keep,
                    semantic_keep,
                    _,
                    conflict,
                    target_diag,
                ) = _build_full_resolution_target(
                    c_v3_logits,
                    history_logits,
                    evidence["history_valid_full"],
                    previous_cv3_logits,
                    gt_cpu,
                    teacher_full,
                )
                _add_target_totals(target_totals, target_diag)

                current_pred = c_v3_logits.argmax(1)
                history_pred = history_logits.argmax(1)
                hard_keep = (evidence["keep_logit_full"][0] > 0) & conflict

                e1_pred = e1_logits.argmax(1)
                c_v3_pred = current_pred
                hysteresis_pred = _apply_keep(
                    current_pred, history_pred, hard_keep
                )
                semantic_oracle_pred = _apply_keep(
                    current_pred, history_pred, semantic_keep
                )
                temporal_oracle_pred = _apply_keep(
                    current_pred, history_pred, oracle_keep
                )

                conflict_count = int(conflict.sum().item())
                diagnostics["decision_frames"] += 1
                diagnostics["conflict_pixels"] += conflict_count
                diagnostics["valid_pixels"] += target_diag["valid_pixels"]
                diagnostics["hard_keep_pixels"] += int(hard_keep.sum().item())
                keep_probability = torch.sigmoid(evidence["keep_logit_full"][0])
                if conflict_count:
                    diagnostics["keep_probability_sum_on_conflicts"] += float(
                        keep_probability[conflict].sum().item()
                    )
                diagnostics["prediction_error_abs_sum"] += float(
                    evidence["prediction_error"].abs().mean().item()
                )
                diagnostics["dynamics_error_abs_sum"] += float(
                    dynamics_state.abs().mean().item()
                )
                diagnostics["history_valid_pixels"] += int(
                    evidence["history_valid_full"].sum().item()
                )
                diagnostics["history_total_pixels"] += int(
                    evidence["history_valid_full"].numel()
                )

                observed = _observe_motion(
                    observer, previous_low, previous_c1, host_low, current_c1
                )
                motion_error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(
                    observed,
                    motion_error,
                    motion_hidden,
                )
                previous_cv3_logits = c_v3_logits.detach()  # never hysteresis output
                previous = (image, host_low.detach(), current_c1.detach())

            predictions = {
                "host": host_pred,
                "e1_base": e1_pred,
                "c_v3_base": c_v3_pred,
                "hysteresis": hysteresis_pred,
                "semantic_oracle": semantic_oracle_pred,
                "semantic_temporal_oracle": temporal_oracle_pred,
            }
            for name, prediction in predictions.items():
                pred_cpu = prediction.squeeze(0).cpu()
                update_confusion_matrix(confusion[name], pred_cpu, gt_cpu)
                seq_vc[name].update(gt_cpu, pred_cpu)

            if teacher_full is not None:
                for name, prediction in predictions.items():
                    score = _pair_mtc(
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
            "mIoU": float(torch.nanmean(compute_iou(confusion[name])).item()),
            "mTC": (
                mtc_sum[name] / mtc_count[name]
                if mtc_count[name]
                else float("nan")
            ),
            "mVC8": (
                vc_sum[name][8] / vc_count[name][8]
                if vc_count[name][8]
                else float("nan")
            ),
            "mVC16": (
                vc_sum[name][16] / vc_count[name][16]
                if vc_count[name][16]
                else float("nan")
            ),
        }
        for name in CANDIDATES
    }

    decision_frames = max(diagnostics["decision_frames"], 1)
    conflict_pixels = max(diagnostics["conflict_pixels"], 1)
    history_total = max(diagnostics["history_total_pixels"], 1)
    diagnostics_out = {
        "decision_frames": diagnostics["decision_frames"],
        "conflict_fraction_of_valid": (
            diagnostics["conflict_pixels"] / max(diagnostics["valid_pixels"], 1)
        ),
        "hard_keep_fraction_of_conflicts": (
            diagnostics["hard_keep_pixels"] / conflict_pixels
        ),
        "keep_probability_on_conflicts": (
            diagnostics["keep_probability_sum_on_conflicts"] / conflict_pixels
        ),
        "prediction_error_abs": (
            diagnostics["prediction_error_abs_sum"] / decision_frames
        ),
        "dynamics_error_abs": (
            diagnostics["dynamics_error_abs_sum"] / decision_frames
        ),
        "history_valid_fraction": (
            diagnostics["history_valid_pixels"] / history_total
        ),
        "target_distribution": _target_rates(target_totals),
        "semantic_history_source": "previous frozen C-V3 full-resolution logits",
        "controller_output_feedback": False,
        "controller_uses_c1": False,
        "dynamics_motion_compensated": True,
        "invalid_history_prediction_error_zeroed": True,
        "decision_resolution": "full output resolution",
        "oracle_interpretation": (
            "semantic_oracle is correctness-only. semantic_temporal_oracle adds "
            "temporal preference only when both candidates are GT-wrong; neither "
            "oracle is a strict upper bound on all possible mTC policies."
        ),
    }
    return metrics, diagnostics_out


def _delta_metrics(candidate, reference):
    return {
        key: candidate[key] - reference[key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }


def _selection_key(metrics):
    base = metrics["c_v3_base"]
    candidate = metrics["hysteresis"]
    preserved = candidate["mIoU"] >= base["mIoU"]
    if preserved:
        return (1, candidate["mTC"], candidate["mIoU"])
    return (0, candidate["mIoU"], candidate["mTC"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--base-checkpoint", default=BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v3-checkpoint", default=C_V3_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--tbptt-steps", type=int, default=TBPTT_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--dynamics-tau-e", type=float, default=DYNAMICS_TAU_E)
    parser.add_argument("--dynamics-k-e", type=float, default=DYNAMICS_K_E)
    parser.add_argument("--dynamics-dt", type=float, default=DYNAMICS_DT)
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
    refiner, cv3_payload = _load_frozen_cv3_refiner(args.c_v3_checkpoint)

    controller = StatefulSemanticHysteresisController(
        num_classes=NUM_CLASSES,
        hidden_channels=CONTROLLER_HIDDEN_CHANNELS,
    ).cuda()
    dynamics = EulerDynamicsError(
        tau_e=args.dynamics_tau_e,
        k_e=args.dynamics_k_e,
        dt=args.dynamics_dt,
    )
    zero_step = _zero_step_check(controller)

    for name, module in (
        ("C-V3 refiner", refiner),
        ("E1 correction", correction),
        ("E1 transportability mask", mask_predictor),
        ("Motion Observer", observer),
        ("Motion Residual", residual),
    ):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError(f"{name} must remain frozen in C-V4")

    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    raft = FrozenRAFT()

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
            refiner,
            controller,
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
            controller,
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
                    "hysteresis",
                    "semantic_oracle",
                    "semantic_temporal_oracle",
                )
            },
            "delta_vs_host": {
                name: _delta_metrics(metrics[name], metrics["host"])
                for name in (
                    "c_v3_base",
                    "hysteresis",
                    "semantic_oracle",
                    "semantic_temporal_oracle",
                )
            },
        }
        row["selection_key"] = list(_selection_key(metrics))
        history.append(row)

        with (result_output / f"epoch_{epoch:03d}.json").open("w") as handle:
            json.dump(row, handle, indent=2)

        if best is None or tuple(row["selection_key"]) > tuple(best["selection_key"]):
            best = row
            torch.save(
                {
                    "experiment": "c_v4_stateful_semantic_hysteresis_main",
                    "epoch": epoch,
                    "controller_state_dict": controller.state_dict(),
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "train": train_stats,
                    "dynamics": dynamics.config(),
                    "architecture": {
                        "controller_hidden_channels": CONTROLLER_HIDDEN_CHANNELS,
                        "controller_uses_c1": False,
                        "semantic_history_source": (
                            "previous frozen C-V3 full-resolution logits"
                        ),
                        "controller_output_feedback": False,
                        "decision_scope": "full-resolution current/history conflicts",
                        "hard_inference_rule": "upsampled keep_logit > 0",
                        "target_priority": (
                            "semantic correctness then temporal consistency"
                        ),
                        "dynamics_motion_compensated": True,
                    },
                },
                output / "best.pt",
            )

        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V4 produced no epoch result")

    base_metrics = best["metrics"]["c_v3_base"]
    candidate = best["metrics"]["hysteresis"]
    summary = {
        "experiment": "C-V4 Stateful Semantic Hysteresis Main",
        "best": best,
        "history": history,
        "zero_step": zero_step,
        "dynamics": dynamics.config(),
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
            "controller_hidden_channels": CONTROLLER_HIDDEN_CHANNELS,
            "controller_uses_c1": False,
            "semantic_history_source": (
                "previous frozen C-V3 full-resolution logits"
            ),
            "controller_output_feedback": False,
            "prediction_error": (
                "P_current_CV3 - P_motion_warped_previous_CV3, zero on invalid"
            ),
            "dynamics_error_role": (
                "motion-compensated temporal decision evidence only"
            ),
            "decision_resolution": "full output resolution",
            "training_loss": (
                "unbalanced per-conflict-pixel BCE; semantic correctness first, "
                "temporal consistency only on GT-wrong ties"
            ),
            "raft_inference": False,
        },
        "selection_rule": {
            "hard_constraint": "hysteresis mIoU >= frozen C-V3 Base mIoU",
            "objective_after_constraint": "maximize hysteresis mTC, then mIoU",
        },
        "target": {
            "reference_c_v3_mIoU": base_metrics["mIoU"],
            "expected_reference_c_v3_mIoU": MIOU_REFERENCE,
            "mIoU_preserved": candidate["mIoU"] >= base_metrics["mIoU"],
            "mTC_target_low": MTC_TARGET_LOW,
            "mTC_target_high": MTC_TARGET_HIGH,
            "mTC_reached_74": candidate["mTC"] >= MTC_TARGET_LOW,
        },
        "tbptt_steps": args.tbptt_steps,
        "epochs": args.epochs,
    }
    with (result_output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
