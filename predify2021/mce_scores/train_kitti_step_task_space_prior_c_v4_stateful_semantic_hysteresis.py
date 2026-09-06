"""C-V4: non-autoregressive stateful semantic hysteresis on frozen C-V3.

Goal:
    preserve frozen C-V3 mIoU while materially increasing mTC.

Frozen semantic path:
    current_t = C-V3(I_t)
    history_t = Warp(C-V3(I_{t-1}), predicted_motion_t)

Decision evidence:
    e_t = P_current - P_history
    tau_e * d epsilon / dt = e_t - K_e * epsilon
    z_t = ConvGRU(P_current, P_history, e_t, epsilon_t, T, Q_mem, validity)

Only at current/history class conflicts does the controller choose Keep-History
or Use-Current. The final controller output is never fed back as the semantic
history for the next frame; the next history always originates from the frozen
C-V3 base. This cuts the positive feedback path that would otherwise allow a
wrong Keep decision to become self-reinforcing drift.

Training uses ordinary unbalanced BCE on conflict pixels with a lexicographic
target:
  1) semantic correctness first -- if exactly one candidate is GT-correct,
     choose it;
  2) only when both candidates are GT-wrong, use frozen RAFT to ask which
     candidate better matches the RAFT-aligned previous frozen C-V3 prediction.

Thus temporal consistency is optimized only after semantic correctness is tied.
No C1 feature is visible to the controller. RAFT is training/evaluation
supervision only and is absent from inference.
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
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import (
    VideoConsistency,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    sequence_groups,
)
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
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    downsample_backward_flow,
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
DYNAMICS_TAU_E = 4.0
DYNAMICS_K_E = 1.0
DYNAMICS_DT = 1.0
CONTROLLER_HIDDEN_CHANNELS = 32
MTC_TARGET_LOW = 0.74
MTC_TARGET_HIGH = 0.76

OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v4_stateful_semantic_hysteresis"
)
RESULT_DEFAULT = (
    "results/kitti_step_task_space_prior_c_v4_stateful_semantic_hysteresis"
)
CANDIDATES = (
    "host",
    "e1_base",
    "c_v3_base",
    "hysteresis",
    "semantic_temporal_oracle",
)


def _downsample_label(label_cpu, low_size, device):
    label = label_cpu.to(device=device, non_blocking=True).float()[None, None]
    return F.interpolate(label, size=low_size, mode="nearest")[0, 0].long()


def _raft_warp_previous_base_prediction(previous_cv3_low, teacher_full):
    """Training-only temporal reference from previous frozen C-V3 prediction."""
    low_size = tuple(previous_cv3_low.shape[-2:])
    teacher_low = downsample_backward_flow(teacher_full, low_size)
    grid, flow_valid = low_flow_grid(teacher_low)
    previous_prediction = previous_cv3_low.argmax(dim=1).float().unsqueeze(1)
    warped = F.grid_sample(
        previous_prediction,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0].long()
    return warped[0], flow_valid[0]


def _decision_target(
    current_probability,
    history_probability,
    history_valid_low,
    previous_cv3_low,
    current_gt_cpu,
    teacher_full,
):
    """Build semantic-first, temporal-second Keep-History targets.

    On a current/history conflict:
      - current correct, history wrong -> Use Current;
      - history correct, current wrong -> Keep History;
      - both wrong -> use RAFT-aligned previous frozen C-V3 prediction as the
        temporal reference, but supervise only when exactly one candidate
        matches that reference;
      - invalid RAFT geometry or a temporal tie -> do not supervise.

    This is a lexicographic target: temporal consistency never overrides a
    known semantic-correctness advantage.
    """
    low_size = tuple(current_probability.shape[-2:])
    current_gt = _downsample_label(
        current_gt_cpu,
        low_size,
        current_probability.device,
    )
    teacher_previous_pred, teacher_valid = _raft_warp_previous_base_prediction(
        previous_cv3_low,
        teacher_full,
    )

    current_pred = current_probability.argmax(dim=1)[0]
    history_pred = history_probability.argmax(dim=1)[0]
    history_valid = history_valid_low[0, 0].bool()
    current_valid = current_gt != IGNORE_LABEL
    conflict = history_valid & current_valid & (current_pred != history_pred)

    current_correct = current_pred == current_gt
    history_correct = history_pred == current_gt
    exactly_one_correct = current_correct ^ history_correct
    both_wrong = ~current_correct & ~history_correct

    current_temporal_match = current_pred == teacher_previous_pred
    history_temporal_match = history_pred == teacher_previous_pred
    temporal_discriminable = (
        teacher_valid
        & (current_temporal_match ^ history_temporal_match)
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

    diagnostics = {
        "valid_pixels": int(current_valid.sum().item()),
        "conflict_pixels": int(conflict.sum().item()),
        "supervised_conflict_pixels": int(supervised.sum().item()),
        "keep_target_pixels": int(keep_history.sum().item()),
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
            (
                conflict
                & both_wrong
                & ~temporal_discriminable
            ).sum().item()
        ),
    }
    return keep_history, supervised, conflict, diagnostics


def _masked_bce(keep_logit, keep_target, supervised):
    selected = supervised.unsqueeze(0).unsqueeze(0)
    if not bool(selected.any()):
        return keep_logit.sum() * 0.0
    target = keep_target.float().unsqueeze(0).unsqueeze(0)
    return F.binary_cross_entropy_with_logits(
        keep_logit[selected],
        target[selected],
    )


def _upsample_bool(mask_low, output_size):
    if mask_low.ndim == 2:
        mask_low = mask_low[None, None]
    elif mask_low.ndim == 3:
        mask_low = mask_low.unsqueeze(1)
    return (
        F.interpolate(mask_low.float(), size=output_size, mode="nearest")[:, 0]
        > 0.5
    )


def _upsample_label(prediction_low, output_size):
    return F.interpolate(
        prediction_low.float().unsqueeze(1),
        size=output_size,
        mode="nearest",
    )[:, 0].long()


def _apply_keep_prediction(current_pred, history_pred_low, keep_low, output_size):
    history_pred = _upsample_label(history_pred_low, output_size)
    keep_full = _upsample_bool(keep_low, output_size)
    output = current_pred.clone()
    output[keep_full] = history_pred[keep_full]
    return output


def _zero_step_check(controller):
    weight_max = float(controller.keep_head.weight.detach().abs().max().item())
    bias_max = float(controller.keep_head.bias.detach().abs().max().item())
    if weight_max != 0.0 or bias_max != 0.0:
        raise RuntimeError(
            "Controller final head must be exactly zero initialized for C-V3 equality"
        )
    return {
        "keep_head_weight_abs_max": weight_max,
        "keep_head_bias_abs_max": bias_max,
        "strict_keep_rule": "keep_logit > 0",
        "zero_step_behavior": "all conflicts use frozen C-V3 current output",
    }


def _new_target_totals():
    return {
        "valid_pixels": 0,
        "conflict_pixels": 0,
        "supervised_conflict_pixels": 0,
        "keep_target_pixels": 0,
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
    previous_image, _, previous_low, previous_c1, _ = frame1

    # Frozen C-V3 uses Host fallback on the first two frames, matching evaluation.
    previous_cv3_low = previous_low.detach()
    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = None
    controller_hidden = None
    dynamics_state = None

    buffered_losses = []
    frames_in_window = 0
    totals = {
        "frames": 0,
        "supervised_frames": 0,
        "optimizer_steps": 0,
        "controller_bce": 0.0,
        "keep_probability_mean": 0.0,
        "hard_keep_fraction_of_conflicts": 0.0,
        "prediction_error_abs": 0.0,
        "dynamics_error_abs": 0.0,
    }
    target_totals = _new_target_totals()

    for frame_index in range(2, len(samples)):
        current_image, host_logits, host_low, current_c1, output_size = (
            _host_observation(model, samples[frame_index])
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

            memory_row, memory_state, _, _ = _frozen_cv3_step(
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
            current_cv3_low = (
                host_low
                + e1["e1_delta_low"]
                + memory_row["delta_refinement"]
            ).detach()
            history_low, history_valid = warp_low_logits(
                previous_cv3_low,
                pending_motion,
            )
            current_probability = F.softmax(current_cv3_low, dim=1)
            history_probability = F.softmax(history_low, dim=1)
            prediction_error = current_probability - history_probability
            dynamics_state = dynamics.step(
                prediction_error,
                dynamics_state,
            ).detach()
            history_valid_low = history_valid.unsqueeze(1).float()

        controller_row = controller(
            current_probability.detach(),
            history_probability.detach(),
            prediction_error.detach(),
            dynamics_state.detach(),
            e1["transportability_low"].detach(),
            memory_row["memory_reliability"].detach(),
            history_valid_low.detach(),
            controller_hidden,
        )
        controller_hidden = controller_row["hidden"]

        with torch.no_grad():
            teacher_full = raft.current_to_previous(current_image, previous_image)
            keep_target, supervised, conflict, target_diag = _decision_target(
                current_probability,
                history_probability,
                history_valid_low,
                previous_cv3_low,
                current_gt,
                teacher_full,
            )
            _add_target_totals(target_totals, target_diag)

        loss = _masked_bce(
            controller_row["keep_logit"],
            keep_target,
            supervised,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite C-V4 controller BCE")
        if bool(supervised.any()):
            buffered_losses.append(loss)
            totals["supervised_frames"] += 1
            totals["controller_bce"] += float(loss.detach().item())

        with torch.no_grad():
            keep_probability = torch.sigmoid(controller_row["keep_logit"])[0, 0]
            hard_keep = (
                (controller_row["keep_logit"][0, 0] > 0)
                & conflict
            )
            conflict_count = int(conflict.sum().item())
            totals["keep_probability_mean"] += float(
                keep_probability[conflict].mean().item()
                if conflict_count else 0.0
            )
            totals["hard_keep_fraction_of_conflicts"] += float(
                hard_keep.sum().item() / max(conflict_count, 1)
            )
            totals["prediction_error_abs"] += float(
                prediction_error.abs().mean().item()
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
            if buffered_losses:
                window_loss = torch.stack(buffered_losses).mean()
                optimizer.zero_grad(set_to_none=True)
                window_loss.backward()
                optimizer.step()
                totals["optimizer_steps"] += 1
            buffered_losses = []
            frames_in_window = 0
            if controller_hidden is not None:
                controller_hidden = controller_hidden.detach()
            if memory_state is not None:
                memory_state = memory_state.detach()
            dynamics_state = dynamics_state.detach()

        previous_image = current_image
        previous_low = host_low.detach()
        previous_c1 = current_c1.detach()
        previous_cv3_low = current_cv3_low.detach()  # never Controller output
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()

    frames = max(totals["frames"], 1)
    supervised_frames = max(totals["supervised_frames"], 1)
    totals["controller_bce"] /= supervised_frames
    for key in (
        "keep_probability_mean",
        "hard_keep_fraction_of_conflicts",
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

    numeric_keys = (
        "frames",
        "supervised_frames",
        "optimizer_steps",
        "controller_bce",
        "keep_probability_mean",
        "hard_keep_fraction_of_conflicts",
        "prediction_error_abs",
        "dynamics_error_abs",
    )
    averaged = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in numeric_keys
    }
    averaged["targets"] = _target_rates(target_totals)
    return averaged


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
        "conflict_fraction": 0.0,
        "hard_keep_fraction_of_conflicts": 0.0,
        "keep_probability_on_conflicts": 0.0,
        "prediction_error_abs": 0.0,
        "dynamics_error_abs": 0.0,
        "history_valid_fraction": 0.0,
    }

    for sequence in FULL9:
        previous = None
        previous_cv3_low = None
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

            if previous is None:
                e1_pred = c_v3_pred = hysteresis_pred = oracle_pred = host_pred
                semantic_state_low = torch.zeros_like(host_low)
                previous_cv3_low = host_low.detach()
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
                e1_pred = c_v3_pred = hysteresis_pred = oracle_pred = host_pred
                previous_cv3_low = host_low.detach()
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
                current_cv3_low = (
                    host_low
                    + e1["e1_delta_low"]
                    + memory_row["delta_refinement"]
                ).detach()
                history_low, history_valid = warp_low_logits(
                    previous_cv3_low,
                    pending_motion,
                )
                current_probability = F.softmax(current_cv3_low, dim=1)
                history_probability = F.softmax(history_low, dim=1)
                prediction_error = current_probability - history_probability
                dynamics_state = dynamics.step(
                    prediction_error,
                    dynamics_state,
                ).detach()
                history_valid_low = history_valid.unsqueeze(1).float()

                controller_row = controller(
                    current_probability,
                    history_probability,
                    prediction_error,
                    dynamics_state,
                    e1["transportability_low"],
                    memory_row["memory_reliability"],
                    history_valid_low,
                    controller_hidden,
                )
                controller_hidden = controller_row["hidden"]

                current_pred_low = current_probability.argmax(dim=1)
                history_pred_low = history_probability.argmax(dim=1)
                conflict = (
                    (current_pred_low != history_pred_low)
                    & history_valid
                )[0]
                keep_low = (
                    (controller_row["keep_logit"][0, 0] > 0)
                    & conflict
                )

                e1_pred = e1_logits.argmax(1)
                c_v3_pred = c_v3_logits.argmax(1)
                hysteresis_pred = _apply_keep_prediction(
                    c_v3_pred,
                    history_pred_low,
                    keep_low,
                    output_size,
                )

                teacher_full = raft.current_to_previous(image, previous_image)
                oracle_keep, _, _, target_diag = _decision_target(
                    current_probability,
                    history_probability,
                    history_valid_low,
                    previous_cv3_low,
                    gt_cpu,
                    teacher_full,
                )
                oracle_pred = _apply_keep_prediction(
                    c_v3_pred,
                    history_pred_low,
                    oracle_keep,
                    output_size,
                )
                _add_target_totals(target_totals, target_diag)

                conflict_count = int(conflict.sum().item())
                diagnostics["decision_frames"] += 1
                diagnostics["conflict_fraction"] += float(
                    conflict.float().mean().item()
                )
                diagnostics["hard_keep_fraction_of_conflicts"] += float(
                    keep_low.sum().item() / max(conflict_count, 1)
                )
                keep_probability = torch.sigmoid(
                    controller_row["keep_logit"][0, 0]
                )
                diagnostics["keep_probability_on_conflicts"] += float(
                    keep_probability[conflict].mean().item()
                    if conflict_count else 0.0
                )
                diagnostics["prediction_error_abs"] += float(
                    prediction_error.abs().mean().item()
                )
                diagnostics["dynamics_error_abs"] += float(
                    dynamics_state.abs().mean().item()
                )
                diagnostics["history_valid_fraction"] += float(
                    history_valid.float().mean().item()
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
                previous_cv3_low = current_cv3_low.detach()  # never hysteresis output
                previous = (image, host_low.detach(), current_c1.detach())

            predictions = {
                "host": host_pred,
                "e1_base": e1_pred,
                "c_v3_base": c_v3_pred,
                "hysteresis": hysteresis_pred,
                "semantic_temporal_oracle": oracle_pred,
            }
            for name, prediction in predictions.items():
                pred_cpu = prediction.squeeze(0).cpu()
                update_confusion_matrix(confusion[name], pred_cpu, gt_cpu)
                seq_vc[name].update(gt_cpu, pred_cpu)

            if previous_image_for_mtc is not None:
                teacher_full = raft.current_to_previous(image, previous_image_for_mtc)
                for name, prediction in predictions.items():
                    score = _pair_mtc(
                        previous_predictions[name], prediction, teacher_full
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
            "mTC": mtc_sum[name] / mtc_count[name] if mtc_count[name] else float("nan"),
            "mVC8": vc_sum[name][8] / vc_count[name][8] if vc_count[name][8] else float("nan"),
            "mVC16": vc_sum[name][16] / vc_count[name][16] if vc_count[name][16] else float("nan"),
        }
        for name in CANDIDATES
    }

    decision_frames = max(diagnostics["decision_frames"], 1)
    for key in (
        "conflict_fraction",
        "hard_keep_fraction_of_conflicts",
        "keep_probability_on_conflicts",
        "prediction_error_abs",
        "dynamics_error_abs",
        "history_valid_fraction",
    ):
        diagnostics[key] /= decision_frames
    diagnostics["target_distribution"] = _target_rates(target_totals)
    diagnostics["semantic_history_source"] = "previous frozen C-V3 base only"
    diagnostics["controller_output_feedback"] = False
    diagnostics["controller_uses_c1"] = False
    diagnostics["oracle_interpretation"] = (
        "semantic correctness first; temporal consistency only when both candidates "
        "are GT-wrong. This policy is diagnostic, not a strict mTC upper bound."
    )
    return metrics, diagnostics


def _delta_metrics(candidate, reference):
    return {
        key: candidate[key] - reference[key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }


def _selection_key(metrics):
    base = metrics["c_v3_base"]
    candidate = metrics["hysteresis"]
    miou_preserved = candidate["mIoU"] >= base["mIoU"]
    if miou_preserved:
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
                "hysteresis": _delta_metrics(
                    metrics["hysteresis"], metrics["c_v3_base"]
                ),
                "semantic_temporal_oracle": _delta_metrics(
                    metrics["semantic_temporal_oracle"], metrics["c_v3_base"]
                ),
            },
            "delta_vs_host": {
                "c_v3_base": _delta_metrics(
                    metrics["c_v3_base"], metrics["host"]
                ),
                "hysteresis": _delta_metrics(
                    metrics["hysteresis"], metrics["host"]
                ),
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
                    "experiment": "c_v4_stateful_semantic_hysteresis",
                    "epoch": epoch,
                    "controller_state_dict": controller.state_dict(),
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "train": train_stats,
                    "dynamics": dynamics.config(),
                    "architecture": {
                        "controller_hidden_channels": CONTROLLER_HIDDEN_CHANNELS,
                        "controller_uses_c1": False,
                        "semantic_history_source": "previous frozen C-V3 base",
                        "controller_output_feedback": False,
                        "decision_scope": "current/history argmax conflicts only",
                        "hard_inference_rule": "keep_logit > 0",
                        "target_priority": "semantic correctness then temporal consistency",
                    },
                },
                output / "best.pt",
            )

        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V4 produced no epoch result")

    best_metrics = best["metrics"]
    c_v3_base = best_metrics["c_v3_base"]
    candidate = best_metrics["hysteresis"]
    summary = {
        "experiment": "C-V4 Stateful Semantic Hysteresis",
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
            "semantic_history_source": "previous frozen C-V3 base only",
            "controller_output_feedback": False,
            "prediction_error": "P_current_CV3 - P_warped_previous_CV3",
            "dynamics_error_role": "temporal decision evidence only",
            "decision_scope": "Base/History class conflicts only",
            "training_loss": "ordinary unbalanced BCE on semantic-first temporal-second conflict targets",
            "raft_inference": False,
        },
        "selection_rule": {
            "hard_constraint": "hysteresis mIoU >= frozen C-V3 Base mIoU",
            "objective_after_constraint": "maximize hysteresis mTC, then mIoU",
        },
        "target": {
            "mIoU_floor": c_v3_base["mIoU"],
            "mTC_target_low": MTC_TARGET_LOW,
            "mTC_target_high": MTC_TARGET_HIGH,
            "miou_preserved": candidate["mIoU"] >= c_v3_base["mIoU"],
            "mtc_reached_74": candidate["mTC"] >= MTC_TARGET_LOW,
            "mtc_in_74_76": MTC_TARGET_LOW <= candidate["mTC"] <= MTC_TARGET_HIGH,
        },
        "tbptt_steps": args.tbptt_steps,
        "epochs": args.epochs,
    }
    with (result_output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
