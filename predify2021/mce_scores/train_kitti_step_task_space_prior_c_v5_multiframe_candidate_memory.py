"""C-V5: Non-Autoregressive Multi-Frame Semantic Candidate Memory.

中文：非自回归多帧语义候选记忆。

结构原则：
1. History Memory（历史记忆）只保存过去冻结 C-V3 的 raw logits（原始分类得分）。
2. Controller output（控制器输出）绝不回灌，避免错误自我强化。
3. 多帧 motion（运动）只累计 flow，不反复 warp logits；每个 raw history 到当前帧只做一次最终采样。
4. Prediction Error（预测误差）与 Dynamics Error（动力学误差）只由 t-1 历史定义，深历史永远不改变误差基准。
5. 训练使用像素级 direct target（直接目标）：Semantic correctness first（语义正确性优先），
   Temporal consistency second（时序一致性次级）。
6. 默认 K=4，不提供 history-length sweep（历史长度扫描）。

该脚本启动后先输出 Multi-frame Oracle（多帧 Oracle）结果。使用 --oracle-only
可只完成结构前提诊断，不训练 Controller。
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
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v4_stateful_semantic_hysteresis_main import (
    _upsample_predicted_flow,
    _warp_full_logits_zero_invalid,
    _warp_low_state_zero_invalid,
    _warp_previous_prediction_with_raft,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT, flow_grid
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    low_flow_grid,
    warp_low_logits,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_multiframe_candidate_memory import (
    MultiFrameSemanticSelector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)

SEED = 0
EPOCHS = 3
TBPTT_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
CONTROLLER_HIDDEN_CHANNELS = 32
HISTORY_LENGTH = 4

DYNAMICS_TAU_E = 4.0
DYNAMICS_K_E = 1.0
DYNAMICS_DT = 1.0

MIOU_REFERENCE = 0.662706834793679
MTC_TARGET_LOW = 0.74
MTC_TARGET_HIGH = 0.76
SINGLE_FRAME_ORACLE_MTC = 0.724701040652116

OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v5_multiframe_candidate_memory"
)
RESULT_DEFAULT = (
    "results/kitti_step_task_space_prior_c_v5_multiframe_candidate_memory"
)
CANDIDATES = (
    "host",
    "e1_base",
    "c_v3_base",
    "c_v5",
    "multiframe_semantic_oracle",
    "multiframe_semantic_temporal_oracle",
)


def _extend_backward_flow(accumulated_flow, accumulated_valid, next_older_flow):
    """Compose current->mid with mid->older without resampling semantic logits.

    中文：只累计 backward flow（反向运动场），不重复采样语义 logits。
    """
    grid, _ = low_flow_grid(accumulated_flow)
    sampled_next = F.grid_sample(
        next_older_flow.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).to(accumulated_flow.dtype)

    _, next_valid_mid = low_flow_grid(next_older_flow)
    next_valid_current = F.grid_sample(
        next_valid_mid.float().unsqueeze(1),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0] > 0.5
    valid = accumulated_valid.bool() & next_valid_current
    composed = accumulated_flow + sampled_next
    composed = composed * valid.unsqueeze(1).to(composed.dtype)
    return composed, valid


def _accumulated_flow_for_age(pending_motion, motion_history, age):
    """Return current->t-age accumulated flow（累计运动） and low validity."""
    if age < 1:
        raise ValueError("age must be >= 1")
    flow = pending_motion
    _, valid = low_flow_grid(flow)
    for index in range(age - 1):
        flow, valid = _extend_backward_flow(
            flow,
            valid,
            motion_history[index],
        )
    return flow, valid


def _build_history_candidates(
    raw_history,
    motion_history,
    pending_motion,
    history_length,
):
    """Warp each frozen raw C-V3 history exactly once to the current frame."""
    rows = []
    available = min(len(raw_history), history_length)
    for age in range(1, available + 1):
        if age > 1 and len(motion_history) < age - 1:
            break
        accumulated_flow, accumulated_valid_low = _accumulated_flow_for_age(
            pending_motion,
            motion_history,
            age,
        )
        warped_logits, valid_full = _warp_full_logits_zero_invalid(
            raw_history[age - 1],
            accumulated_flow,
        )
        low_valid_full = F.interpolate(
            accumulated_valid_low.float().unsqueeze(1),
            size=tuple(valid_full.shape[-2:]),
            mode="nearest",
        )[:, 0] > 0.5
        valid_full = valid_full.bool() & low_valid_full
        warped_logits = warped_logits * valid_full.unsqueeze(1).to(warped_logits.dtype)
        rows.append(
            {
                "age": age,
                "logits": warped_logits,
                "valid_full": valid_full,
                "valid_low_path": accumulated_valid_low.bool(),
                "flow": accumulated_flow,
            }
        )
    return rows


def _pad_history_for_controller(current_probability, candidate_rows, history_length):
    probabilities = []
    validities = []
    low_size = tuple(current_probability.shape[-2:])
    for index in range(history_length):
        if index < len(candidate_rows):
            row = candidate_rows[index]
            low_logits = F.interpolate(
                row["logits"].detach(),
                size=low_size,
                mode="bilinear",
                align_corners=False,
            )
            probabilities.append(F.softmax(low_logits, dim=1))
            path_valid = row["valid_low_path"].float().unsqueeze(1)
            if path_valid.shape[-2:] != low_size:
                path_valid = F.interpolate(path_valid, size=low_size, mode="nearest")
            validities.append(path_valid.to(current_probability.dtype))
        else:
            probabilities.append(current_probability)
            validities.append(
                torch.zeros_like(current_probability[:, :1])
            )
    return probabilities, validities


def _selector_evidence(
    selector,
    dynamics,
    c_v3_logits,
    candidate_rows,
    pending_motion,
    transportability_low,
    memory_reliability_low,
    selector_hidden,
    dynamics_state,
):
    low_size = tuple(transportability_low.shape[-2:])
    current_low = F.interpolate(
        c_v3_logits.detach(),
        size=low_size,
        mode="bilinear",
        align_corners=False,
    )
    current_probability = F.softmax(current_low, dim=1)
    history_probabilities, history_validities = _pad_history_for_controller(
        current_probability,
        candidate_rows,
        selector.history_length,
    )

    history1_probability = history_probabilities[0]
    history1_valid = history_validities[0]
    prediction_error = (
        current_probability - history1_probability
    ) * history1_valid

    if dynamics_state is None:
        warped_dynamics = None
    else:
        warped_dynamics, _ = _warp_low_state_zero_invalid(
            dynamics_state,
            pending_motion,
        )
        warped_dynamics = (
            warped_dynamics
            * history1_valid
            * transportability_low.detach().clamp(0.0, 1.0)
        )

    dynamics_state = dynamics.step(
        prediction_error,
        warped_dynamics,
    ).detach()

    row = selector(
        current_probability.detach(),
        [probability.detach() for probability in history_probabilities],
        prediction_error.detach(),
        dynamics_state.detach(),
        transportability_low.detach(),
        memory_reliability_low.detach(),
        [validity.detach() for validity in history_validities],
        selector_hidden,
    )

    selector_logits_full = F.interpolate(
        row["selector_logits"],
        size=tuple(c_v3_logits.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )
    for index in range(selector.history_length):
        channel = index + 1
        if index < len(candidate_rows):
            valid = candidate_rows[index]["valid_full"]
            selector_logits_full[:, channel][~valid] = -1e4
        else:
            selector_logits_full[:, channel].fill_(-1e4)

    return {
        "row": row,
        "selector_logits_full": selector_logits_full,
        "prediction_error": prediction_error,
        "dynamics_state": dynamics_state,
        "history_probabilities": history_probabilities,
        "history_validities": history_validities,
    }


def _candidate_predictions(current_logits, candidate_rows):
    current_pred = current_logits.argmax(1)
    history_preds = [row["logits"].argmax(1) for row in candidate_rows]
    return current_pred, history_preds


def _apply_selection(current_pred, history_preds, selection):
    output = current_pred.clone()
    for index, history_pred in enumerate(history_preds, start=1):
        mask = selection == index
        output[0][mask] = history_pred[0][mask]
    return output


def _build_multiframe_target(
    current_logits,
    candidate_rows,
    previous_cv3_logits,
    current_gt_cpu,
    teacher_full,
):
    """Build direct K+1 class target（直接 K+1 类目标） at full resolution."""
    current_pred, history_preds = _candidate_predictions(
        current_logits,
        candidate_rows,
    )
    gt = current_gt_cpu.to(current_logits.device, non_blocking=True)
    valid_gt = gt != IGNORE_LABEL

    height, width = gt.shape[-2:]
    semantic_target = torch.zeros((height, width), dtype=torch.long, device=gt.device)
    temporal_target = semantic_target.clone()

    decision = torch.zeros_like(valid_gt)
    for row, history_pred in zip(candidate_rows, history_preds):
        decision |= row["valid_full"][0] & (history_pred[0] != current_pred[0])
    decision &= valid_gt

    current_correct = current_pred[0] == gt
    semantic_supervised = decision & current_correct

    rescue_by_age = [0 for _ in candidate_rows]
    unresolved = decision & ~current_correct
    for index, (row, history_pred) in enumerate(zip(candidate_rows, history_preds)):
        rescue = unresolved & row["valid_full"][0] & (history_pred[0] == gt)
        semantic_target[rescue] = index + 1
        temporal_target[rescue] = index + 1
        semantic_supervised |= rescue
        rescue_by_age[index] = int(rescue.sum().item())
        unresolved &= ~rescue

    temporal_supervised = semantic_supervised.clone()
    temporal_history_by_age = [0 for _ in candidate_rows]
    temporal_current_pixels = 0

    if teacher_full is not None and previous_cv3_logits is not None:
        teacher_previous, teacher_valid = _warp_previous_prediction_with_raft(
            previous_cv3_logits,
            teacher_full,
        )
        teacher_previous = teacher_previous[0]
        teacher_valid = teacher_valid[0].bool()
        temporal_unresolved = unresolved & teacher_valid

        current_match = temporal_unresolved & (current_pred[0] == teacher_previous)
        temporal_target[current_match] = 0
        temporal_supervised |= current_match
        temporal_current_pixels = int(current_match.sum().item())
        temporal_unresolved &= ~current_match

        for index, (row, history_pred) in enumerate(zip(candidate_rows, history_preds)):
            choose = (
                temporal_unresolved
                & row["valid_full"][0]
                & (history_pred[0] == teacher_previous)
            )
            temporal_target[choose] = index + 1
            temporal_supervised |= choose
            temporal_history_by_age[index] = int(choose.sum().item())
            temporal_unresolved &= ~choose

    row = {
        "valid_pixels": int(valid_gt.sum().item()),
        "decision_pixels": int(decision.sum().item()),
        "semantic_supervised_pixels": int(semantic_supervised.sum().item()),
        "temporal_supervised_pixels": int(temporal_supervised.sum().item()),
        "current_correct_decision_pixels": int((decision & current_correct).sum().item()),
        "semantic_rescue_pixels": int(sum(rescue_by_age)),
        "semantic_rescue_by_age": rescue_by_age,
        "temporal_current_pixels": temporal_current_pixels,
        "temporal_history_by_age": temporal_history_by_age,
        "unsupervised_decision_pixels": int((decision & ~temporal_supervised).sum().item()),
    }
    return (
        semantic_target,
        temporal_target,
        temporal_supervised,
        decision,
        row,
    )


def _new_target_totals(history_length):
    return {
        "valid_pixels": 0,
        "decision_pixels": 0,
        "semantic_supervised_pixels": 0,
        "temporal_supervised_pixels": 0,
        "current_correct_decision_pixels": 0,
        "semantic_rescue_pixels": 0,
        "semantic_rescue_by_age": [0] * history_length,
        "temporal_current_pixels": 0,
        "temporal_history_by_age": [0] * history_length,
        "unsupervised_decision_pixels": 0,
    }


def _add_target_totals(total, row):
    for key in (
        "valid_pixels",
        "decision_pixels",
        "semantic_supervised_pixels",
        "temporal_supervised_pixels",
        "current_correct_decision_pixels",
        "semantic_rescue_pixels",
        "temporal_current_pixels",
        "unsupervised_decision_pixels",
    ):
        total[key] += int(row[key])
    for index, value in enumerate(row["semantic_rescue_by_age"]):
        total["semantic_rescue_by_age"][index] += int(value)
    for index, value in enumerate(row["temporal_history_by_age"]):
        total["temporal_history_by_age"][index] += int(value)


def _target_rates(total):
    valid = max(total["valid_pixels"], 1)
    decision = max(total["decision_pixels"], 1)
    return {
        **total,
        "decision_fraction_of_valid": total["decision_pixels"] / valid,
        "semantic_rescue_fraction_of_decisions": total["semantic_rescue_pixels"] / decision,
        "semantic_supervised_fraction_of_decisions": total["semantic_supervised_pixels"] / decision,
        "temporal_supervised_fraction_of_decisions": total["temporal_supervised_pixels"] / decision,
    }


def _zero_step_check(selector):
    weight_max = float(selector.selector_head.weight.detach().abs().max().item())
    bias_max = float(selector.selector_head.bias.detach().abs().max().item())
    if weight_max != 0.0 or bias_max != 0.0:
        raise RuntimeError("C-V5 selector head must be exactly zero initialized")
    return {
        "selector_head_weight_abs_max": weight_max,
        "selector_head_bias_abs_max": bias_max,
        "zero_step_argmax_index": 0,
        "zero_step_behavior": "all valid pixels use frozen C-V3 Current",
    }


def _train_sequence(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    selector,
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
        observer,
        residual,
        frame0,
        frame1,
    )
    previous_image, previous_host_logits, previous_low, previous_c1, _ = frame1

    raw_history = [previous_host_logits.detach()]
    motion_history = []
    previous_cv3_logits = previous_host_logits.detach()

    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = None
    selector_hidden = None
    dynamics_state = None

    loss_sums = []
    loss_counts = []
    frames_in_window = 0
    totals = {
        "frames": 0,
        "supervised_frames": 0,
        "optimizer_steps": 0,
        "supervised_pixels": 0,
        "selector_ce_per_pixel": 0.0,
        "prediction_error_abs": 0.0,
        "dynamics_error_abs": 0.0,
        "selected_candidate_counts": [0] * (selector.history_length + 1),
    }
    target_totals = _new_target_totals(selector.history_length)

    for frame_index in range(2, len(samples)):
        current_image, host_logits, host_low, current_c1, output_size = _host_observation(
            model,
            samples[frame_index],
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

            memory_row, memory_state, _, c_v3_logits = _frozen_cv3_step(
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
            candidate_rows = _build_history_candidates(
                raw_history,
                motion_history,
                pending_motion,
                selector.history_length,
            )

        evidence = _selector_evidence(
            selector,
            dynamics,
            c_v3_logits,
            candidate_rows,
            pending_motion,
            e1["transportability_low"],
            memory_row["memory_reliability"],
            selector_hidden,
            dynamics_state,
        )
        selector_hidden = evidence["row"]["hidden"]
        dynamics_state = evidence["dynamics_state"]

        with torch.no_grad():
            teacher_full = raft.current_to_previous(current_image, previous_image)
            (
                _,
                temporal_target,
                supervised,
                _,
                target_diag,
            ) = _build_multiframe_target(
                c_v3_logits,
                candidate_rows,
                previous_cv3_logits,
                current_gt,
                teacher_full,
            )
            _add_target_totals(target_totals, target_diag)

        selected = supervised
        if bool(selected.any()):
            logits = evidence["selector_logits_full"][0].permute(1, 2, 0)[selected]
            target = temporal_target[selected]
            pixel_loss_sum = F.cross_entropy(logits, target, reduction="sum")
            count = int(selected.sum().item())
            loss_sums.append(pixel_loss_sum)
            loss_counts.append(count)
            totals["supervised_frames"] += 1
            totals["supervised_pixels"] += count
            totals["selector_ce_per_pixel"] += float(pixel_loss_sum.detach().item())

        with torch.no_grad():
            hard_selection = evidence["selector_logits_full"].argmax(1)[0]
            for index in range(selector.history_length + 1):
                totals["selected_candidate_counts"][index] += int(
                    (hard_selection == index).sum().item()
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
            if selector_hidden is not None:
                selector_hidden = selector_hidden.detach()
            if memory_state is not None:
                memory_state = memory_state.detach()
            if dynamics_state is not None:
                dynamics_state = dynamics_state.detach()

        raw_history.insert(0, c_v3_logits.detach())
        raw_history = raw_history[: selector.history_length]
        motion_history.insert(0, pending_motion.detach())
        motion_history = motion_history[: max(selector.history_length - 1, 0)]

        previous_image = current_image
        previous_low = host_low.detach()
        previous_c1 = current_c1.detach()
        previous_cv3_logits = c_v3_logits.detach()
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()

    frames = max(totals["frames"], 1)
    totals["selector_ce_per_pixel"] /= max(totals["supervised_pixels"], 1)
    totals["prediction_error_abs"] /= frames
    totals["dynamics_error_abs"] /= frames
    totals["targets"] = _target_rates(target_totals)
    return totals


def _train_epoch(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    selector,
    dynamics,
    raft,
    groups,
    optimizer,
    tbptt_steps,
):
    selector.train()
    refiner.eval()
    correction.eval()
    mask_predictor.eval()
    rows = []
    target_totals = _new_target_totals(selector.history_length)

    for samples in groups.values():
        row = _train_sequence(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            selector,
            dynamics,
            raft,
            samples,
            optimizer,
            tbptt_steps,
        )
        if row is not None:
            rows.append(row)
            _add_target_totals(target_totals, row["targets"])

    if not rows:
        raise RuntimeError("No valid C-V5 training sequences")

    frame_total = max(sum(row["frames"] for row in rows), 1)
    pixel_total = max(sum(row["supervised_pixels"] for row in rows), 1)
    result = {
        "frames": sum(row["frames"] for row in rows),
        "supervised_frames": sum(row["supervised_frames"] for row in rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "supervised_pixels": sum(row["supervised_pixels"] for row in rows),
        "selector_ce_per_pixel": sum(
            row["selector_ce_per_pixel"] * row["supervised_pixels"]
            for row in rows
        ) / pixel_total,
        "prediction_error_abs": sum(
            row["prediction_error_abs"] * row["frames"] for row in rows
        ) / frame_total,
        "dynamics_error_abs": sum(
            row["dynamics_error_abs"] * row["frames"] for row in rows
        ) / frame_total,
        "selected_candidate_counts": [
            sum(row["selected_candidate_counts"][index] for row in rows)
            for index in range(selector.history_length + 1)
        ],
        "targets": _target_rates(target_totals),
    }
    return result


@torch.inference_mode()
def _evaluate(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    selector,
    dynamics,
    groups,
    raft,
):
    selector.eval()
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
    target_totals = _new_target_totals(selector.history_length)
    selected_counts = [0] * (selector.history_length + 1)

    for sequence in FULL9:
        previous = None
        previous_cv3_logits = None
        pending_motion = motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = None
        memory_state = None
        selector_hidden = None
        dynamics_state = None
        raw_history = []
        motion_history = []
        previous_predictions = {}
        seq_vc = {name: VideoConsistency() for name in CANDIDATES}

        for sample in groups[sequence]:
            image, host_logits, host_low, current_c1, output_size = _host_observation(
                model,
                sample,
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
                e1_pred = c_v3_pred = c_v5_pred = host_pred
                semantic_oracle_pred = temporal_oracle_pred = host_pred
                semantic_state_low = torch.zeros_like(host_low)
                previous_cv3_logits = host_logits.detach()
                raw_history = [host_logits.detach()]
                previous = (image, host_low.detach(), current_c1.detach())
            elif pending_motion is None:
                previous_image, previous_low, previous_c1 = previous
                observed = _observe_motion(
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
                e1_pred = c_v3_pred = c_v5_pred = host_pred
                semantic_oracle_pred = temporal_oracle_pred = host_pred
                previous_cv3_logits = host_logits.detach()
                raw_history = [host_logits.detach()]
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
                candidate_rows = _build_history_candidates(
                    raw_history,
                    motion_history,
                    pending_motion,
                    selector.history_length,
                )
                evidence = _selector_evidence(
                    selector,
                    dynamics,
                    c_v3_logits,
                    candidate_rows,
                    pending_motion,
                    e1["transportability_low"],
                    memory_row["memory_reliability"],
                    selector_hidden,
                    dynamics_state,
                )
                selector_hidden = evidence["row"]["hidden"]
                dynamics_state = evidence["dynamics_state"]

                (
                    semantic_target,
                    temporal_target,
                    _,
                    _,
                    target_diag,
                ) = _build_multiframe_target(
                    c_v3_logits,
                    candidate_rows,
                    previous_cv3_logits,
                    gt_cpu,
                    teacher_full,
                )
                _add_target_totals(target_totals, target_diag)

                current_pred, history_preds = _candidate_predictions(
                    c_v3_logits,
                    candidate_rows,
                )
                hard_selection = evidence["selector_logits_full"].argmax(1)[0]
                for index in range(selector.history_length + 1):
                    selected_counts[index] += int((hard_selection == index).sum().item())

                e1_pred = e1_logits.argmax(1)
                c_v3_pred = current_pred
                c_v5_pred = _apply_selection(current_pred, history_preds, hard_selection)
                semantic_oracle_pred = _apply_selection(
                    current_pred,
                    history_preds,
                    semantic_target,
                )
                temporal_oracle_pred = _apply_selection(
                    current_pred,
                    history_preds,
                    temporal_target,
                )

                observed = _observe_motion(
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
                raw_history = raw_history[: selector.history_length]
                motion_history.insert(0, pending_motion.detach())
                motion_history = motion_history[: max(selector.history_length - 1, 0)]
                previous_cv3_logits = c_v3_logits.detach()
                previous = (image, host_low.detach(), current_c1.detach())
                pending_motion = next_motion.detach()
                motion_hidden = next_motion_hidden.detach()

            predictions = {
                "host": host_pred,
                "e1_base": e1_pred,
                "c_v3_base": c_v3_pred,
                "c_v5": c_v5_pred,
                "multiframe_semantic_oracle": semantic_oracle_pred,
                "multiframe_semantic_temporal_oracle": temporal_oracle_pred,
            }
            for name, prediction in predictions.items():
                pred_cpu = prediction.squeeze(0).cpu()
                update_confusion_matrix(confusion[name], pred_cpu, gt_cpu)
                seq_vc[name].update(gt_cpu, pred_cpu)

            if teacher_full is not None:
                for name, prediction in predictions.items():
                    if name in previous_predictions:
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
            "mTC": mtc_sum[name] / mtc_count[name] if mtc_count[name] else float("nan"),
            "mVC8": vc_sum[name][8] / vc_count[name][8] if vc_count[name][8] else float("nan"),
            "mVC16": vc_sum[name][16] / vc_count[name][16] if vc_count[name][16] else float("nan"),
        }
        for name in CANDIDATES
    }
    diagnostics = {
        "target_distribution": _target_rates(target_totals),
        "selected_candidate_counts": selected_counts,
        "history_length": selector.history_length,
        "semantic_history_source": "detached raw frozen C-V3 logits from t-1..t-K",
        "history_logits_resampling": "one final warp per candidate",
        "controller_output_feedback": False,
        "prediction_error_reference": "t-1 frozen C-V3 only",
        "dynamics_error_reference": "t-1 prediction error only",
        "dynamics_motion_compensated": True,
        "dynamics_transportability_masked": True,
        "raft_inference": False,
    }
    return metrics, diagnostics


def _delta_metrics(candidate, reference):
    return {
        key: candidate[key] - reference[key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }


def _selection_key(metrics):
    base = metrics["c_v3_base"]
    candidate = metrics["c_v5"]
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
    parser.add_argument("--oracle-only", action="store_true")
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
    residual, residual_payload = _load_frozen_residual(
        args.residual_checkpoint,
        observer,
    )
    correction, mask_predictor, base_payload = _load_frozen_e1_base(
        args.base_checkpoint,
        observer,
    )
    refiner, cv3_payload = _load_frozen_cv3_refiner(args.c_v3_checkpoint)

    selector = MultiFrameSemanticSelector(
        num_classes=NUM_CLASSES,
        history_length=HISTORY_LENGTH,
        hidden_channels=CONTROLLER_HIDDEN_CHANNELS,
    ).cuda()
    dynamics = EulerDynamicsError(
        tau_e=args.dynamics_tau_e,
        k_e=args.dynamics_k_e,
        dt=args.dynamics_dt,
    )
    zero_step = _zero_step_check(selector)

    optimizer = torch.optim.AdamW(
        selector.parameters(),
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

    oracle_metrics, oracle_diagnostics = _evaluate(
        model,
        observer,
        residual,
        correction,
        mask_predictor,
        refiner,
        selector,
        dynamics,
        val_groups,
        raft,
    )
    oracle_precheck = {
        "metrics": oracle_metrics,
        "diagnostics": oracle_diagnostics,
        "delta_multiframe_oracle_vs_c_v3": _delta_metrics(
            oracle_metrics["multiframe_semantic_temporal_oracle"],
            oracle_metrics["c_v3_base"],
        ),
        "single_frame_semantic_temporal_oracle_mTC": SINGLE_FRAME_ORACLE_MTC,
        "multiframe_oracle_exceeds_single_frame_oracle_mTC": (
            oracle_metrics["multiframe_semantic_temporal_oracle"]["mTC"]
            > SINGLE_FRAME_ORACLE_MTC
        ),
    }
    with (result_output / "oracle_precheck.json").open("w") as handle:
        json.dump(oracle_precheck, handle, indent=2)
    print(json.dumps({"oracle_precheck": oracle_precheck}, indent=2), flush=True)

    if args.oracle_only:
        return

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
            selector,
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
            selector,
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
                    "c_v5",
                    "multiframe_semantic_oracle",
                    "multiframe_semantic_temporal_oracle",
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
                    "experiment": "c_v5_multiframe_candidate_memory",
                    "epoch": epoch,
                    "selector_state_dict": selector.state_dict(),
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "train": train_stats,
                    "dynamics": dynamics.config(),
                    "architecture": {
                        "history_length": HISTORY_LENGTH,
                        "history_source": "raw detached frozen C-V3 logits",
                        "history_logits_resampling": "one final warp per candidate",
                        "controller_output_feedback": False,
                        "prediction_error_reference": "t-1 frozen C-V3 only",
                        "selector_hidden_channels": CONTROLLER_HIDDEN_CHANNELS,
                        "target_priority": "semantic correctness then temporal consistency",
                    },
                },
                output / "best.pt",
            )
        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V5 produced no epoch result")

    base_metrics = best["metrics"]["c_v3_base"]
    candidate = best["metrics"]["c_v5"]
    summary = {
        "experiment": "C-V5 Non-Autoregressive Multi-Frame Semantic Candidate Memory",
        "oracle_precheck": oracle_precheck,
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
            "history_length": HISTORY_LENGTH,
            "history_source": "raw detached frozen C-V3 logits",
            "history_logits_resampling": "one final warp per candidate",
            "controller_output_feedback": False,
            "prediction_error_reference": "t-1 frozen C-V3 only",
            "dynamics_error_role": "motion-compensated and T-masked temporal decision evidence",
            "decision_resolution": "full output resolution",
            "training_loss": "direct per-pixel K+1 cross entropy on supervised decision pixels",
            "raft_inference": False,
        },
        "selection_rule": {
            "hard_constraint": "C-V5 mIoU >= frozen C-V3 Base mIoU",
            "objective_after_constraint": "maximize C-V5 mTC, then mIoU",
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
