"""C-V7: motion-aligned multi-hypothesis Prediction-Error residual correction.

中文：C-V7 运动对齐多假设预测误差残差修正。

核心变化：
- 删除 C-V6H 的 Current-vs-History / history-age hard selection（当前/历史与历史年龄硬选择）；
- K=4 History（4 帧历史）只用于产生 motion-aligned temporal predictions（运动对齐时序预测）；
- History information（历史信息）必须经过 Prediction Error（预测误差）后才能进入修正路径；
- Error State（误差状态）按 C-V3 Semantic Memory（语义记忆）的范式做运动对齐与可靠性门控；
- 最终只做 bounded logit residual correction（有界分类得分残差修正）；
- 第一版训练只使用 final segmentation CE（最终语义分割交叉熵），不启用额外时序损失或梯度范数平衡。
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
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v4_stateful_semantic_hysteresis_main import (
    _warp_low_state_zero_invalid,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_only import _pair_mtc
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_multihypothesis_error_residual_corrector import (
    MultiHypothesisErrorResidualCorrector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_multihypothesis_error_selector import (
    build_multihypothesis_error_evidence,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v6_error_centric_multihypothesis as c_v6,
)


EXPERIMENT = "c_v7_multihypothesis_error_residual_correction"
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v7_multihypothesis_error_residual_correction"
)
RESULT_DEFAULT = (
    "results/kitti_step_task_space_prior_c_v7_multihypothesis_error_residual_correction"
)
MIOU_HARD_FLOOR = 0.6637739071008685
GATE_MAX = 0.25
ERROR_HIDDEN_CHANNELS = 64
CORRECTION_CHANNELS = 64


def _any_valid_full(candidate_rows, full_size, device):
    """OR of valid full-resolution history support（全分辨率历史有效区域并集）."""
    mask = torch.zeros((1, 1, *full_size), dtype=torch.bool, device=device)
    for row in candidate_rows:
        valid = row["valid_full"].unsqueeze(1).bool()
        if valid.shape[-2:] != tuple(full_size):
            raise ValueError("full-resolution history validity size mismatch")
        mask |= valid
    return mask


def _build_c_v7_evidence(
    corrector,
    dynamics,
    c_v3_logits,
    candidate_rows,
    pending_motion,
    transportability_low,
    memory_reliability_low,
    previous_error_state,
    dynamics_state,
):
    """Build C-V7 correction from Prediction Error only（仅由预测误差构建 C-V7 修正）."""
    low_size = tuple(transportability_low.shape[-2:])

    # Hard gradient boundary（硬梯度边界）: C-V3 and upstream are frozen anchors.
    z_cur = c_v3_logits.detach()
    current_low = F.interpolate(
        z_cur,
        size=low_size,
        mode="bilinear",
        align_corners=False,
    ).detach()
    current_probability = F.softmax(current_low, dim=1).detach()

    history_probabilities, low_path_validities = c_v5._pad_history_for_controller(
        current_probability,
        candidate_rows,
        corrector.history_length,
    )
    history_validities = c_v6._strict_history_validities(
        current_probability,
        candidate_rows,
        low_path_validities,
        corrector.history_length,
    )
    multi = build_multihypothesis_error_evidence(
        current_probability,
        history_probabilities,
        history_validities,
    )
    prediction_errors = [error.detach() for error in multi["prediction_errors"]]
    primary_error = prediction_errors[0]
    history1_valid = history_validities[0].detach()

    # Explicit Dynamics Error（显式动力学误差） remains t-1 based and frozen.
    if dynamics_state is None:
        warped_dynamics = None
    else:
        warped_dynamics, _ = _warp_low_state_zero_invalid(
            dynamics_state,
            pending_motion.detach(),
        )
        warped_dynamics = (
            warped_dynamics
            * history1_valid
            * transportability_low.detach().clamp(0.0, 1.0)
        )
    dynamics_state = dynamics.step(primary_error, warped_dynamics).detach()

    row = corrector(
        prediction_errors=prediction_errors,
        dynamics_error=dynamics_state,
        current_margin=multi["current_margin"].detach(),
        history_margins=[margin.detach() for margin in multi["history_margins"]],
        transportability_low=transportability_low.detach(),
        memory_reliability_low=memory_reliability_low.detach(),
        history_validities_low=[validity.detach() for validity in history_validities],
        backward_motion_low=pending_motion.detach(),
        previous_error_state=previous_error_state,
    )

    # Critical order（关键顺序）:
    # 1) upsample learned bounded correction（先上采样学习到的有界修正）;
    # 2) apply full-resolution validity mask（再乘全分辨率有效性掩码）.
    # Never write a large invalid sentinel before interpolation.
    correction_full = F.interpolate(
        row["correction_low"],
        size=tuple(z_cur.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )
    valid_full = _any_valid_full(candidate_rows, tuple(z_cur.shape[-2:]), z_cur.device)
    correction_full = correction_full * valid_full.to(correction_full.dtype)
    final_logits = z_cur + correction_full

    return {
        "z_cur": z_cur,
        "final_logits": final_logits,
        "correction_full": correction_full,
        "valid_full": valid_full,
        "prediction_errors": prediction_errors,
        "dynamics_state": dynamics_state,
        "row": row,
    }


def _zero_step_check(corrector):
    """E0（零步） must be exactly C-V3 because raw correction is exactly zero."""
    row = {
        "correction_head_weight_abs_max": float(
            corrector.correction_head.weight.detach().abs().max().item()
        ),
        "correction_head_bias_abs_max": float(
            corrector.correction_head.bias.detach().abs().max().item()
        ),
        "gate_head_weight_abs_max": float(
            corrector.gate_head.weight.detach().abs().max().item()
        ),
        "gate_head_bias": float(corrector.gate_head.bias.detach().mean().item()),
    }
    if row["correction_head_weight_abs_max"] != 0.0:
        raise RuntimeError("C-V7 Correction Head weights must be exactly zero initialized")
    if row["correction_head_bias_abs_max"] != 0.0:
        raise RuntimeError("C-V7 Correction Head bias must be exactly zero initialized")
    if row["gate_head_weight_abs_max"] != 0.0:
        raise RuntimeError("C-V7 Gate Head weights must be exactly zero initialized")
    if row["gate_head_bias"] >= 0.0:
        raise RuntimeError("C-V7 Gate Head bias must start negative")
    row["zero_step_behavior"] = (
        "final_logits == detached frozen C-V3 logits because delta_raw == 0 exactly"
    )
    return row


def _train_sequence(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    corrector,
    dynamics,
    samples,
    optimizer,
    tbptt_steps,
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
    _, previous_host_logits, previous_low, previous_c1, _ = frame1

    raw_history = [previous_host_logits.detach()]
    motion_history = []
    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = None
    error_state = None
    dynamics_state = None

    loss_sums = []
    loss_counts = []
    frames_in_window = 0

    totals = {
        "frames": 0,
        "optimizer_steps": 0,
        "valid_pixels": 0,
        "ce_sum": 0.0,
        "correction_abs_sum": 0.0,
        "gate_sum": 0.0,
        "gate_elements": 0,
        "error_state_reliability_sum": 0.0,
    }

    for frame_index in range(2, len(samples)):
        _, host_logits, host_low, current_c1, output_size = c_v5._host_observation(
            model,
            samples[frame_index],
        )
        target = semantic_mask_from_panoptic_png(samples[frame_index]["mask_path"]).to(
            host_logits.device,
            non_blocking=True,
        )

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
                corrector.history_length,
            )

        evidence = _build_c_v7_evidence(
            corrector,
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

        valid_pixels = int((target != c_v5.IGNORE_LABEL).sum().item())
        if valid_pixels > 0:
            ce_sum = F.cross_entropy(
                evidence["final_logits"],
                target.unsqueeze(0),
                ignore_index=c_v5.IGNORE_LABEL,
                reduction="sum",
            )
            if not torch.isfinite(ce_sum):
                raise FloatingPointError("Non-finite C-V7 segmentation CE")
            loss_sums.append(ce_sum)
            loss_counts.append(valid_pixels)
            totals["valid_pixels"] += valid_pixels
            totals["ce_sum"] += float(ce_sum.detach().item())

        with torch.no_grad():
            totals["correction_abs_sum"] += float(
                evidence["correction_full"].abs().mean().item()
            )
            gate = evidence["row"]["gate_low"]
            totals["gate_sum"] += float(gate.sum().item())
            totals["gate_elements"] += int(gate.numel())
            totals["error_state_reliability_sum"] += float(
                evidence["row"]["error_state_reliability"].mean().item()
            )

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
        frames_in_window += 1
        boundary = frames_in_window >= tbptt_steps or frame_index == len(samples) - 1
        if boundary:
            if loss_sums:
                window_loss = torch.stack(loss_sums).sum() / float(max(sum(loss_counts), 1))
                optimizer.zero_grad(set_to_none=True)
                window_loss.backward()
                optimizer.step()
                totals["optimizer_steps"] += 1
            loss_sums = []
            loss_counts = []
            frames_in_window = 0
            if error_state is not None:
                error_state = error_state.detach()
            if memory_state is not None:
                memory_state = memory_state.detach()
            if dynamics_state is not None:
                dynamics_state = dynamics_state.detach()

        raw_history.insert(0, c_v3_logits.detach())
        raw_history = raw_history[: corrector.history_length]
        motion_history.insert(0, pending_motion.detach())
        motion_history = motion_history[: max(corrector.history_length - 1, 0)]
        previous_low = host_low.detach()
        previous_c1 = current_c1.detach()
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()

    frames = max(totals["frames"], 1)
    return {
        "frames": totals["frames"],
        "optimizer_steps": totals["optimizer_steps"],
        "valid_pixels": totals["valid_pixels"],
        "ce_per_pixel": totals["ce_sum"] / float(max(totals["valid_pixels"], 1)),
        "correction_abs_mean": totals["correction_abs_sum"] / float(frames),
        "gate_mean": totals["gate_sum"] / float(max(totals["gate_elements"], 1)),
        "error_state_reliability_mean": totals["error_state_reliability_sum"] / float(frames),
    }


def _train_epoch(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    corrector,
    dynamics,
    groups,
    optimizer,
    tbptt_steps,
):
    corrector.train()
    refiner.eval()
    correction.eval()
    mask_predictor.eval()

    items = list(groups.items())
    random.shuffle(items)
    rows = []
    for _, samples in items:
        row = _train_sequence(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            corrector,
            dynamics,
            samples,
            optimizer,
            tbptt_steps,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("No valid C-V7 training sequences")

    frame_total = max(sum(row["frames"] for row in rows), 1)
    pixel_total = max(sum(row["valid_pixels"] for row in rows), 1)
    return {
        "frames": sum(row["frames"] for row in rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "valid_pixels": sum(row["valid_pixels"] for row in rows),
        "ce_per_pixel": sum(
            row["ce_per_pixel"] * row["valid_pixels"] for row in rows
        ) / float(pixel_total),
        "correction_abs_mean": sum(
            row["correction_abs_mean"] * row["frames"] for row in rows
        ) / float(frame_total),
        "gate_mean": sum(row["gate_mean"] * row["frames"] for row in rows) / float(frame_total),
        "error_state_reliability_mean": sum(
            row["error_state_reliability_mean"] * row["frames"] for row in rows
        ) / float(frame_total),
    }


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

    names = ("host", "c_v3", "c_v7")
    confusion = {
        name: torch.zeros((c_v5.NUM_CLASSES, c_v5.NUM_CLASSES), dtype=torch.int64)
        for name in names
    }
    mtc_sum = {name: 0.0 for name in names}
    mtc_count = {name: 0 for name in names}
    vc_sum = {name: {8: 0.0, 16: 0.0} for name in names}
    vc_count = {name: {8: 0, 16: 0} for name in names}
    correction_abs_sum = 0.0
    gate_sum = 0.0
    gate_elements = 0
    active_frames = 0

    for sequence in c_v5.FULL9:
        samples = groups[sequence]
        previous = None
        pending_motion = None
        motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = None
        memory_state = None
        error_state = None
        dynamics_state = None
        raw_history = []
        motion_history = []
        previous_predictions = {}
        seq_vc = {name: VideoConsistency() for name in names}

        for sample in samples:
            image, host_logits, host_low, current_c1, output_size = c_v5._host_observation(
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
                c_v3_pred = host_pred
                c_v7_pred = host_pred
                semantic_state_low = torch.zeros_like(host_low)
                raw_history = [host_logits.detach()]
                previous = (image, host_low.detach(), current_c1.detach())
            elif pending_motion is None:
                _, previous_low, previous_c1 = previous
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
                c_v3_pred = host_pred
                c_v7_pred = host_pred
                raw_history = [host_logits.detach()]
                previous = (image, host_low.detach(), current_c1.detach())
            else:
                _, previous_low, previous_c1 = previous
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
                    corrector.history_length,
                )
                evidence = _build_c_v7_evidence(
                    corrector,
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

                c_v3_pred = c_v3_logits.argmax(1)
                c_v7_pred = evidence["final_logits"].argmax(1)
                correction_abs_sum += float(evidence["correction_full"].abs().mean().item())
                gate = evidence["row"]["gate_low"]
                gate_sum += float(gate.sum().item())
                gate_elements += int(gate.numel())
                active_frames += 1

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
                raw_history = raw_history[: corrector.history_length]
                motion_history.insert(0, pending_motion.detach())
                motion_history = motion_history[: max(corrector.history_length - 1, 0)]
                previous = (image, host_low.detach(), current_c1.detach())
                pending_motion = next_motion.detach()
                motion_hidden = next_motion_hidden.detach()

            predictions = {
                "host": host_pred,
                "c_v3": c_v3_pred,
                "c_v7": c_v7_pred,
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

        for name in names:
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
        for name in names
    }
    diagnostics = {
        "correction_abs_mean": correction_abs_sum / float(max(active_frames, 1)),
        "gate_mean": gate_sum / float(max(gate_elements, 1)),
        "history_length": corrector.history_length,
        "history_source": "detached raw frozen C-V3 logits",
        "raw_history_to_correction_path": False,
        "prediction_error_bottleneck": True,
        "error_state_motion_aligned": True,
        "error_state_reliability_gated": True,
        "final_writeback": "Z_final = detach(Z_c_v3) + masked upsample(g * tanh(delta_raw))",
        "raft_used_for_model_inference": False,
        "raft_used_for_metric_only": True,
    }
    return metrics, diagnostics


def _delta_metrics(candidate, reference):
    return {
        key: candidate[key] - reference[key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }


def _selection_key(metrics):
    candidate = metrics["c_v7"]
    passed = candidate["mIoU"] >= MIOU_HARD_FLOOR
    if passed:
        return (1, candidate["mTC"], candidate["mIoU"])
    return (0, candidate["mIoU"], candidate["mTC"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=c_v5.FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=c_v5.OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=c_v5.RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--base-checkpoint", default=c_v5.BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v3-checkpoint", default=c_v5.C_V3_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=c_v5.EPOCHS)
    parser.add_argument("--tbptt-steps", type=int, default=c_v5.TBPTT_STEPS)
    parser.add_argument("--lr", type=float, default=c_v5.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=c_v5.WEIGHT_DECAY)
    parser.add_argument("--gate-max", type=float, default=GATE_MAX)
    parser.add_argument("--dynamics-tau-e", type=float, default=c_v5.DYNAMICS_TAU_E)
    parser.add_argument("--dynamics-k-e", type=float, default=c_v5.DYNAMICS_K_E)
    parser.add_argument("--dynamics-dt", type=float, default=c_v5.DYNAMICS_DT)
    parser.add_argument("--seed", type=int, default=c_v5.SEED)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if args.tbptt_steps < 1:
        raise ValueError("tbptt-steps must be >= 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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

    for module in (observer, residual, correction, mask_predictor, refiner):
        if hasattr(module, "requires_grad_"):
            module.requires_grad_(False)
        if hasattr(module, "eval"):
            module.eval()

    corrector = MultiHypothesisErrorResidualCorrector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=c_v5.HISTORY_LENGTH,
        hidden_channels=ERROR_HIDDEN_CHANNELS,
        correction_channels=CORRECTION_CHANNELS,
        gate_max=args.gate_max,
    ).cuda()
    dynamics = EulerDynamicsError(
        tau_e=args.dynamics_tau_e,
        k_e=args.dynamics_k_e,
        dt=args.dynamics_dt,
    )
    zero_step = _zero_step_check(corrector)

    optimizer = torch.optim.AdamW(
        corrector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    raft = FrozenRAFT()

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train)
    all_val_groups = sequence_groups(val)
    missing = [sequence for sequence in c_v5.FULL9 if sequence not in all_val_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {sequence: all_val_groups[sequence] for sequence in c_v5.FULL9}

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    initial_metrics, initial_diagnostics = _evaluate(
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
    e0_delta = _delta_metrics(initial_metrics["c_v7"], initial_metrics["c_v3"])
    if any(abs(float(value)) > 1e-12 for value in e0_delta.values() if math.isfinite(value)):
        raise RuntimeError(f"C-V7 E0 is not exactly C-V3: delta={e0_delta}")

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
            "delta_c_v7_vs_c_v3": _delta_metrics(metrics["c_v7"], metrics["c_v3"]),
            "delta_c_v7_vs_host": _delta_metrics(metrics["c_v7"], metrics["host"]),
            "mIoU_floor_passed": metrics["c_v7"]["mIoU"] >= MIOU_HARD_FLOOR,
        }
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)

        if best is None or _selection_key(metrics) > _selection_key(best["metrics"]):
            best = row
            torch.save(
                {
                    "experiment": EXPERIMENT,
                    "epoch": epoch,
                    "corrector_state_dict": corrector.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "zero_step": zero_step,
                    "dynamics": dynamics.config(),
                    "architecture": {
                        "history_length": c_v5.HISTORY_LENGTH,
                        "prediction_error_definition": "V_k * (P_c_v3 - P_history_k)",
                        "raw_history_to_correction_path": False,
                        "error_state_update": (
                            "H_err_t = ConvGRU(E_t, Q_t * Warp(H_err_t-1, motion))"
                        ),
                        "error_state_motion_aligned": True,
                        "error_state_reliability_gated": True,
                        "correction_head_zero_initialized": True,
                        "gate_head_zero_weight_negative_bias_initialized": True,
                        "z_cur_detached": True,
                        "gate_channels": 1,
                        "gate_max": float(args.gate_max),
                        "bounded_delta": "tanh(delta_raw)",
                        "full_res_mask_order": "upsample correction first, then multiply validity",
                        "training_loss": "final segmentation CE only",
                        "gradient_norm_balancing": False,
                        "temporal_auxiliary_loss": False,
                    },
                    "selection_rule": {
                        "hard_mIoU_floor": MIOU_HARD_FLOOR,
                        "objective_after_floor": "maximize mTC, then mIoU",
                        "fallback": "highest mIoU, then mTC",
                    },
                    "checkpoints": {
                        "fast_b": args.fast_b_checkpoint,
                        "observer": args.observer_checkpoint,
                        "residual": args.residual_checkpoint,
                        "e1_base": args.base_checkpoint,
                        "c_v3": args.c_v3_checkpoint,
                        "residual_experiment": residual_payload.get("experiment"),
                        "e1_experiment": base_payload.get("experiment"),
                        "c_v3_epoch": cv3_payload.get("epoch"),
                    },
                },
                output / "best.pt",
            )

    summary = {
        "experiment": EXPERIMENT,
        "zero_step": zero_step,
        "initial_metrics": initial_metrics,
        "initial_diagnostics": initial_diagnostics,
        "initial_delta_c_v7_vs_c_v3": e0_delta,
        "history": history,
        "best": best,
        "architecture": {
            "history_to_correction": "History -> Prediction -> Prediction Error -> Correction",
            "raw_history_to_correction_path": False,
            "history_length": c_v5.HISTORY_LENGTH,
            "error_state_motion_aligned": True,
            "error_state_reliability_gated": True,
            "z_cur_detached": True,
            "gate_max": float(args.gate_max),
            "full_resolution_validity_after_upsampling": True,
            "training_loss": "segmentation CE only",
            "gradient_norm_balancing": False,
        },
    }
    with (result_output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
