"""C-V7: Motion-Aligned Multi-Hypothesis Prediction-Error Residual Correction.

中文：C-V7 运动对齐多假设预测误差残差修正。

Research boundary / 研究边界：
- Frozen Host / E1 / C-V3（冻结宿主/E1/C-V3）全部保持不变；
- K=4 History Bank（4 帧历史库）继续保存 raw detached frozen C-V3 logits；
- Raw History（原始历史）不能直接进入 Correction Head（修正头）；
- History -> motion-aligned prediction -> Prediction Error -> correction；
- Error Memory（误差记忆）必须 motion warp（运动对齐）+ reliability gating（可靠性门控）；
- final correction 在 logit space（分类得分空间）执行；
- 第一版训练只使用 final segmentation CE（最终语义分割交叉熵）；
- training（训练）不构造 RAFT；RAFT 仅用于 Full9 mTC（平均时序一致性）评测。
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


EXPERIMENT = "c_v7_error_residual_correction"
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v7_error_residual_correction"
)
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v7_error_residual_correction"
MIOU_HARD_FLOOR = 0.6637739071008685  # fixed C-V4 E2
G_MAX = 0.25
GATE_BIAS = -2.0
CANDIDATES = ("host", "c_v3_base", "c_v7")


def _full_any_history_valid(candidate_rows, history_length, full_size, device):
    mask = torch.zeros((1, 1, *full_size), dtype=torch.bool, device=device)
    for index in range(min(len(candidate_rows), history_length)):
        mask |= candidate_rows[index]["valid_full"].unsqueeze(1).bool()
    return mask


def _correction_evidence(
    corrector,
    dynamics,
    c_v3_logits,
    candidate_rows,
    pending_motion,
    transportability_low,
    memory_reliability_low,
    error_hidden,
    dynamics_state,
):
    """Build strict error-only evidence and final bounded residual logits."""
    low_size = tuple(transportability_low.shape[-2:])
    z_cur = c_v3_logits.detach()
    current_low = F.interpolate(
        z_cur,
        size=low_size,
        mode="bilinear",
        align_corners=False,
    )
    current_probability = F.softmax(current_low, dim=1)

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

    if dynamics_state is None:
        warped_dynamics = None
    else:
        warped_dynamics, _ = c_v5._warp_low_state_zero_invalid(
            dynamics_state,
            pending_motion,
        )
        warped_dynamics = (
            warped_dynamics
            * history1_valid
            * transportability_low.detach().clamp(0.0, 1.0)
        )

    dynamics_state = dynamics.step(
        primary_error,
        warped_dynamics,
    ).detach()

    row = corrector(
        prediction_errors=prediction_errors,
        dynamics_error=dynamics_state,
        current_margin=multi["current_margin"].detach(),
        history_margins=[margin.detach() for margin in multi["history_margins"]],
        transportability_low=transportability_low.detach(),
        memory_reliability_low=memory_reliability_low.detach(),
        history_validities_low=[validity.detach() for validity in history_validities],
        backward_motion_low=pending_motion.detach(),
        hidden=error_hidden,
    )

    full_size = tuple(z_cur.shape[-2:])
    # Important: upsample raw outputs first, then apply full-resolution validity.
    delta_full = F.interpolate(
        row["delta_z"],
        size=full_size,
        mode="bilinear",
        align_corners=False,
    )
    gate_full = F.interpolate(
        row["gate"],
        size=full_size,
        mode="bilinear",
        align_corners=False,
    )
    any_valid_full = _full_any_history_valid(
        candidate_rows,
        corrector.history_length,
        full_size,
        z_cur.device,
    )
    valid_float = any_valid_full.to(z_cur.dtype)
    delta_full = delta_full * valid_float
    gate_full = gate_full * valid_float

    # Z_cur is explicitly detached: no gradient can flow into frozen C-V3.
    final_logits = z_cur + gate_full * delta_full

    return {
        "row": row,
        "final_logits_full": final_logits,
        "delta_z_full": delta_full,
        "gate_full": gate_full,
        "any_valid_full": any_valid_full,
        "prediction_errors": prediction_errors,
        "prediction_error": primary_error,
        "dynamics_state": dynamics_state,
        "history_validities": history_validities,
    }


def _zero_step_equality_check(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    corrector,
    dynamics,
    groups,
):
    """Run a real frame and require exact C-V7 == frozen C-V3 at E0."""
    samples = next((rows for rows in groups.values() if len(rows) >= 3), None)
    if samples is None:
        raise RuntimeError("No sequence with at least three frames for C-V7 zero-step check")

    with torch.no_grad():
        frame0 = c_v5._host_observation(model, samples[0])
        frame1 = c_v5._host_observation(model, samples[1])
        pending_motion, _ = c_v5._initialize_motion(observer, residual, frame0, frame1)
        previous_image, previous_host_logits, previous_low, previous_c1, _ = frame1
        _, host_logits, host_low, current_c1, output_size = c_v5._host_observation(
            model, samples[2]
        )
        prior_low, _ = c_v5.warp_low_logits(previous_low, pending_motion)
        semantic_state_low = torch.zeros_like(previous_low)
        e1 = c_v5._frozen_e1_step(
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
        memory_row, _, _, c_v3_logits = c_v5._frozen_cv3_step(
            refiner,
            current_c1,
            host_low,
            prior_low,
            e1,
            pending_motion,
            None,
            output_size,
            host_logits,
        )
        candidate_rows = c_v5._build_history_candidates(
            [previous_host_logits.detach()],
            [],
            pending_motion,
            corrector.history_length,
        )
        evidence = _correction_evidence(
            corrector,
            dynamics,
            c_v3_logits,
            candidate_rows,
            pending_motion,
            e1["transportability_low"],
            memory_row["memory_reliability"],
            None,
            None,
        )

    head_weight = float(corrector.correction_head[-1].weight.detach().abs().max().item())
    head_bias = float(corrector.correction_head[-1].bias.detach().abs().max().item())
    delta_raw = float(evidence["row"]["delta_z_raw"].detach().abs().max().item())
    equality = float(
        (evidence["final_logits_full"] - c_v3_logits.detach()).abs().max().item()
    )
    if head_weight != 0.0 or head_bias != 0.0 or delta_raw != 0.0 or equality != 0.0:
        raise RuntimeError(
            "C-V7 zero-step equality failed: "
            f"head_w={head_weight} head_b={head_bias} delta={delta_raw} equality={equality}"
        )
    return {
        "correction_head_weight_abs_max": head_weight,
        "correction_head_bias_abs_max": head_bias,
        "delta_z_raw_abs_max": delta_raw,
        "c_v7_vs_c_v3_logit_abs_max": equality,
        "gate_bias": float(corrector.gate_head.bias.detach().mean().item()),
        "g_max": corrector.g_max,
    }


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
    error_hidden = None
    dynamics_state = None

    buffered_losses = []
    frames_in_window = 0
    totals = {
        "frames": 0,
        "optimizer_steps": 0,
        "segmentation_ce": 0.0,
        "prediction_error_abs": 0.0,
        "dynamics_error_abs": 0.0,
        "delta_z_abs": 0.0,
        "gate_mean": 0.0,
        "applied_correction_abs": 0.0,
        "error_reliability_mean": 0.0,
        "sign_agreement_mean": 0.0,
        "valid_fraction_mean": 0.0,
    }

    for frame_index in range(2, len(samples)):
        _, host_logits, host_low, current_c1, output_size = c_v5._host_observation(
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
                corrector.history_length,
            )

        evidence = _correction_evidence(
            corrector,
            dynamics,
            c_v3_logits,
            candidate_rows,
            pending_motion,
            e1["transportability_low"],
            memory_row["memory_reliability"],
            error_hidden,
            dynamics_state,
        )
        error_hidden = evidence["row"]["hidden"]
        dynamics_state = evidence["dynamics_state"]

        target = current_gt.to(
            evidence["final_logits_full"].device,
            non_blocking=True,
        ).unsqueeze(0)
        segmentation_ce = F.cross_entropy(
            evidence["final_logits_full"],
            target,
            ignore_index=c_v5.IGNORE_LABEL,
        )
        if not torch.isfinite(segmentation_ce):
            raise FloatingPointError("Non-finite C-V7 segmentation CE")
        buffered_losses.append(segmentation_ce)

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

            row = evidence["row"]
            totals["frames"] += 1
            totals["segmentation_ce"] += float(segmentation_ce.detach().item())
            totals["prediction_error_abs"] += float(
                evidence["prediction_error"].abs().mean().item()
            )
            totals["dynamics_error_abs"] += float(dynamics_state.abs().mean().item())
            totals["delta_z_abs"] += float(row["delta_z"].abs().mean().item())
            totals["gate_mean"] += float(evidence["gate_full"].mean().item())
            totals["applied_correction_abs"] += float(
                (evidence["gate_full"] * evidence["delta_z_full"]).abs().mean().item()
            )
            totals["error_reliability_mean"] += float(
                row["error_reliability"].mean().item()
            )
            totals["sign_agreement_mean"] += float(row["sign_agreement"].mean().item())
            totals["valid_fraction_mean"] += float(row["valid_fraction"].mean().item())

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
            if error_hidden is not None:
                error_hidden = error_hidden.detach()
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
    for key in (
        "segmentation_ce",
        "prediction_error_abs",
        "dynamics_error_abs",
        "delta_z_abs",
        "gate_mean",
        "applied_correction_abs",
        "error_reliability_mean",
        "sign_agreement_mean",
        "valid_fraction_mean",
    ):
        totals[key] /= frames
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
            samples,
            optimizer,
            tbptt_steps,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("No valid C-V7 training sequences")

    frame_total = max(sum(row["frames"] for row in rows), 1)
    result = {
        "frames": sum(row["frames"] for row in rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
    }
    for key in (
        "segmentation_ce",
        "prediction_error_abs",
        "dynamics_error_abs",
        "delta_z_abs",
        "gate_mean",
        "applied_correction_abs",
        "error_reliability_mean",
        "sign_agreement_mean",
        "valid_fraction_mean",
    ):
        result[key] = sum(row[key] * row["frames"] for row in rows) / frame_total
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
        name: torch.zeros((c_v5.NUM_CLASSES, c_v5.NUM_CLASSES), dtype=torch.int64)
        for name in CANDIDATES
    }
    mtc_sum = {name: 0.0 for name in CANDIDATES}
    mtc_count = {name: 0 for name in CANDIDATES}
    vc_sum = {name: {8: 0.0, 16: 0.0} for name in CANDIDATES}
    vc_count = {name: {8: 0, 16: 0} for name in CANDIDATES}
    diagnostics = {
        "frames_with_correction_path": 0,
        "delta_z_abs": 0.0,
        "gate_mean": 0.0,
        "applied_correction_abs": 0.0,
        "error_reliability_mean": 0.0,
        "sign_agreement_mean": 0.0,
    }

    for sequence in c_v5.FULL9:
        previous = None
        pending_motion = motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = None
        memory_state = None
        error_hidden = None
        dynamics_state = None
        raw_history = []
        motion_history = []
        previous_predictions = {}
        seq_vc = {name: c_v5.VideoConsistency() for name in CANDIDATES}

        for sample in groups[sequence]:
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
                c_v3_pred = c_v7_pred = host_pred
                semantic_state_low = torch.zeros_like(host_low)
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
                motion_error = F.softmax(host_low, dim=1) - F.softmax(previous_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(
                    observed,
                    motion_error,
                    None,
                )
                c_v3_pred = c_v7_pred = host_pred
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
                evidence = _correction_evidence(
                    corrector,
                    dynamics,
                    c_v3_logits,
                    candidate_rows,
                    pending_motion,
                    e1["transportability_low"],
                    memory_row["memory_reliability"],
                    error_hidden,
                    dynamics_state,
                )
                error_hidden = evidence["row"]["hidden"]
                dynamics_state = evidence["dynamics_state"]
                c_v3_pred = c_v3_logits.argmax(1)
                c_v7_pred = evidence["final_logits_full"].argmax(1)

                diagnostics["frames_with_correction_path"] += 1
                diagnostics["delta_z_abs"] += float(
                    evidence["row"]["delta_z"].abs().mean().item()
                )
                diagnostics["gate_mean"] += float(evidence["gate_full"].mean().item())
                diagnostics["applied_correction_abs"] += float(
                    (evidence["gate_full"] * evidence["delta_z_full"]).abs().mean().item()
                )
                diagnostics["error_reliability_mean"] += float(
                    evidence["row"]["error_reliability"].mean().item()
                )
                diagnostics["sign_agreement_mean"] += float(
                    evidence["row"]["sign_agreement"].mean().item()
                )

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
                "c_v3_base": c_v3_pred,
                "c_v7": c_v7_pred,
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
    count = max(diagnostics["frames_with_correction_path"], 1)
    for key in (
        "delta_z_abs",
        "gate_mean",
        "applied_correction_abs",
        "error_reliability_mean",
        "sign_agreement_mean",
    ):
        diagnostics[key] /= count
    diagnostics.update(
        {
            "history_length": corrector.history_length,
            "history_source": "raw detached frozen C-V3 logits",
            "raw_history_enters_correction_head": False,
            "error_hidden_motion_aligned": True,
            "error_hidden_reliability_gated": True,
            "z_cur_detached": True,
            "bounded_logit_residual": True,
            "g_max": corrector.g_max,
            "raft_training": False,
            "raft_inference_decision": False,
            "raft_metric_only": True,
        }
    )
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
    parser.add_argument("--g-max", type=float, default=G_MAX)
    parser.add_argument("--gate-bias", type=float, default=GATE_BIAS)
    parser.add_argument("--dynamics-tau-e", type=float, default=c_v5.DYNAMICS_TAU_E)
    parser.add_argument("--dynamics-k-e", type=float, default=c_v5.DYNAMICS_K_E)
    parser.add_argument("--dynamics-dt", type=float, default=c_v5.DYNAMICS_DT)
    parser.add_argument("--seed", type=int, default=c_v5.SEED)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    c_v5._validate_bounded_motion_checkpoint(args.residual_checkpoint)
    model = c_v5.load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = c_v5._load_frozen_observer(args.observer_checkpoint)
    residual, residual_payload = c_v5._load_frozen_residual(
        args.residual_checkpoint,
        observer,
    )
    correction, mask_predictor, base_payload = c_v5._load_frozen_e1_base(
        args.base_checkpoint,
        observer,
    )
    refiner, cv3_payload = c_v5._load_frozen_cv3_refiner(args.c_v3_checkpoint)

    corrector = MultiHypothesisErrorResidualCorrector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=c_v5.HISTORY_LENGTH,
        hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
        g_max=args.g_max,
        gate_bias=args.gate_bias,
    ).cuda()
    dynamics = EulerDynamicsError(
        tau_e=args.dynamics_tau_e,
        k_e=args.dynamics_k_e,
        dt=args.dynamics_dt,
    )

    optimizer = torch.optim.AdamW(
        corrector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train)
    all_val_groups = sequence_groups(val)
    missing = [sequence for sequence in c_v5.FULL9 if sequence not in all_val_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {sequence: all_val_groups[sequence] for sequence in c_v5.FULL9}

    zero_step = _zero_step_equality_check(
        model,
        observer,
        residual,
        correction,
        mask_predictor,
        refiner,
        corrector,
        dynamics,
        train_groups,
    )

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    # RAFT is deliberately constructed only after zero-step/training setup.
    # It is passed only to _evaluate for the published mTC metric.
    raft_metric = FrozenRAFT()

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
            raft_metric,
        )
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_host": _delta_metrics(metrics["c_v7"], metrics["host"]),
            "delta_vs_c_v3": _delta_metrics(metrics["c_v7"], metrics["c_v3_base"]),
        }
        row["selection_key"] = list(_selection_key(metrics))
        history.append(row)

        with (result_output / f"epoch_{epoch:03d}.json").open("w") as handle:
            json.dump(row, handle, indent=2)

        if best is None or tuple(row["selection_key"]) > tuple(best["selection_key"]):
            best = row
            torch.save(
                {
                    "experiment": EXPERIMENT,
                    "epoch": epoch,
                    "corrector_state_dict": corrector.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "train": train_stats,
                    "dynamics": dynamics.config(),
                    "architecture": {
                        "history_length": c_v5.HISTORY_LENGTH,
                        "history_source": "raw detached frozen C-V3 logits",
                        "raw_history_enters_correction_head": False,
                        "prediction_error": "V_k * (P_cur - P_hist_k)",
                        "multi_hypothesis_error_aggregation": True,
                        "error_hidden_motion_aligned": True,
                        "error_hidden_reliability_gated": True,
                        "error_reliability_source": "frozen C-V3 memory_reliability * validity",
                        "correction_head_zero_initialized": True,
                        "gate_channels": 1,
                        "g_max": corrector.g_max,
                        "gate_bias_init": corrector.gate_bias,
                        "z_cur_detached": True,
                        "output": "Z_final = Z_cur + g * tanh(DeltaZ_raw)",
                        "validity_order": "upsample first, full-resolution mask second",
                        "training_loss": "segmentation CE only",
                        "raft_training": False,
                        "raft_inference_decision": False,
                    },
                    "zero_step": zero_step,
                },
                output / "best.pt",
            )
        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V7 produced no epoch result")

    summary = {
        "experiment": "C-V7 Motion-Aligned Multi-Hypothesis Prediction-Error Residual Correction",
        "best": best,
        "history": history,
        "zero_step": zero_step,
        "selection_rule": {
            "hard_constraint": f"C-V7 mIoU >= fixed C-V4 E2 floor {MIOU_HARD_FLOOR:.16f}",
            "objective_after_constraint": "maximize C-V7 mTC, then mIoU",
            "fallback_if_all_fail_floor": "highest mIoU, then mTC",
        },
        "architecture": {
            "history_length": c_v5.HISTORY_LENGTH,
            "history_source": "raw detached frozen C-V3 logits",
            "causal_path": "History -> Prediction -> Prediction Error -> Residual Correction",
            "raw_history_enters_correction_head": False,
            "error_hidden_motion_aligned": True,
            "error_hidden_reliability_gated": True,
            "z_cur_detached": True,
            "bounded_logit_residual": True,
            "g_max": corrector.g_max,
            "gate_bias_init": corrector.gate_bias,
            "correction_head_zero_initialized": True,
            "training_loss": "final segmentation CE only",
            "temporal_loss": False,
            "raft_training": False,
            "raft_metric_only": True,
        },
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
        "tbptt_steps": args.tbptt_steps,
        "epochs": args.epochs,
    }
    with (result_output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
