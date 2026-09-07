"""C-V6H: Hierarchical Error-Centric Multi-Hypothesis Temporal Coding.

中文：分层误差中心多假设时序预测编码。

This keeps the completed no-distillation C-V6 experiment reproducible and
changes only the top-level decision task:
  Stage 1: Current vs History, trained with class-balanced binary CE.
  Stage 2: t-1..t-K, trained only on pixels whose task target is History.

The explicit Dynamics Error remains a t-1 temporal-persistence signal and is
used only by Stage 1.  Stage 2 uses the K validity-gated prediction errors and
history reliability evidence; it does not receive Dynamics Error.
"""

import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

import predify2021.mce_scores.train_kitti_step_task_space_prior_c_v6_error_centric_multihypothesis as base
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_hierarchical_error_selector import (
    HierarchicalMultiHypothesisErrorSelector,
)


EXPERIMENT = "c_v6h_hierarchical_error_selector"
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v6h_hierarchical_error_selector"
)
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v6h_hierarchical_error_selector"
GATE_LOSS_WEIGHT = 1.0
HISTORY_LOSS_WEIGHT = 1.0

_ORIGINAL_SELECTOR_EVIDENCE = base._selector_evidence
_ORIGINAL_ZERO_STEP = base.c_v5._zero_step_check


def _zero_step_check(selector):
    gate_w = float(selector.gate_head.weight.detach().abs().max().item())
    gate_b = float(selector.gate_head.bias.detach().abs().max().item())
    hist_w = float(selector.history_head.weight.detach().abs().max().item())
    hist_b = float(selector.history_head.bias.detach().abs().max().item())
    if max(gate_w, gate_b, hist_w, hist_b) != 0.0:
        raise RuntimeError("C-V6H gate/history heads must be exactly zero initialized")
    return {
        "gate_head_weight_abs_max": gate_w,
        "gate_head_bias_abs_max": gate_b,
        "history_head_weight_abs_max": hist_w,
        "history_head_bias_abs_max": hist_b,
        "zero_step_argmax_index": 0,
        "zero_step_behavior": "hierarchical tie resolves to frozen C-V3 Current",
    }


def _selector_evidence(*args, **kwargs):
    """Reuse validated error construction and expose both hierarchical heads."""
    evidence = _ORIGINAL_SELECTOR_EVIDENCE(*args, **kwargs)
    selector = args[0] if args else kwargs["selector"]
    c_v3_logits = args[2] if len(args) > 2 else kwargs["c_v3_logits"]
    candidate_rows = args[3] if len(args) > 3 else kwargs["candidate_rows"]
    full_size = tuple(c_v3_logits.shape[-2:])
    row = evidence["row"]

    gate_full = F.interpolate(
        row["gate_logits"],
        size=full_size,
        mode="bilinear",
        align_corners=False,
    )
    history_full = F.interpolate(
        row["history_logits"],
        size=full_size,
        mode="bilinear",
        align_corners=False,
    )

    any_valid = torch.zeros_like(gate_full[:, :1], dtype=torch.bool)
    for index in range(selector.history_length):
        if index < len(candidate_rows):
            valid = candidate_rows[index]["valid_full"].unsqueeze(1).bool()
            any_valid |= valid
            history_full[:, index : index + 1] = torch.where(
                valid,
                history_full[:, index : index + 1],
                torch.full_like(history_full[:, index : index + 1], -1.0e4),
            )
        else:
            history_full[:, index : index + 1].fill_(-1.0e4)
    gate_full = gate_full.clone()
    gate_full[:, 1:2] = torch.where(
        any_valid,
        gate_full[:, 1:2],
        torch.full_like(gate_full[:, 1:2], -1.0e4),
    )

    # Compose full-resolution hierarchical logits exactly as in the model.
    best_history = history_full.max(dim=1, keepdim=True).values
    history_relative = history_full - best_history
    history_relative = torch.where(
        any_valid,
        history_relative,
        torch.full_like(history_relative, -1.0e4),
    )
    evidence["gate_logits_full"] = gate_full
    evidence["history_logits_full"] = history_full
    evidence["selector_logits_full"] = torch.cat(
        (gate_full[:, :1], gate_full[:, 1:2] + history_relative),
        dim=1,
    )
    return evidence


def _balanced_gate_loss_sums(gate_logits, gate_target):
    """Return separate Current/History CE sums so each side gets 0.5 weight."""
    per_pixel = F.cross_entropy(gate_logits, gate_target, reduction="none")
    history_mask = gate_target == 1
    current_mask = ~history_mask
    current_sum = per_pixel[current_mask].sum() if bool(current_mask.any()) else per_pixel.sum() * 0.0
    history_sum = per_pixel[history_mask].sum() if bool(history_mask.any()) else per_pixel.sum() * 0.0
    return current_sum, int(current_mask.sum().item()), history_sum, int(history_mask.sum().item())


def _balanced_gate_mean(current_sum, current_count, history_sum, history_count):
    terms = []
    if current_count:
        terms.append(current_sum / float(current_count))
    if history_count:
        terms.append(history_sum / float(history_count))
    if not terms:
        return current_sum * 0.0
    return torch.stack(terms).mean()


def _train_sequence_distilled(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    selector,
    dynamics,
    teacher_selector,
    teacher_dynamics,
    teacher_evidence_fn,
    raft,
    samples,
    optimizer,
    tbptt_steps,
    distill_weight,
    distill_temperature,
):
    del teacher_selector, teacher_dynamics, teacher_evidence_fn, distill_temperature
    if float(distill_weight) != 0.0:
        raise ValueError("C-V6H is the no-distillation structural test; distill-weight must be 0")
    if len(samples) < 3:
        return None

    c_v5 = base.c_v5
    frame0 = c_v5._host_observation(model, samples[0])
    frame1 = c_v5._host_observation(model, samples[1])
    pending_motion, motion_hidden = c_v5._initialize_motion(observer, residual, frame0, frame1)
    previous_image, previous_host_logits, previous_low, previous_c1, _ = frame1

    raw_history = [previous_host_logits.detach()]
    motion_history = []
    previous_cv3_logits = previous_host_logits.detach()
    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = None
    selector_hidden = None
    dynamics_state = None

    window_gate_current_sums = []
    window_gate_current_counts = []
    window_gate_history_sums = []
    window_gate_history_counts = []
    window_history_sums = []
    window_history_counts = []
    frames_in_window = 0

    totals = {
        "frames": 0,
        "supervised_frames": 0,
        "optimizer_steps": 0,
        "supervised_pixels": 0,
        "distill_pixels": 0,
        "selector_ce_per_pixel": 0.0,
        "distill_kl_per_pixel": 0.0,
        "teacher_student_agree_pixels": 0,
        "teacher_student_compare_pixels": 0,
        "prediction_error_abs": 0.0,
        "dynamics_error_abs": 0.0,
        "selected_candidate_counts": [0] * (selector.history_length + 1),
        "gate_current_target_pixels": 0,
        "gate_history_target_pixels": 0,
        "gate_pred_current_pixels": 0,
        "gate_pred_history_pixels": 0,
        "gate_true_history_pred_history": 0,
        "gate_false_history_pred_history": 0,
        "history_supervised_pixels": 0,
        "history_target_counts": [0] * selector.history_length,
        "history_pred_counts": [0] * selector.history_length,
        "gate_current_loss_sum": 0.0,
        "gate_history_loss_sum": 0.0,
        "history_loss_sum": 0.0,
    }
    target_totals = c_v5._new_target_totals(selector.history_length)

    for frame_index in range(2, len(samples)):
        current_image, host_logits, host_low, current_c1, output_size = c_v5._host_observation(
            model, samples[frame_index]
        )
        current_gt = c_v5.semantic_mask_from_panoptic_png(samples[frame_index]["mask_path"])

        with torch.no_grad():
            prior_low, _ = c_v5.warp_low_logits(previous_low, pending_motion)
            e1 = c_v5._frozen_e1_step(
                correction, mask_predictor, current_c1, host_low, prior_low,
                pending_motion, semantic_state_low, transport_hidden,
                semantic_hidden, mask_hidden,
            )
            transport_hidden = e1["transport_hidden"]
            semantic_hidden = e1["semantic_hidden"]
            mask_hidden = e1["mask_hidden"]
            semantic_state_low = e1["semantic_state_low"]
            memory_row, memory_state, _, c_v3_logits = c_v5._frozen_cv3_step(
                refiner, current_c1, host_low, prior_low, e1, pending_motion,
                memory_state, output_size, host_logits,
            )
            candidate_rows = c_v5._build_history_candidates(
                raw_history, motion_history, pending_motion, selector.history_length
            )

        evidence = _selector_evidence(
            selector, dynamics, c_v3_logits, candidate_rows, pending_motion,
            e1["transportability_low"], memory_row["memory_reliability"],
            selector_hidden, dynamics_state,
        )
        selector_hidden = evidence["row"]["hidden"]
        dynamics_state = evidence["dynamics_state"]

        with torch.no_grad():
            teacher_full = raft.current_to_previous(current_image, previous_image)
            _, temporal_target, supervised, _, target_diag = c_v5._build_multiframe_target(
                c_v3_logits, candidate_rows, previous_cv3_logits, current_gt, teacher_full
            )
            c_v5._add_target_totals(target_totals, target_diag)

        if bool(supervised.any()):
            gate_logits = evidence["gate_logits_full"][0].permute(1, 2, 0)[supervised]
            task_target = temporal_target[supervised]
            gate_target = (task_target > 0).long()
            current_sum, current_count, history_gate_sum, history_gate_count = (
                _balanced_gate_loss_sums(gate_logits, gate_target)
            )
            window_gate_current_sums.append(current_sum)
            window_gate_current_counts.append(current_count)
            window_gate_history_sums.append(history_gate_sum)
            window_gate_history_counts.append(history_gate_count)
            totals["supervised_frames"] += 1
            totals["supervised_pixels"] += int(supervised.sum().item())
            totals["gate_current_target_pixels"] += current_count
            totals["gate_history_target_pixels"] += history_gate_count
            totals["gate_current_loss_sum"] += float(current_sum.detach().item())
            totals["gate_history_loss_sum"] += float(history_gate_sum.detach().item())

            with torch.no_grad():
                gate_pred = gate_logits.argmax(dim=-1)
                totals["gate_pred_current_pixels"] += int((gate_pred == 0).sum().item())
                totals["gate_pred_history_pixels"] += int((gate_pred == 1).sum().item())
                totals["gate_true_history_pred_history"] += int(
                    ((gate_target == 1) & (gate_pred == 1)).sum().item()
                )
                totals["gate_false_history_pred_history"] += int(
                    ((gate_target == 0) & (gate_pred == 1)).sum().item()
                )

            history_mask = task_target > 0
            if bool(history_mask.any()):
                history_logits = evidence["history_logits_full"][0].permute(1, 2, 0)[supervised][history_mask]
                history_target = task_target[history_mask] - 1
                history_sum = F.cross_entropy(history_logits, history_target, reduction="sum")
                history_count = int(history_target.numel())
                window_history_sums.append(history_sum)
                window_history_counts.append(history_count)
                totals["history_supervised_pixels"] += history_count
                totals["history_loss_sum"] += float(history_sum.detach().item())
                with torch.no_grad():
                    history_pred = history_logits.argmax(dim=-1)
                    for age_index in range(selector.history_length):
                        totals["history_target_counts"][age_index] += int(
                            (history_target == age_index).sum().item()
                        )
                        totals["history_pred_counts"][age_index] += int(
                            (history_pred == age_index).sum().item()
                        )

        with torch.no_grad():
            hard_selection = evidence["selector_logits_full"].argmax(1)[0]
            for index in range(selector.history_length + 1):
                totals["selected_candidate_counts"][index] += int(
                    (hard_selection == index).sum().item()
                )
            totals["prediction_error_abs"] += float(evidence["prediction_error"].abs().mean().item())
            totals["dynamics_error_abs"] += float(dynamics_state.abs().mean().item())
            observed_motion = c_v5._observe_motion(
                observer, previous_low, previous_c1, host_low, current_c1
            )
            motion_error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion, motion_error, motion_hidden
            )

        totals["frames"] += 1
        frames_in_window += 1
        boundary = frames_in_window >= tbptt_steps or frame_index == len(samples) - 1
        if boundary:
            gate_current_sum = torch.stack(window_gate_current_sums).sum() if window_gate_current_sums else c_v3_logits.sum() * 0.0
            gate_history_sum = torch.stack(window_gate_history_sums).sum() if window_gate_history_sums else c_v3_logits.sum() * 0.0
            gate_loss = _balanced_gate_mean(
                gate_current_sum,
                sum(window_gate_current_counts),
                gate_history_sum,
                sum(window_gate_history_counts),
            )
            if window_history_sums:
                history_loss = torch.stack(window_history_sums).sum() / float(max(sum(window_history_counts), 1))
            else:
                history_loss = gate_loss * 0.0
            window_loss = GATE_LOSS_WEIGHT * gate_loss + HISTORY_LOSS_WEIGHT * history_loss
            optimizer.zero_grad(set_to_none=True)
            window_loss.backward()
            optimizer.step()
            totals["optimizer_steps"] += 1

            window_gate_current_sums = []
            window_gate_current_counts = []
            window_gate_history_sums = []
            window_gate_history_counts = []
            window_history_sums = []
            window_history_counts = []
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
    current_count = max(totals["gate_current_target_pixels"], 1)
    history_gate_count = max(totals["gate_history_target_pixels"], 1)
    history_count = max(totals["history_supervised_pixels"], 1)
    gate_current_ce = totals["gate_current_loss_sum"] / current_count
    gate_history_ce = totals["gate_history_loss_sum"] / history_gate_count
    gate_balanced_ce = 0.5 * (gate_current_ce + gate_history_ce)
    history_ce = totals["history_loss_sum"] / history_count
    totals["gate_balanced_ce"] = gate_balanced_ce
    totals["history_ce"] = history_ce
    totals["selector_ce_per_pixel"] = gate_balanced_ce + history_ce
    totals["history_gate_recall"] = totals["gate_true_history_pred_history"] / history_gate_count
    pred_history = max(totals["gate_pred_history_pixels"], 1)
    totals["history_gate_precision"] = totals["gate_true_history_pred_history"] / pred_history
    totals["teacher_student_agreement"] = 0.0
    totals["prediction_error_abs"] /= frames
    totals["dynamics_error_abs"] /= frames
    totals["targets"] = c_v5._target_rates(target_totals)
    return totals


def _train_epoch_distilled(
    model, observer, residual, correction, mask_predictor, refiner, selector,
    dynamics, teacher_selector, teacher_dynamics, teacher_evidence_fn, raft,
    groups, optimizer, tbptt_steps, distill_weight, distill_temperature,
):
    selector.train()
    refiner.eval()
    correction.eval()
    mask_predictor.eval()
    rows = []
    target_totals = base.c_v5._new_target_totals(selector.history_length)
    for samples in groups.values():
        row = _train_sequence_distilled(
            model, observer, residual, correction, mask_predictor, refiner,
            selector, dynamics, teacher_selector, teacher_dynamics,
            teacher_evidence_fn, raft, samples, optimizer, tbptt_steps,
            distill_weight, distill_temperature,
        )
        if row is not None:
            rows.append(row)
            base.c_v5._add_target_totals(target_totals, row["targets"])
    if not rows:
        raise RuntimeError("No valid C-V6H training sequences")

    frame_total = max(sum(row["frames"] for row in rows), 1)
    result = {
        "frames": sum(row["frames"] for row in rows),
        "supervised_frames": sum(row["supervised_frames"] for row in rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "supervised_pixels": sum(row["supervised_pixels"] for row in rows),
        "distill_pixels": 0,
        "distill_kl_per_pixel": 0.0,
        "teacher_student_agreement": 0.0,
        "prediction_error_abs": sum(row["prediction_error_abs"] * row["frames"] for row in rows) / frame_total,
        "dynamics_error_abs": sum(row["dynamics_error_abs"] * row["frames"] for row in rows) / frame_total,
        "selected_candidate_counts": [
            sum(row["selected_candidate_counts"][i] for row in rows)
            for i in range(selector.history_length + 1)
        ],
        "gate_current_target_pixels": sum(row["gate_current_target_pixels"] for row in rows),
        "gate_history_target_pixels": sum(row["gate_history_target_pixels"] for row in rows),
        "gate_pred_current_pixels": sum(row["gate_pred_current_pixels"] for row in rows),
        "gate_pred_history_pixels": sum(row["gate_pred_history_pixels"] for row in rows),
        "gate_true_history_pred_history": sum(row["gate_true_history_pred_history"] for row in rows),
        "history_supervised_pixels": sum(row["history_supervised_pixels"] for row in rows),
        "history_target_counts": [sum(row["history_target_counts"][i] for row in rows) for i in range(selector.history_length)],
        "history_pred_counts": [sum(row["history_pred_counts"][i] for row in rows) for i in range(selector.history_length)],
        "targets": base.c_v5._target_rates(target_totals),
        "distillation": {"weight": 0.0, "teacher_inference": False},
    }
    current_loss_sum = sum(row["gate_current_loss_sum"] for row in rows)
    history_gate_loss_sum = sum(row["gate_history_loss_sum"] for row in rows)
    history_loss_sum = sum(row["history_loss_sum"] for row in rows)
    current_n = max(result["gate_current_target_pixels"], 1)
    history_n = max(result["gate_history_target_pixels"], 1)
    history_stage_n = max(result["history_supervised_pixels"], 1)
    result["gate_balanced_ce"] = 0.5 * (
        current_loss_sum / current_n + history_gate_loss_sum / history_n
    )
    result["history_ce"] = history_loss_sum / history_stage_n
    result["selector_ce_per_pixel"] = result["gate_balanced_ce"] + result["history_ce"]
    result["history_gate_recall"] = result["gate_true_history_pred_history"] / history_n
    result["history_gate_precision"] = result["gate_true_history_pred_history"] / max(result["gate_pred_history_pixels"], 1)
    return result


def _arg_value(argv, name, default):
    args = list(sys.argv[1:] if argv is None else argv)
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix):]
    return default


def _rewrite_summary(argv):
    result_dir = Path(_arg_value(argv, "--result-output", RESULT_DEFAULT))
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        return
    with summary_path.open() as handle:
        summary = json.load(handle)
    summary["experiment"] = "C-V6H Hierarchical Error-Centric Multi-Hypothesis Temporal Coding"
    summary["architecture"].update(
        {
            "decision_decomposition": "Stage1 Current-vs-History; Stage2 t-1..t-K",
            "stage1_semantic_evidence": "strict e1 + Dynamics Error only",
            "stage1_loss": "balanced Current/History CE (0.5 / 0.5)",
            "stage2_semantic_evidence": "strict e1..eK; no Dynamics Error",
            "stage2_loss": "history-age CE only on History-target pixels",
            "training_loss": "balanced gate CE + conditional history CE; no distillation",
            "five_way_unweighted_ce": False,
        }
    )
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)


def _patch_base():
    base.MultiHypothesisErrorSelector = HierarchicalMultiHypothesisErrorSelector
    base._selector_evidence = _selector_evidence
    base._train_sequence_distilled = _train_sequence_distilled
    base._train_epoch_distilled = _train_epoch_distilled
    base.EXPERIMENT = EXPERIMENT
    base.OUTPUT_DEFAULT = OUTPUT_DEFAULT
    base.RESULT_DEFAULT = RESULT_DEFAULT
    base.DISTILL_WEIGHT = 0.0
    base.c_v5._zero_step_check = _zero_step_check


def main(argv=None):
    _patch_base()
    try:
        result = base.main(argv)
        _rewrite_summary(argv)
        return result
    finally:
        base.c_v5._zero_step_check = _ORIGINAL_ZERO_STEP


if __name__ == "__main__":
    main()
