"""C-V7: Prediction-Error Candidate Utility Estimation.

中文：C-V7 预测误差候选效用估计。

Frozen foundation:
  Host -> Motion -> E1 -> C-V3 -> K=4 one-warp Candidate Bank.

Only the top controller changes.  C-V7 does NOT classify Current/t-1/.../t-K.
For every historical hypothesis it predicts a continuous relative utility
against frozen C-V3 Current.  Current utility is fixed to zero.

Training supervision:
  semantic utility = log p_history(GT) - log p_current(GT)

Positive-gain and negative-gain hypotheses are normalized separately, avoiding
the C-V6 Current-collapse shortcut.  RAFT is training-only and contributes a
small temporal ranking loss only when two candidates are semantically tied.
"""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v6_error_centric_multihypothesis as c_v6,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_multihypothesis_error_utility import (
    MultiHypothesisErrorUtilityEstimator,
    balanced_utility_regression_loss,
    semantic_first_temporal_tiebreak_loss,
)


EXPERIMENT = "c_v7_error_utility_estimator"
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v7_error_utility_estimator"
)
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v7_error_utility_estimator"

UTILITY_CLIP = 4.0
NEUTRAL_DELTA = 0.10
NEUTRAL_WEIGHT = 0.10
SEMANTIC_TIE_DELTA = 0.25
TEMPORAL_RANK_MARGIN = 0.05
TEMPORAL_RANK_WEIGHT = 0.10


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
    """Build C-V7 evidence while keeping history semantics behind error interface."""
    low_size = tuple(transportability_low.shape[-2:])
    current_low = F.interpolate(
        c_v3_logits.detach(),
        size=low_size,
        mode="bilinear",
        align_corners=False,
    )
    current_probability = F.softmax(current_low, dim=1)
    history_probabilities, low_path_validities = c_v5._pad_history_for_controller(
        current_probability,
        candidate_rows,
        selector.history_length,
    )
    history_validities = c_v6._strict_history_validities(
        current_probability,
        candidate_rows,
        low_path_validities,
        selector.history_length,
    )
    multi = c_v6.build_multihypothesis_error_evidence(
        current_probability,
        history_probabilities,
        history_validities,
    )
    prediction_errors = multi["prediction_errors"]
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
    row = selector(
        prediction_errors,
        dynamics_state,
        current_probability,
        multi["current_margin"],
        multi["history_margins"],
        transportability_low.detach(),
        memory_reliability_low.detach(),
        [validity.detach() for validity in history_validities],
        selector_hidden,
    )

    selector_logits_full_raw = F.interpolate(
        row["selector_logits"],
        size=tuple(c_v3_logits.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )
    utility_full_raw = selector_logits_full_raw[:, 1:]

    selector_channels = [selector_logits_full_raw[:, :1]]
    utility_channels = []
    for index in range(selector.history_length):
        raw_utility = utility_full_raw[:, index : index + 1]
        if index < len(candidate_rows):
            valid = candidate_rows[index]["valid_full"].unsqueeze(1)
            masked = torch.where(
                valid,
                raw_utility,
                torch.full_like(raw_utility, -1e4),
            )
        else:
            masked = torch.full_like(raw_utility, -1e4)
        selector_channels.append(masked)
        utility_channels.append(masked)

    selector_logits_full = torch.cat(selector_channels, dim=1)
    candidate_utilities_full = torch.cat(utility_channels, dim=1)
    return {
        "row": row,
        "selector_logits_full": selector_logits_full,
        "candidate_utilities_full": candidate_utilities_full,
        "prediction_error": primary_error,
        "prediction_errors": prediction_errors,
        "dynamics_state": dynamics_state,
        "history_validities": history_validities,
    }


def _gather_gt_log_probability(logits, gt):
    log_probability = F.log_softmax(logits.detach(), dim=1)[0]
    safe_gt = gt.clamp(0, logits.shape[1] - 1)
    return log_probability.gather(0, safe_gt.unsqueeze(0)).squeeze(0)


def _build_utility_targets(
    current_logits,
    candidate_rows,
    history_length,
    current_gt_cpu,
    previous_cv3_logits,
    teacher_full,
):
    """Return continuous semantic gains plus temporal tie-break evidence."""
    gt = current_gt_cpu.to(current_logits.device, non_blocking=True)
    valid_gt = gt != c_v5.IGNORE_LABEL
    current_log_gt = _gather_gt_log_probability(current_logits, gt)
    current_pred = current_logits.argmax(1)[0]

    gains = []
    validities = []
    history_predictions = []
    for index in range(history_length):
        if index < len(candidate_rows):
            row = candidate_rows[index]
            valid = row["valid_full"][0] & valid_gt
            history_log_gt = _gather_gt_log_probability(row["logits"], gt)
            gain = (history_log_gt - current_log_gt).clamp(
                -UTILITY_CLIP,
                UTILITY_CLIP,
            )
            gain = torch.where(valid, gain, torch.zeros_like(gain))
            history_pred = row["logits"].argmax(1)[0]
        else:
            valid = torch.zeros_like(valid_gt)
            gain = torch.zeros_like(current_log_gt)
            history_pred = current_pred
        gains.append(gain)
        validities.append(valid)
        history_predictions.append(history_pred)

    semantic_gains = torch.stack(gains, dim=0).unsqueeze(0)
    valid_mask = torch.stack(validities, dim=0).unsqueeze(0)

    # RAFT is used only for semantic-tie temporal ordering during training.
    temporal_valid = torch.zeros_like(valid_gt).unsqueeze(0).unsqueeze(0)
    current_match = torch.zeros_like(valid_gt)
    history_matches = [torch.zeros_like(valid_gt) for _ in range(history_length)]
    if teacher_full is not None and previous_cv3_logits is not None:
        teacher_previous, teacher_valid = c_v5._warp_previous_prediction_with_raft(
            previous_cv3_logits,
            teacher_full,
        )
        teacher_previous = teacher_previous[0]
        teacher_valid = teacher_valid[0].bool() & valid_gt
        temporal_valid = teacher_valid.unsqueeze(0).unsqueeze(0)
        current_match = teacher_valid & (current_pred == teacher_previous)
        history_matches = [
            teacher_valid & validities[index] & (history_predictions[index] == teacher_previous)
            for index in range(history_length)
        ]

    temporal_matches = torch.stack(
        [current_match, *history_matches],
        dim=0,
    ).unsqueeze(0)
    return {
        "semantic_gains": semantic_gains,
        "valid_mask": valid_mask,
        "temporal_matches": temporal_matches,
        "temporal_valid": temporal_valid,
    }


def _train_sequence_utility(
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
    if len(samples) < 3:
        return None
    if distill_weight > 0.0 or teacher_selector is not None:
        raise ValueError("C-V7 utility baseline does not use C-V5 teacher distillation")

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

    loss_sums = []
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
        "utility_regression_loss_sum": 0.0,
        "temporal_tiebreak_loss_sum": 0.0,
        "utility_positive": 0,
        "utility_negative": 0,
        "utility_neutral": 0,
        "temporal_rank_pairs": 0,
    }
    target_totals = c_v5._new_target_totals(selector.history_length)

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
            _, _, _, _, target_diag = c_v5._build_multiframe_target(
                c_v3_logits,
                candidate_rows,
                previous_cv3_logits,
                current_gt,
                teacher_full,
            )
            c_v5._add_target_totals(target_totals, target_diag)
            utility_target = _build_utility_targets(
                c_v3_logits,
                candidate_rows,
                selector.history_length,
                current_gt,
                previous_cv3_logits,
                teacher_full,
            )

        predicted_utility = evidence["candidate_utilities_full"]
        semantic_gains = utility_target["semantic_gains"]
        valid_mask = utility_target["valid_mask"]
        utility_loss, counts = balanced_utility_regression_loss(
            predicted_utility,
            semantic_gains,
            valid_mask,
            neutral_delta=NEUTRAL_DELTA,
            neutral_weight=NEUTRAL_WEIGHT,
        )
        temporal_loss, pair_count = semantic_first_temporal_tiebreak_loss(
            predicted_utility,
            semantic_gains,
            valid_mask,
            utility_target["temporal_matches"],
            utility_target["temporal_valid"],
            semantic_tie_delta=SEMANTIC_TIE_DELTA,
            rank_margin=TEMPORAL_RANK_MARGIN,
        )
        loss = utility_loss + TEMPORAL_RANK_WEIGHT * temporal_loss
        if bool(valid_mask.any()):
            loss_sums.append(loss)
            totals["supervised_frames"] += 1
            valid_count = int(valid_mask.sum().item())
            totals["supervised_pixels"] += valid_count
            totals["utility_regression_loss_sum"] += float(utility_loss.detach().item())
            totals["temporal_tiebreak_loss_sum"] += float(temporal_loss.detach().item())
            totals["utility_positive"] += counts["positive"]
            totals["utility_negative"] += counts["negative"]
            totals["utility_neutral"] += counts["neutral"]
            totals["temporal_rank_pairs"] += int(pair_count)

        with torch.no_grad():
            hard_selection = evidence["selector_logits_full"].argmax(1)[0]
            for index in range(selector.history_length + 1):
                totals["selected_candidate_counts"][index] += int(
                    (hard_selection == index).sum().item()
                )
            totals["prediction_error_abs"] += float(
                evidence["prediction_error"].abs().mean().item()
            )
            totals["dynamics_error_abs"] += float(dynamics_state.abs().mean().item())
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
                window_loss = torch.stack(loss_sums).mean()
                optimizer.zero_grad(set_to_none=True)
                window_loss.backward()
                optimizer.step()
                totals["optimizer_steps"] += 1
            loss_sums = []
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
    supervised_frames = max(totals["supervised_frames"], 1)
    totals["prediction_error_abs"] /= frames
    totals["dynamics_error_abs"] /= frames
    totals["utility_regression_loss"] = (
        totals.pop("utility_regression_loss_sum") / supervised_frames
    )
    totals["temporal_tiebreak_loss"] = (
        totals.pop("temporal_tiebreak_loss_sum") / supervised_frames
    )
    totals["targets"] = c_v5._target_rates(target_totals)
    totals["distillation"] = {
        "teacher": None,
        "weight": 0.0,
        "teacher_inference": False,
    }
    return totals


def _train_epoch_utility(
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
    groups,
    optimizer,
    tbptt_steps,
    distill_weight,
    distill_temperature,
):
    selector.train()
    refiner.eval()
    correction.eval()
    mask_predictor.eval()
    rows = []
    target_totals = c_v5._new_target_totals(selector.history_length)
    for samples in groups.values():
        row = _train_sequence_utility(
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
        )
        if row is not None:
            rows.append(row)
            c_v5._add_target_totals(target_totals, row["targets"])
    if not rows:
        raise RuntimeError("No valid C-V7 training sequences")

    frame_total = max(sum(row["frames"] for row in rows), 1)
    supervised_frames = max(sum(row["supervised_frames"] for row in rows), 1)
    result = {
        "frames": sum(row["frames"] for row in rows),
        "supervised_frames": sum(row["supervised_frames"] for row in rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "supervised_pixels": sum(row["supervised_pixels"] for row in rows),
        # Compatibility fields consumed by the inherited C-V6 result writer.
        "distill_pixels": 0,
        "selector_ce_per_pixel": 0.0,
        "distill_kl_per_pixel": 0.0,
        "teacher_student_agreement": 0.0,
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
        "utility_regression_loss": sum(
            row["utility_regression_loss"] * row["supervised_frames"] for row in rows
        ) / supervised_frames,
        "temporal_tiebreak_loss": sum(
            row["temporal_tiebreak_loss"] * row["supervised_frames"] for row in rows
        ) / supervised_frames,
        "utility_positive": sum(row["utility_positive"] for row in rows),
        "utility_negative": sum(row["utility_negative"] for row in rows),
        "utility_neutral": sum(row["utility_neutral"] for row in rows),
        "temporal_rank_pairs": sum(row["temporal_rank_pairs"] for row in rows),
        "targets": c_v5._target_rates(target_totals),
        "distillation": {"teacher": None, "weight": 0.0, "teacher_inference": False},
        "utility_training": {
            "target": "log p_history(GT) - log p_current(GT)",
            "clip": UTILITY_CLIP,
            "neutral_delta": NEUTRAL_DELTA,
            "positive_negative_group_balanced": True,
            "neutral_weight": NEUTRAL_WEIGHT,
            "semantic_tie_delta": SEMANTIC_TIE_DELTA,
            "temporal_rank_margin": TEMPORAL_RANK_MARGIN,
            "temporal_rank_weight": TEMPORAL_RANK_WEIGHT,
        },
    }
    return result


def _rewrite_row(row):
    if not isinstance(row, dict):
        return row
    metrics = row.get("metrics")
    if isinstance(metrics, dict) and "c_v6" in metrics:
        metrics["c_v7"] = metrics.pop("c_v6")
    delta = row.get("delta_vs_c_v3_base")
    if isinstance(delta, dict) and "c_v6" in delta:
        delta["c_v7"] = delta.pop("c_v6")
    diagnostics = row.get("diagnostics")
    if isinstance(diagnostics, dict):
        diagnostics.update(
            {
                "controller_type": "shared candidate utility estimator",
                "controller_semantic_input": (
                    "strict-validity-gated e1..eK plus compact current semantic state"
                ),
                "raw_history_probability_in_controller": False,
                "current_utility": "fixed zero",
                "history_decision_rule": "argmax positive relative utility",
                "shared_history_scorer": True,
            }
        )
    return row


def _rewrite_artifacts(output_dir, result_dir):
    result_dir = Path(result_dir)
    if (result_dir / "oracle_precheck.json").is_file():
        payload = json.loads((result_dir / "oracle_precheck.json").read_text())
        if isinstance(payload.get("metrics"), dict) and "c_v6" in payload["metrics"]:
            payload["metrics"]["c_v7"] = payload["metrics"].pop("c_v6")
        (result_dir / "oracle_precheck.json").write_text(json.dumps(payload, indent=2))

    for path in sorted(result_dir.glob("epoch_*.json")):
        row = _rewrite_row(json.loads(path.read_text()))
        path.write_text(json.dumps(row, indent=2))

    summary_path = result_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        summary["experiment"] = "C-V7 Prediction-Error Candidate Utility Estimation"
        summary["oracle_precheck"] = summary.get("oracle_precheck", {})
        oracle_metrics = summary["oracle_precheck"].get("metrics")
        if isinstance(oracle_metrics, dict) and "c_v6" in oracle_metrics:
            oracle_metrics["c_v7"] = oracle_metrics.pop("c_v6")
        summary["best"] = _rewrite_row(summary.get("best", {}))
        summary["history"] = [_rewrite_row(row) for row in summary.get("history", [])]
        summary["architecture"] = {
            "history_length": c_v5.HISTORY_LENGTH,
            "history_source": "raw detached frozen C-V3 logits",
            "candidate_bank_semantics_enter_controller": False,
            "history_semantic_interface": "strict-validity-gated prediction error e1..eK",
            "current_state": "learned compact projection of frozen C-V3 Current probability",
            "error_state": "recurrent multi-hypothesis Error State plus explicit Dynamics Error",
            "controller": "shared per-history Candidate Utility Estimator",
            "current_utility": 0.0,
            "history_logits_resampling": "one final warp per candidate",
            "controller_output_feedback": False,
            "training_loss": (
                "balanced continuous semantic-utility regression + semantic-first "
                "RAFT temporal tie-break ranking"
            ),
            "teacher_inference": False,
            "raft_inference": False,
        }
        summary["selection_rule"] = {
            "hard_constraint": "C-V7 mIoU >= frozen C-V3 Base mIoU",
            "objective_after_constraint": "maximize C-V7 mTC, then mIoU",
        }
        summary_path.write_text(json.dumps(summary, indent=2))

    checkpoint_path = Path(output_dir) / "best.pt"
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu")
        payload["experiment"] = EXPERIMENT
        metrics = payload.get("metrics")
        if isinstance(metrics, dict) and "c_v6" in metrics:
            metrics["c_v7"] = metrics.pop("c_v6")
        payload["diagnostics"] = _rewrite_row(
            {"diagnostics": payload.get("diagnostics", {})}
        )["diagnostics"]
        payload["architecture"] = {
            "controller": "shared prediction-error candidate utility estimator",
            "current_utility": 0.0,
            "shared_history_scorer": True,
            "utility_target": "log p_history(GT) - log p_current(GT)",
            "positive_negative_group_balanced": True,
            "temporal_rule": "RAFT tie-break only inside semantic tie band",
            "raft_inference": False,
        }
        torch.save(payload, checkpoint_path)


def _arg_value(argv, flag, default):
    argv = list(argv or [])
    if flag in argv:
        index = argv.index(flag)
        if index + 1 >= len(argv):
            raise ValueError(f"Missing value for {flag}")
        return argv[index + 1]
    return default


def main(argv=None):
    argv = list(argv or [])
    # The first C-V7 run is deliberately teacher-free.
    if "--distill-weight" in argv:
        index = argv.index("--distill-weight")
        if index + 1 < len(argv) and float(argv[index + 1]) != 0.0:
            raise ValueError("C-V7 baseline requires --distill-weight 0")

    output_dir = _arg_value(argv, "--output", OUTPUT_DEFAULT)
    result_dir = _arg_value(argv, "--result-output", RESULT_DEFAULT)

    original_class = c_v6.MultiHypothesisErrorSelector
    original_evidence = c_v6._selector_evidence
    original_train_epoch = c_v6._train_epoch_distilled
    original_experiment = c_v6.EXPERIMENT
    original_output = c_v6.OUTPUT_DEFAULT
    original_result = c_v6.RESULT_DEFAULT
    original_distill = c_v6.DISTILL_WEIGHT
    try:
        c_v6.MultiHypothesisErrorSelector = MultiHypothesisErrorUtilityEstimator
        c_v6._selector_evidence = _selector_evidence
        c_v6._train_epoch_distilled = _train_epoch_utility
        c_v6.EXPERIMENT = EXPERIMENT
        c_v6.OUTPUT_DEFAULT = OUTPUT_DEFAULT
        c_v6.RESULT_DEFAULT = RESULT_DEFAULT
        c_v6.DISTILL_WEIGHT = 0.0
        c_v6.main(argv)
        _rewrite_artifacts(output_dir, result_dir)
    finally:
        c_v6.MultiHypothesisErrorSelector = original_class
        c_v6._selector_evidence = original_evidence
        c_v6._train_epoch_distilled = original_train_epoch
        c_v6.EXPERIMENT = original_experiment
        c_v6.OUTPUT_DEFAULT = original_output
        c_v6.RESULT_DEFAULT = original_result
        c_v6.DISTILL_WEIGHT = original_distill


if __name__ == "__main__":
    main()
