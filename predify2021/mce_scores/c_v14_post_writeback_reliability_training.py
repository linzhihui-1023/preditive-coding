"""Training/evaluation helpers for C-V14 post-writeback reliability acceptance.

中文：C-V14 回写后可靠性接受控制训练与评测辅助模块。

C-V14 keeps C-V13 semantic proposal generation and the 0.10 feature-relative
bound, but moves a single-channel reliability map after bounded writeback.
Training = all-pixel CE + C-V4-correct protection KL + beneficial/harmful
acceptance BCE. Acceptance BCE uses detached temporal context, so it updates
only the reliability head and never backpropagates into the semantic proposal.
"""

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import VideoConsistency
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_only import FULL9, IGNORE_LABEL, NUM_CLASSES, _pair_mtc
from predify2021.mce_scores import c_v12_temporal_semantic_feature_training as c_v12
from predify2021.mce_scores import train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature


HISTORY_LENGTH = 4
SEMANTIC_CHANNELS = 128
C4_CHANNELS = 2048
RESIDUAL_SCALE = 0.10
PROTECTION_LOSS_WEIGHT = 1.0
ACCEPTANCE_LOSS_WEIGHT = 1.0
C_V4_CHECKPOINT_DEFAULT = c_v12.C_V4_CHECKPOINT_DEFAULT

host_feature_observation = c_v12.host_feature_observation
legacy_observation = c_v12.legacy_observation
build_prediction_error_and_temporal_state = c_v12.build_prediction_error_and_temporal_state
cv4_baseline_logits = c_v12.cv4_baseline_logits
formal_rescue_mask = c_v12.formal_rescue_mask


def decode_post_writeback_feature_update(
    model,
    corrector,
    observation,
    error_row,
    transportability_low,
    memory_reliability_low,
):
    row = corrector(
        prediction_errors=error_row["prediction_errors"],
        history_validities_low=error_row["history_validities_low"],
        temporal_hidden=error_row["temporal_hidden"],
        dynamics_error=error_row["dynamics_state"],
        transportability_low=transportability_low.detach(),
        memory_reliability_low=memory_reliability_low.detach(),
        current_c4=observation["c4"],
    )
    final_logits = model.decode_from_host_feature(
        HostFeature(
            row["corrected_c4"],
            observation["c1"],
            observation["output_size"],
        )
    )
    # Proposal is diagnostic/training-target only. No proposal gradient is used.
    with torch.no_grad():
        proposal_logits = model.decode_from_host_feature(
            HostFeature(
                row["proposal_c4"].detach(),
                observation["c1"],
                observation["output_size"],
            )
        )
    feature_delta_logits = final_logits - observation["host_logits"].detach()
    return final_logits, proposal_logits.detach(), feature_delta_logits, row


def protection_kl_loss(final_logits, c_v4_logits, gt_gpu):
    """Protect pixels where frozen C-V4 is already GT-correct; GT is training-only."""
    valid = gt_gpu.ne(IGNORE_LABEL)
    c_v4_pred = c_v4_logits.argmax(1)[0]
    protect_mask = valid & c_v4_pred.eq(gt_gpu)
    if int(protect_mask.sum().item()) == 0:
        return final_logits.sum() * 0.0, protect_mask
    teacher = F.softmax(c_v4_logits.detach(), dim=1)
    student_log = F.log_softmax(final_logits, dim=1)
    kl_map = F.kl_div(student_log, teacher, reduction="none").sum(dim=1)[0]
    return kl_map[protect_mask].mean(), protect_mask


@torch.no_grad()
def proposal_acceptance_targets(host_logits, proposal_logits, gt_gpu):
    """Training-only labels from the semantic proposal effect.

    positive: current Host wrong, full bounded proposal correct.
    negative: current Host correct, full bounded proposal wrong.
    all other pixels are ignored.
    """
    valid = gt_gpu.ne(IGNORE_LABEL)
    host_pred = host_logits.argmax(1)[0]
    proposal_pred = proposal_logits.argmax(1)[0]
    beneficial = valid & host_pred.ne(gt_gpu) & proposal_pred.eq(gt_gpu)
    harmful = valid & host_pred.eq(gt_gpu) & proposal_pred.ne(gt_gpu)
    supervised = beneficial | harmful
    target = beneficial.to(host_logits.dtype)
    return target, supervised, beneficial, harmful


def acceptance_bce_loss(corrector, correction_row, output_size, target, supervised):
    """Auxiliary acceptance BCE: gradients update reliability_head only."""
    if int(supervised.sum().item()) == 0:
        zero = correction_row["reliability_logit"].sum() * 0.0
        return zero, None
    aux_low = corrector.reliability_logits_from_detached_context(
        correction_row["temporal_latent"]
    )
    aux_full = F.interpolate(
        aux_low,
        size=tuple(output_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    bce = F.binary_cross_entropy_with_logits(
        aux_full[supervised],
        target[supervised],
        reduction="mean",
    )
    return bce, aux_full


def _initial_state(model, observer, residual, samples):
    if len(samples) < 3:
        return None
    frame0 = host_feature_observation(model, samples[0])
    frame1 = host_feature_observation(model, samples[1])
    pending_motion, motion_hidden = c_v5._initialize_motion(
        observer,
        residual,
        legacy_observation(frame0),
        legacy_observation(frame1),
    )
    return frame0, frame1, pending_motion, motion_hidden


def train_sequence(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    c_v4_controller,
    dynamics,
    corrector,
    samples,
    optimizer,
    gradient_accumulation_steps,
):
    initial = _initial_state(model, observer, residual, samples)
    if initial is None:
        return None
    _, previous, pending_motion, motion_hidden = initial
    previous_host_logits = previous["host_logits"].detach()
    previous_low = previous["host_low"].detach()
    previous_c1 = previous["c1"].detach()
    raw_history = [previous_host_logits]
    motion_history = []
    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = controller_hidden = dynamics_state = None

    optimizer.zero_grad(set_to_none=True)
    frames_in_window = 0
    totals = {
        "frames": 0,
        "optimizer_steps": 0,
        "final_ce": 0.0,
        "protection_kl": 0.0,
        "acceptance_bce": 0.0,
        "acceptance_supervised_pixels": 0,
        "acceptance_positive_pixels": 0,
        "acceptance_negative_pixels": 0,
        "positive_reliability_sum": 0.0,
        "negative_reliability_sum": 0.0,
        "hard_positive_accepted": 0,
        "hard_negative_rejected": 0,
        "rescue_pixels": 0,
        "rescue_recovered": 0,
        "baseline_correct_pixels": 0,
        "baseline_correct_damaged": 0,
        "delta_c4_abs": 0.0,
        "delta_c4_rms_ratio": 0.0,
        "bounded_semantic_delta_rms_ratio": 0.0,
        "raw_semantic_delta_c4_abs": 0.0,
        "feature_delta_logits_abs": 0.0,
        "reliability_mean": 0.0,
        "reliability_std": 0.0,
    }

    for frame_index in range(2, len(samples)):
        observation = host_feature_observation(model, samples[frame_index])
        gt_cpu = semantic_mask_from_panoptic_png(samples[frame_index]["mask_path"])

        with torch.no_grad():
            prior_low, _ = c_v5.warp_low_logits(previous_low, pending_motion)
            e1 = c_v5._frozen_e1_step(
                correction, mask_predictor, observation["c1"], observation["host_low"],
                prior_low, pending_motion, semantic_state_low,
                transport_hidden, semantic_hidden, mask_hidden,
            )
            transport_hidden = e1["transport_hidden"]
            semantic_hidden = e1["semantic_hidden"]
            mask_hidden = e1["mask_hidden"]
            semantic_state_low = e1["semantic_state_low"]
            memory_row, memory_state, _, c_v3_logits = c_v5._frozen_cv3_step(
                refiner, observation["c1"], observation["host_low"], prior_low,
                e1, pending_motion, memory_state, observation["output_size"],
                observation["host_logits"],
            )
            candidate_rows = c_v5._build_history_candidates(
                raw_history, motion_history, pending_motion, corrector.history_length
            )
            error_row = build_prediction_error_and_temporal_state(
                c_v4_controller, dynamics, c_v3_logits, candidate_rows,
                pending_motion, e1["transportability_low"],
                memory_row["memory_reliability"], controller_hidden,
                dynamics_state, corrector.history_length,
            )
            controller_hidden = error_row["temporal_hidden"]
            dynamics_state = error_row["dynamics_state"]
            c_v4_logits, _ = cv4_baseline_logits(
                c_v3_logits, candidate_rows, error_row["temporal_row"]
            )
            rescue = formal_rescue_mask(c_v4_logits, candidate_rows, gt_cpu)

        final_logits, proposal_logits, feature_delta_logits, correction_row = (
            decode_post_writeback_feature_update(
                model, corrector, observation, error_row,
                e1["transportability_low"], memory_row["memory_reliability"],
            )
        )
        target = gt_cpu.to(final_logits.device, non_blocking=True).long().unsqueeze(0)
        valid = target[0].ne(IGNORE_LABEL)
        pixel_ce = F.cross_entropy(
            final_logits, target, ignore_index=IGNORE_LABEL, reduction="none"
        )[0]
        final_ce = pixel_ce[valid].mean()
        protect_kl, baseline_correct = protection_kl_loss(
            final_logits, c_v4_logits, target[0]
        )

        acceptance_target, acceptance_mask, beneficial, harmful = proposal_acceptance_targets(
            observation["host_logits"].detach(), proposal_logits, target[0]
        )
        accept_bce, _ = acceptance_bce_loss(
            corrector,
            correction_row,
            observation["output_size"],
            acceptance_target,
            acceptance_mask,
        )
        loss = (
            final_ce
            + PROTECTION_LOSS_WEIGHT * protect_kl
            + ACCEPTANCE_LOSS_WEIGHT * accept_bce
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite C-V14 post-writeback reliability loss")
        loss.backward()
        frames_in_window += 1

        with torch.no_grad():
            final_pred = final_logits.argmax(1)[0]
            reliability_full = F.interpolate(
                correction_row["reliability"],
                size=observation["output_size"],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            positive_count = int(beneficial.sum().item())
            negative_count = int(harmful.sum().item())
            totals["frames"] += 1
            totals["final_ce"] += float(final_ce.item())
            totals["protection_kl"] += float(protect_kl.item())
            totals["acceptance_bce"] += float(accept_bce.item())
            totals["acceptance_supervised_pixels"] += int(acceptance_mask.sum().item())
            totals["acceptance_positive_pixels"] += positive_count
            totals["acceptance_negative_pixels"] += negative_count
            totals["positive_reliability_sum"] += float(reliability_full[beneficial].sum().item()) if positive_count else 0.0
            totals["negative_reliability_sum"] += float(reliability_full[harmful].sum().item()) if negative_count else 0.0
            totals["hard_positive_accepted"] += int((beneficial & reliability_full.gt(0.5)).sum().item())
            totals["hard_negative_rejected"] += int((harmful & reliability_full.le(0.5)).sum().item())
            totals["rescue_pixels"] += int(rescue.sum().item())
            totals["rescue_recovered"] += int((rescue & final_pred.eq(target[0])).sum().item())
            totals["baseline_correct_pixels"] += int(baseline_correct.sum().item())
            totals["baseline_correct_damaged"] += int(
                (baseline_correct & final_pred.ne(target[0])).sum().item()
            )
            current = observation["c4"]
            final_delta = correction_row["delta_c4"]
            proposal_delta = correction_row["bounded_semantic_delta_c4"]
            current_rms = current.square().mean().sqrt().clamp_min(1e-8)
            totals["delta_c4_abs"] += float(final_delta.abs().mean().item())
            totals["delta_c4_rms_ratio"] += float(final_delta.square().mean().sqrt().div(current_rms).item())
            totals["bounded_semantic_delta_rms_ratio"] += float(proposal_delta.square().mean().sqrt().div(current_rms).item())
            totals["raw_semantic_delta_c4_abs"] += float(correction_row["raw_semantic_delta_c4"].abs().mean().item())
            totals["feature_delta_logits_abs"] += float(feature_delta_logits.abs().mean().item())
            totals["reliability_mean"] += float(correction_row["reliability"].mean().item())
            totals["reliability_std"] += float(correction_row["reliability"].std().item())

            observed_motion = c_v5._observe_motion(
                observer, previous_low, previous_c1,
                observation["host_low"], observation["c1"],
            )
            motion_error = F.softmax(observation["host_low"], dim=1) - F.softmax(prior_low, dim=1)
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion, motion_error, motion_hidden
            )

        boundary = frames_in_window >= gradient_accumulation_steps or frame_index == len(samples) - 1
        if boundary:
            for parameter in corrector.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(float(frames_in_window))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            totals["optimizer_steps"] += 1
            frames_in_window = 0

        raw_history.insert(0, c_v3_logits.detach())
        raw_history = raw_history[: corrector.history_length]
        motion_history.insert(0, pending_motion.detach())
        motion_history = motion_history[: max(corrector.history_length - 1, 0)]
        previous_low = observation["host_low"].detach()
        previous_c1 = observation["c1"].detach()
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()
        memory_state = memory_state.detach() if memory_state is not None else None
        controller_hidden = controller_hidden.detach() if controller_hidden is not None else None
        dynamics_state = dynamics_state.detach() if dynamics_state is not None else None

    frames = max(totals["frames"], 1)
    for key in (
        "final_ce", "protection_kl", "acceptance_bce", "delta_c4_abs",
        "delta_c4_rms_ratio", "bounded_semantic_delta_rms_ratio",
        "raw_semantic_delta_c4_abs", "feature_delta_logits_abs",
        "reliability_mean", "reliability_std",
    ):
        totals[key] /= frames
    totals["rescue_recovery_rate"] = totals["rescue_recovered"] / max(totals["rescue_pixels"], 1)
    totals["baseline_correct_damage_rate"] = totals["baseline_correct_damaged"] / max(totals["baseline_correct_pixels"], 1)
    totals["positive_reliability_mean"] = totals["positive_reliability_sum"] / max(totals["acceptance_positive_pixels"], 1)
    totals["negative_reliability_mean"] = totals["negative_reliability_sum"] / max(totals["acceptance_negative_pixels"], 1)
    totals["hard_positive_accept_recall"] = totals["hard_positive_accepted"] / max(totals["acceptance_positive_pixels"], 1)
    totals["hard_negative_reject_rate"] = totals["hard_negative_rejected"] / max(totals["acceptance_negative_pixels"], 1)
    return totals


def train_epoch(
    model, observer, residual, correction, mask_predictor, refiner,
    c_v4_controller, dynamics, corrector, groups, optimizer,
    gradient_accumulation_steps,
):
    model.eval(); refiner.eval(); correction.eval(); mask_predictor.eval(); c_v4_controller.eval()
    corrector.train()
    rows = []
    for samples in groups.values():
        row = train_sequence(
            model, observer, residual, correction, mask_predictor, refiner,
            c_v4_controller, dynamics, corrector, samples, optimizer,
            gradient_accumulation_steps,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("No valid C-V14 training sequences")

    frame_total = max(sum(row["frames"] for row in rows), 1)
    sum_keys = (
        "frames", "optimizer_steps", "acceptance_supervised_pixels",
        "acceptance_positive_pixels", "acceptance_negative_pixels",
        "hard_positive_accepted", "hard_negative_rejected",
        "rescue_pixels", "rescue_recovered", "baseline_correct_pixels",
        "baseline_correct_damaged",
    )
    result = {key: sum(row[key] for row in rows) for key in sum_keys}
    result.update({
        "protection_loss_weight": PROTECTION_LOSS_WEIGHT,
        "acceptance_loss_weight": ACCEPTANCE_LOSS_WEIGHT,
        "rescue_supervision": False,
        "acceptance_supervision": "Host wrong->proposal correct positive; Host correct->proposal wrong negative; others ignored",
    })
    for key in (
        "final_ce", "protection_kl", "acceptance_bce", "delta_c4_abs",
        "delta_c4_rms_ratio", "bounded_semantic_delta_rms_ratio",
        "raw_semantic_delta_c4_abs", "feature_delta_logits_abs",
        "reliability_mean", "reliability_std",
    ):
        result[key] = sum(row[key] * row["frames"] for row in rows) / frame_total
    positive_sum = sum(row["positive_reliability_sum"] for row in rows)
    negative_sum = sum(row["negative_reliability_sum"] for row in rows)
    result["rescue_recovery_rate"] = result["rescue_recovered"] / max(result["rescue_pixels"], 1)
    result["baseline_correct_damage_rate"] = result["baseline_correct_damaged"] / max(result["baseline_correct_pixels"], 1)
    result["positive_reliability_mean"] = positive_sum / max(result["acceptance_positive_pixels"], 1)
    result["negative_reliability_mean"] = negative_sum / max(result["acceptance_negative_pixels"], 1)
    result["hard_positive_accept_recall"] = result["hard_positive_accepted"] / max(result["acceptance_positive_pixels"], 1)
    result["hard_negative_reject_rate"] = result["hard_negative_rejected"] / max(result["acceptance_negative_pixels"], 1)
    return result


@torch.inference_mode()
def evaluate(
    model, observer, residual, correction, mask_predictor, refiner,
    c_v4_controller, dynamics, corrector, groups, raft_metric,
):
    names = ("host", "c_v3_base", "c_v4_frozen", "c_v14")
    confusion = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in names}
    mtc_sum = {name: 0.0 for name in names}
    mtc_count = {name: 0 for name in names}
    vc_sum = {name: {8: 0.0, 16: 0.0} for name in names}
    vc_count = {name: {8: 0, 16: 0} for name in names}
    diag = {
        "correction_frames": 0,
        "delta_c4_abs": 0.0,
        "delta_c4_rms_ratio": 0.0,
        "bounded_semantic_delta_rms_ratio": 0.0,
        "raw_semantic_delta_c4_abs": 0.0,
        "feature_delta_logits_abs": 0.0,
        "reliability_mean": 0.0,
        "reliability_std": 0.0,
        "acceptance_positive_pixels": 0,
        "acceptance_negative_pixels": 0,
        "positive_reliability_sum": 0.0,
        "negative_reliability_sum": 0.0,
        "hard_positive_accepted": 0,
        "hard_negative_rejected": 0,
        "rescue_pixels": 0,
        "rescue_recovered": 0,
        "baseline_correct_pixels": 0,
        "baseline_correct_damaged": 0,
    }
    corrector.eval(); c_v4_controller.eval(); model.eval()

    for sequence in FULL9:
        samples = groups[sequence]
        if len(samples) < 2:
            continue
        first = host_feature_observation(model, samples[0])
        second = host_feature_observation(model, samples[1])
        seq_vc = {name: VideoConsistency() for name in names}
        previous_predictions = None

        for index, observation in enumerate((first, second)):
            gt_cpu = semantic_mask_from_panoptic_png(samples[index]["mask_path"])
            host_pred = observation["host_logits"].argmax(1)
            predictions = {name: host_pred for name in names}
            for name, prediction in predictions.items():
                pred_cpu = prediction[0].cpu()
                update_confusion_matrix(confusion[name], pred_cpu, gt_cpu)
                seq_vc[name].update(gt_cpu, pred_cpu)
            if previous_predictions is not None:
                teacher = raft_metric.current_to_previous(observation["image"], first["image"])
                for name, prediction in predictions.items():
                    score = _pair_mtc(previous_predictions[name], prediction, teacher)
                    if torch.isfinite(torch.tensor(score)):
                        mtc_sum[name] += score; mtc_count[name] += 1
            previous_predictions = {name: pred.detach() for name, pred in predictions.items()}

        pending_motion, motion_hidden = c_v5._initialize_motion(
            observer, residual, legacy_observation(first), legacy_observation(second)
        )
        previous = second
        previous_low = second["host_low"].detach()
        previous_c1 = second["c1"].detach()
        raw_history = [second["host_logits"].detach()]
        motion_history = []
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = torch.zeros_like(previous_low)
        memory_state = controller_hidden = dynamics_state = None

        for frame_index in range(2, len(samples)):
            observation = host_feature_observation(model, samples[frame_index])
            gt_cpu = semantic_mask_from_panoptic_png(samples[frame_index]["mask_path"])
            prior_low, _ = c_v5.warp_low_logits(previous_low, pending_motion)
            e1 = c_v5._frozen_e1_step(
                correction, mask_predictor, observation["c1"], observation["host_low"],
                prior_low, pending_motion, semantic_state_low,
                transport_hidden, semantic_hidden, mask_hidden,
            )
            transport_hidden = e1["transport_hidden"]
            semantic_hidden = e1["semantic_hidden"]
            mask_hidden = e1["mask_hidden"]
            semantic_state_low = e1["semantic_state_low"]
            memory_row, memory_state, _, c_v3_logits = c_v5._frozen_cv3_step(
                refiner, observation["c1"], observation["host_low"], prior_low,
                e1, pending_motion, memory_state, observation["output_size"],
                observation["host_logits"],
            )
            candidate_rows = c_v5._build_history_candidates(
                raw_history, motion_history, pending_motion, corrector.history_length
            )
            error_row = build_prediction_error_and_temporal_state(
                c_v4_controller, dynamics, c_v3_logits, candidate_rows,
                pending_motion, e1["transportability_low"],
                memory_row["memory_reliability"], controller_hidden,
                dynamics_state, corrector.history_length,
            )
            controller_hidden = error_row["temporal_hidden"]
            dynamics_state = error_row["dynamics_state"]
            c_v4_logits, _ = cv4_baseline_logits(
                c_v3_logits, candidate_rows, error_row["temporal_row"]
            )
            final_logits, proposal_logits, feature_delta_logits, correction_row = (
                decode_post_writeback_feature_update(
                    model, corrector, observation, error_row,
                    e1["transportability_low"], memory_row["memory_reliability"],
                )
            )
            predictions = {
                "host": observation["host_logits"].argmax(1),
                "c_v3_base": c_v3_logits.argmax(1),
                "c_v4_frozen": c_v4_logits.argmax(1),
                "c_v14": final_logits.argmax(1),
            }
            for name, prediction in predictions.items():
                pred_cpu = prediction[0].cpu()
                update_confusion_matrix(confusion[name], pred_cpu, gt_cpu)
                seq_vc[name].update(gt_cpu, pred_cpu)

            teacher = raft_metric.current_to_previous(observation["image"], previous["image"])
            for name, prediction in predictions.items():
                score = _pair_mtc(previous_predictions[name], prediction, teacher)
                if torch.isfinite(torch.tensor(score)):
                    mtc_sum[name] += score; mtc_count[name] += 1
            previous_predictions = {name: pred.detach() for name, pred in predictions.items()}

            rescue = formal_rescue_mask(c_v4_logits, candidate_rows, gt_cpu)
            gt_gpu = gt_cpu.to(final_logits.device, non_blocking=True)
            valid = gt_gpu.ne(IGNORE_LABEL)
            base_pred = c_v4_logits.argmax(1)[0]
            final_pred = final_logits.argmax(1)[0]
            baseline_correct = valid & base_pred.eq(gt_gpu)
            _, _, beneficial, harmful = proposal_acceptance_targets(
                observation["host_logits"].detach(), proposal_logits, gt_gpu
            )
            reliability_full = F.interpolate(
                correction_row["reliability"],
                size=observation["output_size"],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            positive_count = int(beneficial.sum().item())
            negative_count = int(harmful.sum().item())

            diag["correction_frames"] += 1
            diag["acceptance_positive_pixels"] += positive_count
            diag["acceptance_negative_pixels"] += negative_count
            diag["positive_reliability_sum"] += float(reliability_full[beneficial].sum().item()) if positive_count else 0.0
            diag["negative_reliability_sum"] += float(reliability_full[harmful].sum().item()) if negative_count else 0.0
            diag["hard_positive_accepted"] += int((beneficial & reliability_full.gt(0.5)).sum().item())
            diag["hard_negative_rejected"] += int((harmful & reliability_full.le(0.5)).sum().item())
            diag["rescue_pixels"] += int(rescue.sum().item())
            diag["rescue_recovered"] += int((rescue & final_pred.eq(gt_gpu)).sum().item())
            diag["baseline_correct_pixels"] += int(baseline_correct.sum().item())
            diag["baseline_correct_damaged"] += int((baseline_correct & final_pred.ne(gt_gpu)).sum().item())
            current = observation["c4"]
            current_rms = current.square().mean().sqrt().clamp_min(1e-8)
            final_delta = correction_row["delta_c4"]
            proposal_delta = correction_row["bounded_semantic_delta_c4"]
            diag["delta_c4_abs"] += float(final_delta.abs().mean().item())
            diag["delta_c4_rms_ratio"] += float(final_delta.square().mean().sqrt().div(current_rms).item())
            diag["bounded_semantic_delta_rms_ratio"] += float(proposal_delta.square().mean().sqrt().div(current_rms).item())
            diag["raw_semantic_delta_c4_abs"] += float(correction_row["raw_semantic_delta_c4"].abs().mean().item())
            diag["feature_delta_logits_abs"] += float(feature_delta_logits.abs().mean().item())
            diag["reliability_mean"] += float(correction_row["reliability"].mean().item())
            diag["reliability_std"] += float(correction_row["reliability"].std().item())

            observed_motion = c_v5._observe_motion(
                observer, previous_low, previous_c1,
                observation["host_low"], observation["c1"],
            )
            motion_error = F.softmax(observation["host_low"], dim=1) - F.softmax(prior_low, dim=1)
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion, motion_error, motion_hidden
            )
            raw_history.insert(0, c_v3_logits.detach())
            raw_history = raw_history[: corrector.history_length]
            motion_history.insert(0, pending_motion.detach())
            motion_history = motion_history[: max(corrector.history_length - 1, 0)]
            previous = observation
            previous_low = observation["host_low"].detach()
            previous_c1 = observation["c1"].detach()
            pending_motion = next_motion.detach()
            motion_hidden = next_motion_hidden.detach()
            memory_state = memory_state.detach() if memory_state is not None else None
            controller_hidden = controller_hidden.detach() if controller_hidden is not None else None
            dynamics_state = dynamics_state.detach() if dynamics_state is not None else None

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
    frames = max(diag["correction_frames"], 1)
    for key in (
        "delta_c4_abs", "delta_c4_rms_ratio", "bounded_semantic_delta_rms_ratio",
        "raw_semantic_delta_c4_abs", "feature_delta_logits_abs",
        "reliability_mean", "reliability_std",
    ):
        diag[key] /= frames
    diag["rescue_recovery_rate"] = diag["rescue_recovered"] / max(diag["rescue_pixels"], 1)
    diag["baseline_correct_damage_rate"] = diag["baseline_correct_damaged"] / max(diag["baseline_correct_pixels"], 1)
    diag["positive_reliability_mean"] = diag["positive_reliability_sum"] / max(diag["acceptance_positive_pixels"], 1)
    diag["negative_reliability_mean"] = diag["negative_reliability_sum"] / max(diag["acceptance_negative_pixels"], 1)
    diag["hard_positive_accept_recall"] = diag["hard_positive_accepted"] / max(diag["acceptance_positive_pixels"], 1)
    diag["hard_negative_reject_rate"] = diag["hard_negative_rejected"] / max(diag["acceptance_negative_pixels"], 1)
    diag.update({
        "temporal_base": "frozen C-V4 Stateful Semantic Hysteresis",
        "history_length": corrector.history_length,
        "history_source": "raw detached frozen C-V3 logits",
        "semantic_correction_target": "current Host c4 2048D feature",
        "semantic_correction_source": "concat validity-gated e1..e4 only",
        "bounded_residual_scale": corrector.residual_scale,
        "reliability_shape": "single-channel pixel-wise",
        "reliability_position": "after bounded semantic writeback",
        "final_composition": "Decoder(c4 + reliability * bounded semantic Delta-c4)",
        "c_v4_role": "frozen temporal reference/teacher + hidden state; not logit composition",
        "controller_output_feedback": False,
        "corrected_output_history_feedback": False,
        "rescue_supervision": False,
        "acceptance_supervision": "training-only beneficial/harmful proposal target",
        "raft_training": False,
        "raft_metric_only": True,
    })
    return metrics, diag
