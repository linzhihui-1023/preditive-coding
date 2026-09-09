"""Training/evaluation helpers for C-V14 post-writeback reliability acceptance.

中文：C-V14 后回写可靠性接受训练与评测辅助模块。

C-V14 keeps C-V13's learned semantic proposal path and 0.10 bounded c4 residual,
but moves temporal reliability after the bounded writeback. A one-channel
Acceptance Head（接受头）is trained with explicit beneficial/harmful proposal labels.

Training:
    L = all-pixel segmentation CE
      + C-V4-correct-region protection KL
      + acceptance BCE.

Acceptance BCE uses GT only to create training labels and its temporal context is
detached, so this auxiliary loss updates only the Acceptance Head. There is no
Rescue CE, temporal loss, or RAFT training teacher.
"""

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import VideoConsistency
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_only import FULL9, IGNORE_LABEL, NUM_CLASSES, _pair_mtc
from predify2021.mce_scores import c_v13_bounded_reliability_feature_training as c_v13
from predify2021.mce_scores import train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature


HISTORY_LENGTH = 4
SEMANTIC_CHANNELS = 128
C4_CHANNELS = 2048
RESIDUAL_SCALE = 0.10
PROTECTION_LOSS_WEIGHT = 1.0
ACCEPTANCE_LOSS_WEIGHT = 1.0
C_V4_CHECKPOINT_DEFAULT = c_v13.C_V4_CHECKPOINT_DEFAULT
C_V13_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v13_bounded_reliability_feature_update/best.pt"
)

host_feature_observation = c_v13.host_feature_observation
legacy_observation = c_v13.legacy_observation
build_prediction_error_and_temporal_state = c_v13.build_prediction_error_and_temporal_state
cv4_baseline_logits = c_v13.cv4_baseline_logits
formal_rescue_mask = c_v13.formal_rescue_mask
protection_kl_loss = c_v13.protection_kl_loss


def decode_postwriteback_reliability(
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
    proposal_logits = model.decode_from_host_feature(
        HostFeature(
            row["proposal_corrected_c4"],
            observation["c1"],
            observation["output_size"],
        )
    )
    final_logits = model.decode_from_host_feature(
        HostFeature(
            row["corrected_c4"],
            observation["c1"],
            observation["output_size"],
        )
    )
    return final_logits, proposal_logits, row


def acceptance_targets(host_logits, proposal_logits, gt_gpu):
    """Training-only labels for whether the bounded proposal should be accepted."""
    valid = gt_gpu.ne(IGNORE_LABEL)
    host_pred = host_logits.detach().argmax(1)[0]
    proposal_pred = proposal_logits.detach().argmax(1)[0]
    beneficial = valid & host_pred.ne(gt_gpu) & proposal_pred.eq(gt_gpu)
    harmful = valid & host_pred.eq(gt_gpu) & proposal_pred.ne(gt_gpu)
    supervised = beneficial | harmful
    target = beneficial.to(proposal_logits.dtype)
    return target, supervised, beneficial, harmful


def acceptance_bce_loss(
    acceptance_logit_aux,
    host_logits,
    proposal_logits,
    gt_gpu,
):
    target, supervised, beneficial, harmful = acceptance_targets(
        host_logits,
        proposal_logits,
        gt_gpu,
    )
    full_logit = F.interpolate(
        acceptance_logit_aux,
        size=tuple(gt_gpu.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    supervised_pixels = int(supervised.sum().item())
    if supervised_pixels == 0:
        loss = full_logit.sum() * 0.0
    else:
        loss = F.binary_cross_entropy_with_logits(
            full_logit[supervised],
            target[supervised],
        )
    return loss, {
        "target": target,
        "supervised": supervised,
        "beneficial": beneficial,
        "harmful": harmful,
        "full_logit": full_logit,
    }


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


def _new_totals():
    return {
        "frames": 0,
        "optimizer_steps": 0,
        "final_ce": 0.0,
        "protection_kl": 0.0,
        "acceptance_bce": 0.0,
        "rescue_pixels": 0,
        "proposal_rescue_recovered": 0,
        "final_rescue_recovered": 0,
        "baseline_correct_pixels": 0,
        "proposal_baseline_correct_damaged": 0,
        "final_baseline_correct_damaged": 0,
        "acceptance_supervised_pixels": 0,
        "acceptance_beneficial_pixels": 0,
        "acceptance_harmful_pixels": 0,
        "acceptance_beneficial_sum": 0.0,
        "acceptance_harmful_sum": 0.0,
        "proposal_delta_c4_abs": 0.0,
        "proposal_delta_c4_rms_ratio": 0.0,
        "final_delta_c4_abs": 0.0,
        "final_delta_c4_rms_ratio": 0.0,
        "acceptance_mean": 0.0,
        "acceptance_std": 0.0,
    }


def _accumulate_diagnostics(
    totals,
    observation,
    c_v4_logits,
    proposal_logits,
    final_logits,
    correction_row,
    rescue,
    gt_gpu,
    acceptance_info,
):
    with torch.no_grad():
        proposal_pred = proposal_logits.argmax(1)[0]
        final_pred = final_logits.argmax(1)[0]
        c_v4_pred = c_v4_logits.argmax(1)[0]
        valid = gt_gpu.ne(IGNORE_LABEL)
        baseline_correct = valid & c_v4_pred.eq(gt_gpu)

        totals["frames"] += 1
        totals["rescue_pixels"] += int(rescue.sum().item())
        totals["proposal_rescue_recovered"] += int(
            (rescue & proposal_pred.eq(gt_gpu)).sum().item()
        )
        totals["final_rescue_recovered"] += int(
            (rescue & final_pred.eq(gt_gpu)).sum().item()
        )
        totals["baseline_correct_pixels"] += int(baseline_correct.sum().item())
        totals["proposal_baseline_correct_damaged"] += int(
            (baseline_correct & proposal_pred.ne(gt_gpu)).sum().item()
        )
        totals["final_baseline_correct_damaged"] += int(
            (baseline_correct & final_pred.ne(gt_gpu)).sum().item()
        )

        beneficial = acceptance_info["beneficial"]
        harmful = acceptance_info["harmful"]
        full_acceptance = torch.sigmoid(acceptance_info["full_logit"])
        beneficial_pixels = int(beneficial.sum().item())
        harmful_pixels = int(harmful.sum().item())
        totals["acceptance_supervised_pixels"] += beneficial_pixels + harmful_pixels
        totals["acceptance_beneficial_pixels"] += beneficial_pixels
        totals["acceptance_harmful_pixels"] += harmful_pixels
        if beneficial_pixels:
            totals["acceptance_beneficial_sum"] += float(full_acceptance[beneficial].sum().item())
        if harmful_pixels:
            totals["acceptance_harmful_sum"] += float(full_acceptance[harmful].sum().item())

        proposal_delta = correction_row["proposal_delta_c4"]
        final_delta = correction_row["final_delta_c4"]
        current = observation["c4"]
        current_rms = current.square().mean().sqrt().clamp_min(1e-8)
        totals["proposal_delta_c4_abs"] += float(proposal_delta.abs().mean().item())
        totals["proposal_delta_c4_rms_ratio"] += float(
            proposal_delta.square().mean().sqrt().div(current_rms).item()
        )
        totals["final_delta_c4_abs"] += float(final_delta.abs().mean().item())
        totals["final_delta_c4_rms_ratio"] += float(
            final_delta.square().mean().sqrt().div(current_rms).item()
        )
        totals["acceptance_mean"] += float(correction_row["acceptance"].mean().item())
        totals["acceptance_std"] += float(correction_row["acceptance"].std().item())


def _finish_totals(totals):
    frames = max(totals["frames"], 1)
    for key in (
        "final_ce",
        "protection_kl",
        "acceptance_bce",
        "proposal_delta_c4_abs",
        "proposal_delta_c4_rms_ratio",
        "final_delta_c4_abs",
        "final_delta_c4_rms_ratio",
        "acceptance_mean",
        "acceptance_std",
    ):
        totals[key] /= frames
    totals["proposal_rescue_recovery_rate"] = (
        totals["proposal_rescue_recovered"] / max(totals["rescue_pixels"], 1)
    )
    totals["final_rescue_recovery_rate"] = (
        totals["final_rescue_recovered"] / max(totals["rescue_pixels"], 1)
    )
    totals["proposal_baseline_correct_damage_rate"] = (
        totals["proposal_baseline_correct_damaged"]
        / max(totals["baseline_correct_pixels"], 1)
    )
    totals["final_baseline_correct_damage_rate"] = (
        totals["final_baseline_correct_damaged"]
        / max(totals["baseline_correct_pixels"], 1)
    )
    totals["acceptance_mean_on_beneficial"] = (
        totals["acceptance_beneficial_sum"]
        / max(totals["acceptance_beneficial_pixels"], 1)
    )
    totals["acceptance_mean_on_harmful"] = (
        totals["acceptance_harmful_sum"]
        / max(totals["acceptance_harmful_pixels"], 1)
    )
    return totals


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
    totals = _new_totals()

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

        final_logits, proposal_logits, correction_row = decode_postwriteback_reliability(
            model, corrector, observation, error_row,
            e1["transportability_low"], memory_row["memory_reliability"],
        )
        target = gt_cpu.to(final_logits.device, non_blocking=True).long().unsqueeze(0)
        valid = target[0].ne(IGNORE_LABEL)
        pixel_ce = F.cross_entropy(
            final_logits,
            target,
            ignore_index=IGNORE_LABEL,
            reduction="none",
        )[0]
        final_ce = pixel_ce[valid].mean()
        protect_kl, _ = protection_kl_loss(final_logits, c_v4_logits, target[0])
        accept_bce, acceptance_info = acceptance_bce_loss(
            correction_row["acceptance_logit_aux"],
            observation["host_logits"],
            proposal_logits,
            target[0],
        )
        loss = (
            final_ce
            + PROTECTION_LOSS_WEIGHT * protect_kl
            + ACCEPTANCE_LOSS_WEIGHT * accept_bce
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite C-V14 acceptance-controlled feature loss")
        loss.backward()
        frames_in_window += 1

        totals["final_ce"] += float(final_ce.item())
        totals["protection_kl"] += float(protect_kl.item())
        totals["acceptance_bce"] += float(accept_bce.item())
        _accumulate_diagnostics(
            totals,
            observation,
            c_v4_logits,
            proposal_logits,
            final_logits,
            correction_row,
            rescue,
            target[0],
            acceptance_info,
        )

        with torch.no_grad():
            observed_motion = c_v5._observe_motion(
                observer, previous_low, previous_c1,
                observation["host_low"], observation["c1"],
            )
            motion_error = F.softmax(observation["host_low"], dim=1) - F.softmax(prior_low, dim=1)
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion, motion_error, motion_hidden
            )

        boundary = (
            frames_in_window >= gradient_accumulation_steps
            or frame_index == len(samples) - 1
        )
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

    return _finish_totals(totals)


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

    result = _new_totals()
    result["frames"] = sum(row["frames"] for row in rows)
    result["optimizer_steps"] = sum(row["optimizer_steps"] for row in rows)
    integer_keys = (
        "rescue_pixels", "proposal_rescue_recovered", "final_rescue_recovered",
        "baseline_correct_pixels", "proposal_baseline_correct_damaged",
        "final_baseline_correct_damaged", "acceptance_supervised_pixels",
        "acceptance_beneficial_pixels", "acceptance_harmful_pixels",
    )
    for key in integer_keys:
        result[key] = sum(row[key] for row in rows)
    result["acceptance_beneficial_sum"] = sum(row["acceptance_beneficial_sum"] for row in rows)
    result["acceptance_harmful_sum"] = sum(row["acceptance_harmful_sum"] for row in rows)

    frame_total = max(result["frames"], 1)
    average_keys = (
        "final_ce", "protection_kl", "acceptance_bce",
        "proposal_delta_c4_abs", "proposal_delta_c4_rms_ratio",
        "final_delta_c4_abs", "final_delta_c4_rms_ratio",
        "acceptance_mean", "acceptance_std",
    )
    for key in average_keys:
        result[key] = sum(row[key] * row["frames"] for row in rows) / frame_total

    result["proposal_rescue_recovery_rate"] = (
        result["proposal_rescue_recovered"] / max(result["rescue_pixels"], 1)
    )
    result["final_rescue_recovery_rate"] = (
        result["final_rescue_recovered"] / max(result["rescue_pixels"], 1)
    )
    result["proposal_baseline_correct_damage_rate"] = (
        result["proposal_baseline_correct_damaged"] / max(result["baseline_correct_pixels"], 1)
    )
    result["final_baseline_correct_damage_rate"] = (
        result["final_baseline_correct_damaged"] / max(result["baseline_correct_pixels"], 1)
    )
    result["acceptance_mean_on_beneficial"] = (
        result["acceptance_beneficial_sum"] / max(result["acceptance_beneficial_pixels"], 1)
    )
    result["acceptance_mean_on_harmful"] = (
        result["acceptance_harmful_sum"] / max(result["acceptance_harmful_pixels"], 1)
    )
    result["protection_loss_weight"] = PROTECTION_LOSS_WEIGHT
    result["acceptance_loss_weight"] = ACCEPTANCE_LOSS_WEIGHT
    result["rescue_supervision"] = False
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
    diag = _new_totals()
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
            final_logits, proposal_logits, correction_row = decode_postwriteback_reliability(
                model, corrector, observation, error_row,
                e1["transportability_low"], memory_row["memory_reliability"],
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
            _, acceptance_info = acceptance_bce_loss(
                correction_row["acceptance_logit_aux"],
                observation["host_logits"],
                proposal_logits,
                gt_gpu,
            )
            _accumulate_diagnostics(
                diag,
                observation,
                c_v4_logits,
                proposal_logits,
                final_logits,
                correction_row,
                rescue,
                gt_gpu,
                acceptance_info,
            )

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
    diag = _finish_totals(diag)
    diag.update({
        "temporal_base": "frozen C-V4 Stateful Semantic Hysteresis",
        "history_length": corrector.history_length,
        "history_source": "raw detached frozen C-V3 logits",
        "semantic_proposal_source": "concat validity-gated e1..e4 only",
        "semantic_proposal_target": "current Host c4 2048D feature",
        "proposal_bound": corrector.residual_scale,
        "acceptance_shape": "single-channel pixel-wise",
        "acceptance_range": "[0,1]",
        "acceptance_position": "after bounded writeback",
        "final_composition": "Decoder(c4 + acceptance * bounded semantic Delta-c4)",
        "acceptance_aux_supervision": "beneficial/harmful proposal BCE; GT training/diagnostic only",
        "controller_output_feedback": False,
        "corrected_output_history_feedback": False,
        "rescue_supervision": False,
        "raft_training": False,
        "raft_metric_only": True,
    })
    return metrics, diag
