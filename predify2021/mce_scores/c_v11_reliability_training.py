"""Training helpers for C-V11 Reliability-Conditioned Expansion.

中文：C-V11 可靠性条件幅度扩张训练辅助模块。
"""

import torch
from torch.nn import functional as F

from predify2021.mce_scores import (
    diagnose_c_v7_semantic_decodability_probes as diagnostic,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v7_error_residual_correction as c_v7,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v9_proposal_supervision as c_v9,
)

BASE_GAIN = 0.25
PROPOSAL_LOSS_WEIGHT = c_v9.PROPOSAL_LOSS_WEIGHT
RELIABILITY_LOSS_WEIGHT = 1.0


class ControlStats:
    """Accumulate detached control means without per-frame GPU synchronization."""

    def __init__(self, corrector):
        self.corrector = corrector
        self.handle = None
        self.count = 0
        self.sums = {
            "acceptance_mean": None,
            "expansion_reliability_mean": None,
            "base_alpha_mean": None,
            "expansion_mean": None,
            "alpha_mean": None,
        }

    def _hook(self, _module, _inputs, output):
        if not isinstance(output, dict):
            return
        values = {
            "acceptance_mean": output.get("acceptance"),
            "expansion_reliability_mean": output.get("expansion_reliability"),
            "base_alpha_mean": output.get("base_alpha"),
            "expansion_mean": output.get("expansion"),
            "alpha_mean": output.get("alpha"),
        }
        if any(value is None for value in values.values()):
            return
        for key, value in values.items():
            scalar = value.detach().mean()
            self.sums[key] = scalar if self.sums[key] is None else self.sums[key] + scalar
        self.count += 1

    def __enter__(self):
        self.handle = self.corrector.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def result(self):
        if self.count == 0:
            return {"control_frames": 0, **{key: 0.0 for key in self.sums}}
        return {
            "control_frames": self.count,
            **{key: float((value / self.count).item()) for key, value in self.sums.items()},
        }


def compose_window_loss(final_losses, proposal_losses, reliability_losses):
    if not final_losses:
        raise ValueError("final_losses must be non-empty")
    final_mean = torch.stack(final_losses).mean()
    proposal_mean = torch.stack(proposal_losses).mean() if proposal_losses else final_mean * 0.0
    reliability_mean = (
        torch.stack(reliability_losses).mean()
        if reliability_losses
        else final_mean * 0.0
    )
    total = (
        final_mean
        + PROPOSAL_LOSS_WEIGHT * proposal_mean
        + RELIABILITY_LOSS_WEIGHT * reliability_mean
    )
    return total, final_mean, proposal_mean, reliability_mean


def expansion_reliability_supervision(
    c_v3_logits,
    delta_z_low,
    reliability_logit_low,
    any_valid_full,
    current_gt_cpu,
):
    """Train whether correction beyond 0.25 is definitely useful or harmful.

    Positive: current wrong, 0.25*tanh(Proposal) still wrong, full tanh(Proposal) correct.
    Negative: current correct, full tanh(Proposal) wrong.
    Other pixels are ignored. GT never enters model inputs or inference.
    """
    z_cur = c_v3_logits.detach()
    full_size = tuple(z_cur.shape[-2:])
    delta_full = F.interpolate(
        delta_z_low,
        size=full_size,
        mode="bilinear",
        align_corners=False,
    ) * any_valid_full.to(z_cur.dtype)
    z_025 = z_cur + BASE_GAIN * delta_full
    z_100 = z_cur + delta_full

    gt = current_gt_cpu.to(z_cur.device, non_blocking=True).long()
    valid = gt.ne(c_v5.IGNORE_LABEL) & any_valid_full[0, 0].bool()
    current_pred = z_cur.argmax(1)[0]
    pred_025 = z_025.argmax(1)[0]
    pred_100 = z_100.argmax(1)[0]
    positive = valid & current_pred.ne(gt) & pred_025.ne(gt) & pred_100.eq(gt)
    negative = valid & current_pred.eq(gt) & pred_100.ne(gt)

    logit_full = F.interpolate(
        reliability_logit_low,
        size=full_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    n_pos = int(positive.sum().item())
    n_neg = int(negative.sum().item())
    losses = []
    if n_pos:
        losses.append(F.softplus(-logit_full[positive]).mean())
    if n_neg:
        losses.append(F.softplus(logit_full[negative]).mean())
    loss = torch.stack(losses).mean() if losses else logit_full.sum() * 0.0

    with torch.no_grad():
        probability = torch.sigmoid(logit_full)
        pos_sum = float(probability[positive].sum().item()) if n_pos else 0.0
        neg_sum = float(probability[negative].sum().item()) if n_neg else 0.0
    return loss, {
        "positive_pixels": n_pos,
        "negative_pixels": n_neg,
        "positive_probability_sum": pos_sum,
        "negative_probability_sum": neg_sum,
    }


def train_sequence(
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
    pending_motion, motion_hidden = c_v5._initialize_motion(observer, residual, frame0, frame1)
    _, previous_host_logits, previous_low, previous_c1, _ = frame1
    raw_history = [previous_host_logits.detach()]
    motion_history = []
    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = error_hidden = dynamics_state = None
    buffered_final, buffered_proposal, buffered_reliability = [], [], []
    frames_in_window = 0

    totals = {
        "frames": 0,
        "frames_with_rescue": 0,
        "frames_with_reliability_supervision": 0,
        "rescue_pixels": 0,
        "reliability_positive_pixels": 0,
        "reliability_negative_pixels": 0,
        "reliability_positive_probability_sum": 0.0,
        "reliability_negative_probability_sum": 0.0,
        "optimizer_steps": 0,
        "segmentation_ce": 0.0,
        "proposal_rescue_ce_sum": 0.0,
        "reliability_ce_frame_sum": 0.0,
        "total_loss": 0.0,
        "window_final_ce": 0.0,
        "window_proposal_rescue_ce": 0.0,
        "window_reliability_ce": 0.0,
        "prediction_error_abs": 0.0,
        "dynamics_error_abs": 0.0,
        "delta_z_abs": 0.0,
        "alpha_full_mean": 0.0,
        "applied_correction_abs": 0.0,
        "error_reliability_mean": 0.0,
        "sign_agreement_mean": 0.0,
        "valid_fraction_mean": 0.0,
        "proposal_raw_rescue_recovered": 0,
        "proposal_raw_current_correct_pixels": 0,
        "proposal_raw_current_correct_damaged": 0,
    }

    for frame_index in range(2, len(samples)):
        _, host_logits, host_low, current_c1, output_size = c_v5._host_observation(
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
                raw_history, motion_history, pending_motion, corrector.history_length
            )
            masks = diagnostic._build_diagnostic_masks(c_v3_logits, candidate_rows, current_gt)

        evidence = c_v7._correction_evidence(
            corrector, dynamics, c_v3_logits, candidate_rows, pending_motion,
            e1["transportability_low"], memory_row["memory_reliability"],
            error_hidden, dynamics_state,
        )
        row = evidence["row"]
        error_hidden = row["hidden"]
        dynamics_state = evidence["dynamics_state"]
        target = current_gt.to(evidence["final_logits_full"].device, non_blocking=True).long().unsqueeze(0)
        segmentation_ce = F.cross_entropy(
            evidence["final_logits_full"], target, ignore_index=c_v5.IGNORE_LABEL
        )
        proposal_ce, z_raw, rescue_pixels = c_v9._proposal_rescue_ce(
            c_v3_logits, row["delta_z_raw"], evidence["any_valid_full"],
            current_gt, masks["rescue"],
        )

        # Reliability auxiliary gradient stops at the shared Control Pre.
        supervised_logit = corrector.expansion_reliability_head(
            row["control_hidden"].detach()
        )
        reliability_ce, rel_stats = expansion_reliability_supervision(
            c_v3_logits, row["delta_z"], supervised_logit,
            evidence["any_valid_full"], current_gt,
        )
        if not all(torch.isfinite(x) for x in (segmentation_ce, proposal_ce, reliability_ce)):
            raise FloatingPointError("Non-finite C-V11 training loss")

        buffered_final.append(segmentation_ce)
        if rescue_pixels:
            buffered_proposal.append(proposal_ce)
        rel_pixels = rel_stats["positive_pixels"] + rel_stats["negative_pixels"]
        if rel_pixels:
            buffered_reliability.append(reliability_ce)

        with torch.no_grad():
            observed_motion = c_v5._observe_motion(
                observer, previous_low, previous_c1, host_low, current_c1
            )
            motion_error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion, motion_error, motion_hidden
            )
            totals["frames"] += 1
            totals["segmentation_ce"] += float(segmentation_ce.item())
            totals["prediction_error_abs"] += float(evidence["prediction_error"].abs().mean().item())
            totals["dynamics_error_abs"] += float(dynamics_state.abs().mean().item())
            totals["delta_z_abs"] += float(row["delta_z"].abs().mean().item())
            totals["alpha_full_mean"] += float(evidence["gate_full"].mean().item())
            totals["applied_correction_abs"] += float(
                (evidence["gate_full"] * evidence["delta_z_full"]).abs().mean().item()
            )
            totals["error_reliability_mean"] += float(row["error_reliability"].mean().item())
            totals["sign_agreement_mean"] += float(row["sign_agreement"].mean().item())
            totals["valid_fraction_mean"] += float(row["valid_fraction"].mean().item())

            if rescue_pixels:
                totals["frames_with_rescue"] += 1
                totals["rescue_pixels"] += rescue_pixels
                totals["proposal_rescue_ce_sum"] += float(proposal_ce.item()) * rescue_pixels
                gt_gpu = current_gt.to(z_raw.device, non_blocking=True).long()
                raw_pred = z_raw.argmax(1)[0]
                rescue = masks["rescue"].to(raw_pred.device).bool()
                totals["proposal_raw_rescue_recovered"] += int((rescue & raw_pred.eq(gt_gpu)).sum().item())

            current_correct = masks["current_correct"].to(z_raw.device).bool()
            n_correct = int(current_correct.sum().item())
            totals["proposal_raw_current_correct_pixels"] += n_correct
            if n_correct:
                gt_gpu = current_gt.to(z_raw.device, non_blocking=True).long()
                raw_pred = z_raw.argmax(1)[0]
                totals["proposal_raw_current_correct_damaged"] += int(
                    (current_correct & raw_pred.ne(gt_gpu)).sum().item()
                )
            if rel_pixels:
                totals["frames_with_reliability_supervision"] += 1
                totals["reliability_ce_frame_sum"] += float(reliability_ce.item())
                totals["reliability_positive_pixels"] += rel_stats["positive_pixels"]
                totals["reliability_negative_pixels"] += rel_stats["negative_pixels"]
                totals["reliability_positive_probability_sum"] += rel_stats["positive_probability_sum"]
                totals["reliability_negative_probability_sum"] += rel_stats["negative_probability_sum"]

        frames_in_window += 1
        boundary = frames_in_window >= tbptt_steps or frame_index == len(samples) - 1
        if boundary:
            if buffered_final:
                loss, final_mean, proposal_mean, reliability_mean = compose_window_loss(
                    buffered_final, buffered_proposal, buffered_reliability
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                totals["optimizer_steps"] += 1
                totals["total_loss"] += float(loss.detach().item())
                totals["window_final_ce"] += float(final_mean.detach().item())
                totals["window_proposal_rescue_ce"] += float(proposal_mean.detach().item())
                totals["window_reliability_ce"] += float(reliability_mean.detach().item())
            buffered_final, buffered_proposal, buffered_reliability = [], [], []
            frames_in_window = 0
            error_hidden = error_hidden.detach() if error_hidden is not None else None
            memory_state = memory_state.detach() if memory_state is not None else None
            dynamics_state = dynamics_state.detach() if dynamics_state is not None else None

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
        "segmentation_ce", "prediction_error_abs", "dynamics_error_abs",
        "delta_z_abs", "alpha_full_mean", "applied_correction_abs",
        "error_reliability_mean", "sign_agreement_mean", "valid_fraction_mean",
    ):
        totals[key] /= frames
    steps = max(totals["optimizer_steps"], 1)
    for key in ("total_loss", "window_final_ce", "window_proposal_rescue_ce", "window_reliability_ce"):
        totals[key] /= steps

    rescue_den = max(totals["rescue_pixels"], 1)
    totals["proposal_rescue_ce"] = totals.pop("proposal_rescue_ce_sum") / rescue_den
    totals["proposal_raw_rescue_recovery_rate"] = totals["proposal_raw_rescue_recovered"] / rescue_den
    correct_den = max(totals["proposal_raw_current_correct_pixels"], 1)
    totals["proposal_raw_current_correct_damage_rate"] = totals["proposal_raw_current_correct_damaged"] / correct_den
    rel_frames = max(totals["frames_with_reliability_supervision"], 1)
    totals["expansion_reliability_ce"] = totals.pop("reliability_ce_frame_sum") / rel_frames
    pos_den = max(totals["reliability_positive_pixels"], 1)
    neg_den = max(totals["reliability_negative_pixels"], 1)
    totals["expansion_reliability_positive_mean"] = totals.pop("reliability_positive_probability_sum") / pos_den
    totals["expansion_reliability_negative_mean"] = totals.pop("reliability_negative_probability_sum") / neg_den
    return totals


def train_epoch(
    model, observer, residual, correction, mask_predictor, refiner,
    corrector, dynamics, groups, optimizer, tbptt_steps,
):
    corrector.train()
    refiner.eval(); correction.eval(); mask_predictor.eval()
    rows = []
    for samples in groups.values():
        row = train_sequence(
            model, observer, residual, correction, mask_predictor, refiner,
            corrector, dynamics, samples, optimizer, tbptt_steps,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("No valid C-V11 training sequences")

    frame_total = max(sum(r["frames"] for r in rows), 1)
    step_total = max(sum(r["optimizer_steps"] for r in rows), 1)
    rescue_total = max(sum(r["rescue_pixels"] for r in rows), 1)
    correct_total = max(sum(r["proposal_raw_current_correct_pixels"] for r in rows), 1)
    rel_frame_total = max(sum(r["frames_with_reliability_supervision"] for r in rows), 1)
    pos_total = max(sum(r["reliability_positive_pixels"] for r in rows), 1)
    neg_total = max(sum(r["reliability_negative_pixels"] for r in rows), 1)
    result = {
        "frames": sum(r["frames"] for r in rows),
        "frames_with_rescue": sum(r["frames_with_rescue"] for r in rows),
        "frames_with_reliability_supervision": sum(r["frames_with_reliability_supervision"] for r in rows),
        "rescue_pixels": sum(r["rescue_pixels"] for r in rows),
        "reliability_positive_pixels": sum(r["reliability_positive_pixels"] for r in rows),
        "reliability_negative_pixels": sum(r["reliability_negative_pixels"] for r in rows),
        "optimizer_steps": sum(r["optimizer_steps"] for r in rows),
        "proposal_loss_weight": PROPOSAL_LOSS_WEIGHT,
        "reliability_loss_weight": RELIABILITY_LOSS_WEIGHT,
    }
    frame_keys = (
        "segmentation_ce", "prediction_error_abs", "dynamics_error_abs",
        "delta_z_abs", "alpha_full_mean", "applied_correction_abs",
        "error_reliability_mean", "sign_agreement_mean", "valid_fraction_mean",
    )
    for key in frame_keys:
        result[key] = sum(r[key] * r["frames"] for r in rows) / frame_total
    for key in ("total_loss", "window_final_ce", "window_proposal_rescue_ce", "window_reliability_ce"):
        result[key] = sum(r[key] * r["optimizer_steps"] for r in rows) / step_total
    result["proposal_rescue_ce"] = sum(r["proposal_rescue_ce"] * r["rescue_pixels"] for r in rows) / rescue_total
    result["proposal_raw_rescue_recovery_rate"] = sum(r["proposal_raw_rescue_recovered"] for r in rows) / rescue_total
    result["proposal_raw_current_correct_pixels"] = sum(r["proposal_raw_current_correct_pixels"] for r in rows)
    result["proposal_raw_current_correct_damage_rate"] = sum(r["proposal_raw_current_correct_damaged"] for r in rows) / correct_total
    result["expansion_reliability_ce"] = sum(r["expansion_reliability_ce"] * r["frames_with_reliability_supervision"] for r in rows) / rel_frame_total
    result["expansion_reliability_positive_mean"] = sum(r["expansion_reliability_positive_mean"] * r["reliability_positive_pixels"] for r in rows) / pos_total
    result["expansion_reliability_negative_mean"] = sum(r["expansion_reliability_negative_mean"] * r["reliability_negative_pixels"] for r in rows) / neg_total
    return result
