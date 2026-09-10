"""C-V16-A training/evaluation: only the fresh Gate receives L_seg."""

import torch
from torch.nn import functional as F

from predify2021.mce_scores import (
    c_v15_proposal_conditioned_soft_acceptance_training as _cv15,
)
from predify2021.mce_scores import (
    c_v14_post_writeback_reliability_training as _base,
)


HISTORY_LENGTH = _cv15.HISTORY_LENGTH
SEMANTIC_CHANNELS = _cv15.SEMANTIC_CHANNELS
C4_CHANNELS = _cv15.C4_CHANNELS
RESIDUAL_SCALE = _cv15.RESIDUAL_SCALE
NUM_CLASSES = _cv15.NUM_CLASSES
IGNORE_LABEL = _cv15.IGNORE_LABEL
FULL9 = _cv15.FULL9


def _zero_supervision_contract():
    return {
        "training_objective": "L_seg only",
        "l_seg": True,
        "l_accept_backprop": False,
        "l_protect_backprop": False,
        "gt_beneficial_harmful_backprop": False,
        "rescue_ce": False,
        "temporal_loss": False,
        "raft_training": False,
    }


def _gradient_contract(corrector):
    per_module = {}
    all_finite = True
    all_abs_sum = 0.0
    for name in corrector.gate_module_names():
        values = []
        for parameter in getattr(corrector, name).parameters():
            if parameter.grad is not None:
                values.append(float(parameter.grad.abs().sum().item()))
                all_finite = all_finite and bool(torch.isfinite(parameter.grad).all())
        per_module[name] = {
            "gradient_present": bool(values),
            "gradient_finite": all_finite,
            "gradient_abs_sum": sum(values),
            "gradient_non_zero": sum(values) > 0.0,
        }
        all_abs_sum += sum(values)

    proposal_has_gradient = any(
        parameter.grad is not None
        for name in corrector.proposal_module_names()
        for parameter in getattr(corrector, name).parameters()
    )
    return {
        "gate_gradient_finite": all_finite,
        "gate_gradient_non_zero": all_abs_sum > 0.0,
        "gate_gradient_abs_sum": all_abs_sum,
        "gate_gradient_by_module": per_module,
        "proposal_gradient_present": proposal_has_gradient,
        "proposal_parameters_receive_gradient": False,
    }


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
    initial = _base._initial_state(model, observer, residual, samples)
    if initial is None:
        return None
    _, previous, pending_motion, motion_hidden = initial
    previous_low = previous["host_low"].detach()
    previous_c1 = previous["c1"].detach()
    raw_history = [previous["host_logits"].detach()]
    motion_history = []
    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = controller_hidden = dynamics_state = None

    optimizer.zero_grad(set_to_none=True)
    frames_in_window = 0
    gradient_contract = None
    totals = {
        "frames": 0,
        "optimizer_steps": 0,
        "l_seg": 0.0,
        "l_accept": 0.0,
        "l_protect": 0.0,
        "gt_beneficial_harmful": 0.0,
        "delta_c4_abs": 0.0,
        "reliability_mean": 0.0,
        "reliability_std": 0.0,
    }

    for frame_index in range(2, len(samples)):
        observation = _base.host_feature_observation(model, samples[frame_index])
        gt_cpu = _base.semantic_mask_from_panoptic_png(
            samples[frame_index]["mask_path"]
        )

        # Proposal generation and all temporal context are frozen.  Keep this
        # block under no_grad; the corrector forward below is the only graph.
        with torch.no_grad():
            prior_low, _ = _base.c_v5.warp_low_logits(previous_low, pending_motion)
            e1 = _base.c_v5._frozen_e1_step(
                correction,
                mask_predictor,
                observation["c1"],
                observation["host_low"],
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
            memory_row, memory_state, _, c_v3_logits = _base.c_v5._frozen_cv3_step(
                refiner,
                observation["c1"],
                observation["host_low"],
                prior_low,
                e1,
                pending_motion,
                memory_state,
                observation["output_size"],
                observation["host_logits"],
            )
            candidate_rows = _base.c_v5._build_history_candidates(
                raw_history,
                motion_history,
                pending_motion,
                corrector.history_length,
            )
            error_row = _base.build_prediction_error_and_temporal_state(
                c_v4_controller,
                dynamics,
                c_v3_logits,
                candidate_rows,
                pending_motion,
                e1["transportability_low"],
                memory_row["memory_reliability"],
                controller_hidden,
                dynamics_state,
                corrector.history_length,
            )
            controller_hidden = error_row["temporal_hidden"]
            dynamics_state = error_row["dynamics_state"]

        final_logits, _, feature_delta_logits, correction_row = (
            _base.decode_post_writeback_feature_update(
                model,
                corrector,
                observation,
                error_row,
                e1["transportability_low"],
                memory_row["memory_reliability"],
            )
        )
        target = gt_cpu.to(final_logits.device, non_blocking=True).long().unsqueeze(0)
        valid = target[0].ne(IGNORE_LABEL)
        pixel_ce = F.cross_entropy(
            final_logits,
            target,
            ignore_index=IGNORE_LABEL,
            reduction="none",
        )[0]
        l_seg = pixel_ce[valid].mean()
        if not torch.isfinite(l_seg):
            raise FloatingPointError("Non-finite C-V16-A L_seg")
        l_seg.backward()
        frames_in_window += 1
        if gradient_contract is None:
            gradient_contract = _gradient_contract(corrector)

        with torch.no_grad():
            totals["frames"] += 1
            totals["l_seg"] += float(l_seg.item())
            totals["delta_c4_abs"] += float(correction_row["delta_c4"].abs().mean().item())
            totals["reliability_mean"] += float(correction_row["reliability"].mean().item())
            totals["reliability_std"] += float(correction_row["reliability"].std().item())
            observed_motion = _base.c_v5._observe_motion(
                observer,
                previous_low,
                previous_c1,
                observation["host_low"],
                observation["c1"],
            )
            motion_error = F.softmax(observation["host_low"], dim=1) - F.softmax(
                prior_low,
                dim=1,
            )
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion,
                motion_error,
                motion_hidden,
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
        controller_hidden = (
            controller_hidden.detach() if controller_hidden is not None else None
        )
        dynamics_state = dynamics_state.detach() if dynamics_state is not None else None

    frames = max(totals["frames"], 1)
    for key in ("l_seg", "delta_c4_abs", "reliability_mean", "reliability_std"):
        totals[key] /= frames
    totals.update(_zero_supervision_contract())
    totals["gradient_contract"] = gradient_contract or _gradient_contract(corrector)
    return totals


def train_epoch(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    c_v4_controller,
    dynamics,
    corrector,
    groups,
    optimizer,
    gradient_accumulation_steps,
):
    model.eval()
    refiner.eval()
    correction.eval()
    mask_predictor.eval()
    c_v4_controller.eval()
    corrector.train()
    rows = []
    for samples in groups.values():
        row = train_sequence(
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
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("No valid C-V16-A training sequences")

    frame_total = max(sum(row["frames"] for row in rows), 1)
    result = {
        "frames": sum(row["frames"] for row in rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "l_seg": sum(row["l_seg"] * row["frames"] for row in rows) / frame_total,
        "l_accept": 0.0,
        "l_protect": 0.0,
        "gt_beneficial_harmful": 0.0,
        "delta_c4_abs": sum(row["delta_c4_abs"] * row["frames"] for row in rows)
        / frame_total,
        "reliability_mean": sum(row["reliability_mean"] * row["frames"] for row in rows)
        / frame_total,
        "reliability_std": sum(row["reliability_std"] * row["frames"] for row in rows)
        / frame_total,
    }
    result.update(_zero_supervision_contract())
    result["gradient_contract"] = next(
        row["gradient_contract"] for row in rows if row["gradient_contract"] is not None
    )
    return result


@torch.inference_mode()
def evaluate(*args, **kwargs):
    """Reuse C-V15's read-only metric protocol with C-V16-A naming."""
    metrics, diagnostics = _cv15.evaluate(*args, **kwargs)
    metrics["c_v16_a"] = metrics.pop("c_v15")
    diagnostics.update(_zero_supervision_contract())
    diagnostics["evaluation_model"] = "C-V16-A fresh Gate"
    return metrics, diagnostics
