"""C-V6S review fixes.

中文：C-V6S 代码审查修正版。

This module keeps the C-V6S architecture unchanged and fixes four correctness
issues found before the first 3-Epoch run:
1. strict probability-level C-V6S(E0) == frozen C-V6R(E1) checking;
2. hard low-resolution validity masking for P_hist_temp before semantic refine;
3. Persistence opportunity = CurrentWrong & HistoryAvailable, so multi-frame
   semantic rescue is not restricted to AnyIndividualHistoryCorrect;
4. refresh the current Persistence teacher after optimizer.step() at every
   TBPTT boundary, so the next window never consumes a stale pre-update teacher.
"""

import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

import predify2021.mce_scores.train_kitti_step_task_space_prior_c_v6s_semantic_budget_refinement_impl as base


v5 = base.v5
v6 = base.v6
HISTORY_LENGTH = base.HISTORY_LENGTH

PROBABILITY_IDENTITY_TOL = 1.0e-6
LAMBDA_IDENTITY_TOL = 1.0e-7

_BASE_EVALUATE = base._evaluate
_ZERO_STEP_IDENTITY = None


def _composer_is_exact_identity(composer):
    tensors = (
        composer.delta_head.weight,
        composer.delta_head.bias,
        composer.attenuation_head.weight,
        composer.attenuation_head.bias,
    )
    return all(bool(torch.count_nonzero(t.detach()).item() == 0) for t in tensors)


def _semantic_refinement_output(
    composer,
    c_v3_logits,
    candidate_rows,
    temporal_full,
    transportability_low,
    memory_reliability_low,
):
    """C-V6S semantic refinement with strict low-resolution validity masking."""
    global _ZERO_STEP_IDENTITY

    low_size = tuple(transportability_low.shape[-2:])
    current_full = F.softmax(c_v3_logits.detach().float(), dim=1)
    current_low = v6._resize_probability(current_full, low_size)

    history_low, validity_low = v6._pad_history_low(
        current_low,
        candidate_rows,
        HISTORY_LENGTH,
    )
    history_union_low = (
        torch.cat(validity_low, dim=1).sum(dim=1, keepdim=True) > 0.5
    ).to(current_low.dtype)

    temporal_history_low = v6._resize_probability(
        temporal_full["history_probability"].detach(),
        low_size,
    )
    # Validity is a hard constraint, not only an input feature. Re-mask after
    # bilinear resize so invalid boundary pixels cannot leak temporal semantics.
    temporal_history_low = v6._renormalize_probability(
        temporal_history_low,
        history_union_low,
    )

    row = composer(
        current_low,
        temporal_history_low,
        [item.detach() for item in history_low],
        [item.detach() for item in validity_low],
        transportability_low.detach().clamp(0.0, 1.0),
        memory_reliability_low.detach().clamp(0.0, 1.0),
    )

    full_size = tuple(c_v3_logits.shape[-2:])
    delta_full = F.interpolate(
        row["delta_semantic"],
        size=full_size,
        mode="bilinear",
        align_corners=False,
    )
    attenuation_raw_full = F.interpolate(
        row["attenuation_raw"],
        size=full_size,
        mode="bilinear",
        align_corners=False,
    )

    temporal_history = temporal_full["history_probability"].detach().float()
    refined_unnormalized = temporal_history * torch.exp(
        delta_full.clamp(min=-12.0, max=12.0)
    )
    history_available = temporal_full["history_validities"].any(
        dim=1,
        keepdim=True,
    )
    refined_history = v6._renormalize_probability(
        refined_unnormalized,
        history_available,
    )

    attenuation = attenuation_raw_full.clamp(0.0, 1.0)
    budget_utilization = 1.0 - attenuation
    lambda_temp = temporal_full["lambda_probability"].detach()
    lambda_final = lambda_temp * budget_utilization

    if bool((lambda_final > lambda_temp + 1.0e-8).any()):
        raise RuntimeError("C-V6S violated lambda_final <= lambda_temp")

    output_probability = (
        (1.0 - lambda_final) * current_full
        + lambda_final * refined_history
    )
    output_probability = v6._renormalize_probability(output_probability)

    if _composer_is_exact_identity(composer):
        history_diff = float(
            (refined_history - temporal_history).abs().max().detach().item()
        )
        lambda_diff = float(
            (lambda_final - lambda_temp).abs().max().detach().item()
        )
        output_diff = float(
            (
                output_probability
                - temporal_full["output_probability"].detach().float()
            ).abs().max().detach().item()
        )
        delta_abs = float(delta_full.abs().max().detach().item())
        attenuation_abs = float(attenuation.abs().max().detach().item())
        row_diag = {
            "history_probability_max_abs_diff": history_diff,
            "lambda_max_abs_diff": lambda_diff,
            "output_probability_max_abs_diff": output_diff,
            "delta_semantic_max_abs": delta_abs,
            "attenuation_max_abs": attenuation_abs,
        }
        if _ZERO_STEP_IDENTITY is None:
            _ZERO_STEP_IDENTITY = dict(row_diag)
        else:
            for key, value in row_diag.items():
                _ZERO_STEP_IDENTITY[key] = max(_ZERO_STEP_IDENTITY[key], value)

        if history_diff > PROBABILITY_IDENTITY_TOL:
            raise RuntimeError(
                "C-V6S zero-step failed: refined history differs from C-V6R history"
            )
        if output_diff > PROBABILITY_IDENTITY_TOL:
            raise RuntimeError(
                "C-V6S zero-step failed: output probability differs from C-V6R-E1"
            )
        if lambda_diff > LAMBDA_IDENTITY_TOL:
            raise RuntimeError(
                "C-V6S zero-step failed: lambda_final differs from lambda_temp"
            )
        if delta_abs != 0.0 or attenuation_abs != 0.0:
            raise RuntimeError(
                "C-V6S zero-step failed: semantic refinement heads are not exact zero"
            )

    return {
        "current_probability": current_full,
        "temporal_history_probability": temporal_history,
        "refined_history_probability": refined_history,
        "output_probability": output_probability,
        "lambda_temp": lambda_temp,
        "lambda_final": lambda_final,
        "attenuation": attenuation,
        "budget_utilization": budget_utilization,
        "history_available": history_available,
        "history_union_low": history_union_low,
        "delta_semantic": delta_full,
        "composer_row": row,
    }


def _train_sequence(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    temporal_corrector,
    dynamics,
    composer,
    raft,
    samples,
    optimizer,
    tbptt_steps,
):
    if len(samples) < 4:
        return None

    frame0 = v5._host_observation(model, samples[0])
    frame1 = v5._host_observation(model, samples[1])
    pending_motion, motion_hidden = v5._initialize_motion(
        observer,
        residual,
        frame0,
        frame1,
    )
    previous_image, _, previous_low, previous_c1, _ = frame1

    raw_history = []
    motion_history = []
    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = None
    correction_hidden = None
    dynamics_state = None

    previous_correction = None
    previous_persistence_opportunity = None
    previous_output_correct = None
    previous_gt = None

    buffered_losses = []
    frames_in_window = 0
    totals = {
        "frames": 0,
        "supervised_frames": 0,
        "optimizer_steps": 0,
        "out_loss": 0.0,
        "semantic_loss": 0.0,
        "persistence_loss": 0.0,
        "total_loss": 0.0,
        "out_pixels": 0,
        "semantic_pixels": 0,
        "persistence_pixels": 0,
        "lambda_temp": 0.0,
        "lambda_final": 0.0,
        "attenuation": 0.0,
        "lambda_temp_rescue": 0.0,
        "lambda_final_rescue": 0.0,
        "lambda_temp_protection": 0.0,
        "lambda_final_protection": 0.0,
        "rescue_frames": 0,
        "protection_frames": 0,
    }

    for frame_index in range(2, len(samples)):
        current_image, host_logits, host_low, current_c1, output_size = v5._host_observation(
            model,
            samples[frame_index],
        )
        current_gt = v5.semantic_mask_from_panoptic_png(samples[frame_index]["mask_path"])

        with torch.no_grad():
            prior_low, _ = v5.warp_low_logits(previous_low, pending_motion)
            e1 = v5._frozen_e1_step(
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
            memory_row, memory_state, _, c_v3_logits = v5._frozen_cv3_step(
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
            observed_motion = v5._observe_motion(
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

        if not raw_history:
            raw_history = [c_v3_logits.detach()]
            previous_correction = torch.zeros_like(
                F.softmax(c_v3_logits.detach().float(), dim=1)
            )
            previous_persistence_opportunity = torch.zeros_like(
                current_gt,
                dtype=torch.bool,
            )
            previous_output_correct = (
                c_v3_logits.argmax(1)[0].detach().cpu() == current_gt
            ) & (current_gt != v5.IGNORE_LABEL)
            previous_gt = current_gt
            previous_image = current_image
            previous_low = host_low.detach()
            previous_c1 = current_c1.detach()
            pending_motion = next_motion.detach()
            motion_hidden = next_motion_hidden.detach()
            totals["frames"] += 1
            continue

        with torch.no_grad():
            candidate_rows = v6._build_probability_history_candidates(
                raw_history,
                motion_history,
                pending_motion,
                HISTORY_LENGTH,
            )

        temporal_evidence, temporal_full = base._frozen_temporal_step(
            temporal_corrector,
            dynamics,
            c_v3_logits,
            candidate_rows,
            pending_motion,
            e1["transportability_low"],
            memory_row["memory_reliability"],
            correction_hidden,
            dynamics_state,
        )
        correction_hidden = temporal_evidence["row"]["correction_hidden"].detach()
        dynamics_state = temporal_evidence["dynamics_state"].detach()

        refined = _semantic_refinement_output(
            composer,
            c_v3_logits,
            candidate_rows,
            temporal_full,
            e1["transportability_low"],
            memory_row["memory_reliability"],
        )

        with torch.no_grad():
            targets = v6._semantic_targets(temporal_full, candidate_rows, current_gt)
            teacher_full = raft.current_to_previous(current_image, previous_image)
            persistence_opportunity = (
                targets["valid_gt"]
                & ~targets["current_correct"]
                & refined["history_available"][0, 0].detach().bool()
            )

        valid_mask = targets["valid_gt"]
        history_mask = (
            targets["valid_gt"]
            & refined["history_available"][0, 0].detach().bool()
        )
        out_loss, out_pixels = base._masked_nll(
            refined["output_probability"],
            current_gt,
            valid_mask,
        )
        semantic_loss, semantic_pixels = base._masked_nll(
            refined["refined_history_probability"],
            current_gt,
            history_mask,
        )

        current_correction = (
            refined["output_probability"] - refined["current_probability"]
        )
        persistence_loss, persistence_pixels = v6._strict_persistence_loss(
            current_correction,
            previous_correction,
            current_gt,
            previous_gt,
            persistence_opportunity,
            previous_persistence_opportunity,
            previous_output_correct,
            teacher_full,
        )
        total_loss = (
            base.LOSS_OUT_WEIGHT * out_loss
            + base.LOSS_SEM_WEIGHT * semantic_loss
            + base.LOSS_PERSIST_WEIGHT * persistence_loss
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite C-V6S training loss")
        buffered_losses.append(total_loss)

        with torch.no_grad():
            current_output_correct = (
                refined["output_probability"].argmax(1)[0] == targets["gt"]
            ) & targets["valid_gt"]
            rescue = targets["rescue"].to(refined["lambda_final"].device)
            protection = targets["protection"].to(refined["lambda_final"].device)
            totals["frames"] += 1
            totals["supervised_frames"] += 1
            totals["out_loss"] += float(out_loss.item())
            totals["semantic_loss"] += float(semantic_loss.item())
            totals["persistence_loss"] += float(persistence_loss.item())
            totals["total_loss"] += float(total_loss.item())
            totals["out_pixels"] += out_pixels
            totals["semantic_pixels"] += semantic_pixels
            totals["persistence_pixels"] += persistence_pixels
            totals["lambda_temp"] += float(refined["lambda_temp"].mean().item())
            totals["lambda_final"] += float(refined["lambda_final"].mean().item())
            totals["attenuation"] += float(refined["attenuation"].mean().item())
            if bool(rescue.any()):
                totals["lambda_temp_rescue"] += float(
                    refined["lambda_temp"][0, 0][rescue].mean().item()
                )
                totals["lambda_final_rescue"] += float(
                    refined["lambda_final"][0, 0][rescue].mean().item()
                )
                totals["rescue_frames"] += 1
            if bool(protection.any()):
                totals["lambda_temp_protection"] += float(
                    refined["lambda_temp"][0, 0][protection].mean().item()
                )
                totals["lambda_final_protection"] += float(
                    refined["lambda_final"][0, 0][protection].mean().item()
                )
                totals["protection_frames"] += 1

        frames_in_window += 1
        boundary = frames_in_window >= tbptt_steps or frame_index == len(samples) - 1
        if boundary:
            window_loss = torch.stack(buffered_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            window_loss.backward()
            optimizer.step()
            totals["optimizer_steps"] += 1
            buffered_losses = []
            frames_in_window = 0

            # The SemanticBudgetRefiner has changed. Refresh exactly the current
            # frame output before it becomes the next frame's Persistence teacher.
            if frame_index != len(samples) - 1:
                with torch.no_grad():
                    refreshed = _semantic_refinement_output(
                        composer,
                        c_v3_logits,
                        candidate_rows,
                        temporal_full,
                        e1["transportability_low"],
                        memory_row["memory_reliability"],
                    )
                    current_correction = (
                        refreshed["output_probability"]
                        - refreshed["current_probability"]
                    ).detach()
                    current_output_correct = (
                        refreshed["output_probability"].argmax(1)[0]
                        == targets["gt"]
                    ) & targets["valid_gt"]

        raw_history.insert(0, c_v3_logits.detach())
        raw_history = raw_history[:HISTORY_LENGTH]
        motion_history.insert(0, pending_motion.detach())
        motion_history = motion_history[: HISTORY_LENGTH - 1]

        previous_correction = current_correction.detach()
        previous_persistence_opportunity = persistence_opportunity.detach().cpu()
        previous_output_correct = current_output_correct.detach().cpu()
        previous_gt = current_gt
        previous_image = current_image
        previous_low = host_low.detach()
        previous_c1 = current_c1.detach()
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()

    supervised = max(totals["supervised_frames"], 1)
    for key in (
        "out_loss",
        "semantic_loss",
        "persistence_loss",
        "total_loss",
        "lambda_temp",
        "lambda_final",
        "attenuation",
    ):
        totals[key] /= supervised
    totals["lambda_temp_rescue"] /= max(totals["rescue_frames"], 1)
    totals["lambda_final_rescue"] /= max(totals["rescue_frames"], 1)
    totals["lambda_temp_protection"] /= max(totals["protection_frames"], 1)
    totals["lambda_final_protection"] /= max(totals["protection_frames"], 1)
    return totals


def _evaluate(*args, **kwargs):
    """Wrap base evaluation and attach probability-level identity diagnostics."""
    global _ZERO_STEP_IDENTITY
    composer = kwargs.get("composer")
    if composer is None and len(args) >= 9:
        composer = args[8]
    identity_mode = composer is not None and _composer_is_exact_identity(composer)
    if identity_mode:
        _ZERO_STEP_IDENTITY = None

    metrics, diagnostics = _BASE_EVALUATE(*args, **kwargs)
    if identity_mode:
        identity = _ZERO_STEP_IDENTITY or {
            "history_probability_max_abs_diff": float("inf"),
            "lambda_max_abs_diff": float("inf"),
            "output_probability_max_abs_diff": float("inf"),
            "delta_semantic_max_abs": float("inf"),
            "attenuation_max_abs": float("inf"),
        }
        identity["passed_probability_identity"] = bool(
            identity["history_probability_max_abs_diff"] <= PROBABILITY_IDENTITY_TOL
            and identity["output_probability_max_abs_diff"] <= PROBABILITY_IDENTITY_TOL
            and identity["lambda_max_abs_diff"] <= LAMBDA_IDENTITY_TOL
            and identity["delta_semantic_max_abs"] == 0.0
            and identity["attenuation_max_abs"] == 0.0
        )
        diagnostics["zero_step_probability_identity"] = identity
        if not identity["passed_probability_identity"]:
            # Base main already refuses to train when hard disagreement is nonzero.
            diagnostics["c_v6s_vs_c_v6r_disagreement_pixels"] += 1
    return metrics, diagnostics


def _arg_value(argv, name, default):
    args = list(sys.argv[1:] if argv is None else argv)
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return default


def _rewrite_metadata(argv):
    result_dir = Path(_arg_value(argv, "--result-output", base.RESULT_DEFAULT))
    zero_path = result_dir / "zero_step_check.json"
    if zero_path.exists() and _ZERO_STEP_IDENTITY is not None:
        with zero_path.open() as handle:
            payload = json.load(handle)
        identity = dict(_ZERO_STEP_IDENTITY)
        identity["probability_tolerance"] = PROBABILITY_IDENTITY_TOL
        identity["lambda_tolerance"] = LAMBDA_IDENTITY_TOL
        identity["passed_probability_identity"] = bool(
            identity["history_probability_max_abs_diff"] <= PROBABILITY_IDENTITY_TOL
            and identity["output_probability_max_abs_diff"] <= PROBABILITY_IDENTITY_TOL
            and identity["lambda_max_abs_diff"] <= LAMBDA_IDENTITY_TOL
            and identity["delta_semantic_max_abs"] == 0.0
            and identity["attenuation_max_abs"] == 0.0
        )
        payload["probability_identity"] = identity
        with zero_path.open("w") as handle:
            json.dump(payload, handle, indent=2)

    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open() as handle:
            summary = json.load(handle)
        summary["implementation"] = "c_v6s_reviewfix_probability_identity_persistence_refresh"
        summary["architecture"]["low_resolution_temporal_history_validity"] = (
            "hard union mask reapplied after bilinear resize"
        )
        summary["losses"]["persistence"] = (
            "strict successful-correction teacher; current/previous opportunity = "
            "CurrentWrong & HistoryAvailable, independent of individual-history argmax correctness"
        )
        summary["training_correctness"] = {
            "tbptt_boundary_teacher_refresh": True,
            "probability_level_zero_step_check": True,
            "lambda_final_never_exceeds_lambda_temp": True,
        }
        with summary_path.open("w") as handle:
            json.dump(summary, handle, indent=2)


def _patch_base():
    base._semantic_refinement_output = _semantic_refinement_output
    base._train_sequence = _train_sequence
    base._evaluate = _evaluate


def main(argv=None):
    _patch_base()
    result = base.main(argv)
    _rewrite_metadata(argv)
    return result


if __name__ == "__main__":
    main()
