"""C-V6S: semantic refinement under a frozen C-V6R-E1 temporal budget.

中文：冻结 C-V6R-E1 时序修正预算下的语义细化。

Structural decision from completed C-V6R (commit aa390a2):
- C-V6R-E1 already reaches mTC=75.8568%, so the temporal branch is frozen.
- The next problem is semantic quality of the correction content, not stronger
  temporal smoothing.

C-V6S therefore keeps the entire C-V6R-E1 temporal path unchanged:
    K=4 temporal attention -> e_t / epsilon_t -> ConvGRU -> lambda_temp.
It trains only SemanticBudgetRefiner:
    P_hist_sem = Normalize(P_hist_temp * exp(Delta_sem))
    lambda_final = lambda_temp * (1 - clamp(d_sem, 0, 1))
    P_out = (1-lambda_final) P_cur + lambda_final P_hist_sem.

Delta_sem and d_sem are exactly zero initialized. Before training, a Dev3
zero-step check requires C-V6S hard predictions to equal frozen C-V6R-E1 at
every evaluated pixel.

Losses are deliberately simple and independently normalized by active pixels:
    L_out     = CE(P_out, GT) on valid GT pixels
    L_sem     = CE(P_hist_sem, GT) on valid GT & history-available pixels
    L_persist = the existing strict C-V6R persistence loss
No teacher KL, no Protection KL, no gate loss, no temporal-attention loss, and
no loss-weight sweep are introduced.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

import predify2021.mce_scores.train_kitti_step_task_space_prior_c_v6_soft_temporal_correction_impl as v6
import predify2021.mce_scores.train_kitti_step_task_space_prior_c_v6r_soft_temporal_correction_impl as v6r
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_budget_refiner import (
    SemanticBudgetRefiner,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)


v5 = v6.v5
SEED = 0
EPOCHS = 3
TBPTT_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
HISTORY_LENGTH = 4
SCORE_HIDDEN_CHANNELS = 16
REFINE_HIDDEN_CHANNELS = 32

DYNAMICS_TAU_E = 4.0
DYNAMICS_K_E = 1.0
DYNAMICS_DT = 1.0

LOSS_OUT_WEIGHT = 1.0
LOSS_SEM_WEIGHT = 1.0
LOSS_PERSIST_WEIGHT = 1.0

CV4_E2_MIOU_REFERENCE = 0.6637739071008685
CV6R_E1_MTC_REFERENCE = 0.7585676063705213
MTC_PRESERVATION_FLOOR = 0.755
ZERO_STEP_SEQUENCES = ("0002", "0010", "0018")

CV6R_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v6r_soft_temporal_correction/best.pt"
)
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v6s_semantic_budget_refinement"
)
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v6s_semantic_budget_refinement"
CANDIDATES = ("host", "e1_base", "c_v3_base", "c_v6r_teacher", "c_v6s")


def _load_frozen_cv6r(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("epoch") != 1:
        raise RuntimeError(
            "C-V6S requires the C-V6R Epoch-1 checkpoint; "
            f"received epoch={payload.get('epoch')}"
        )
    state = payload.get("corrector_state_dict")
    if state is None:
        raise RuntimeError("C-V6R checkpoint is missing corrector_state_dict")
    corrector = v6r.CV6RSoftTemporalCorrector(
        num_classes=v5.NUM_CLASSES,
        history_length=HISTORY_LENGTH,
        attention_hidden_channels=v6.ATTENTION_HIDDEN_CHANNELS,
        correction_hidden_channels=v6.CORRECTION_HIDDEN_CHANNELS,
        gate_init_bias=0.0,
    ).cuda()
    corrector.load_state_dict(state, strict=True)
    corrector.eval()
    corrector.requires_grad_(False)
    return corrector, payload


def _frozen_temporal_step(
    corrector,
    dynamics,
    c_v3_logits,
    candidate_rows,
    pending_motion,
    transportability_low,
    memory_reliability_low,
    correction_hidden,
    dynamics_state,
):
    with torch.no_grad():
        evidence = v6._low_correction_evidence(
            corrector,
            dynamics,
            c_v3_logits,
            candidate_rows,
            pending_motion,
            transportability_low,
            memory_reliability_low,
            correction_hidden,
            dynamics_state,
        )
        temporal_full = v6r._full_resolution_soft_output(
            c_v3_logits,
            candidate_rows,
            evidence["row"],
        )
    return evidence, temporal_full


def _semantic_refinement_output(
    composer,
    c_v3_logits,
    candidate_rows,
    temporal_full,
    transportability_low,
    memory_reliability_low,
):
    low_size = tuple(transportability_low.shape[-2:])
    current_full = F.softmax(c_v3_logits.detach().float(), dim=1)
    current_low = v6._resize_probability(current_full, low_size)
    temporal_history_low = v6._resize_probability(
        temporal_full["history_probability"].detach(),
        low_size,
    )
    history_low, validity_low = v6._pad_history_low(
        current_low,
        candidate_rows,
        HISTORY_LENGTH,
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
    # Equivalent to softmax(log(P_hist_temp)+Delta), but avoids explicit log(0).
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

    output_probability = (
        (1.0 - lambda_final) * current_full
        + lambda_final * refined_history
    )
    output_probability = v6._renormalize_probability(output_probability)
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
        "delta_semantic": delta_full,
        "composer_row": row,
    }


def _masked_nll(probability, gt_cpu, mask):
    gt = gt_cpu.to(probability.device, non_blocking=True)
    active = mask.to(probability.device).bool() & (gt != v5.IGNORE_LABEL)
    if not bool(active.any()):
        return probability.sum() * 0.0, 0
    per_pixel = F.nll_loss(
        probability.clamp_min(1.0e-8).log(),
        gt.unsqueeze(0),
        ignore_index=v5.IGNORE_LABEL,
        reduction="none",
    )[0]
    return per_pixel[active].mean(), int(active.sum().item())


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
    previous_rescue = None
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
            previous_rescue = torch.zeros_like(current_gt, dtype=torch.bool)
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

        temporal_evidence, temporal_full = _frozen_temporal_step(
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

        valid_mask = targets["valid_gt"]
        history_mask = (
            targets["valid_gt"]
            & refined["history_available"][0, 0].detach().cpu()
        )
        out_loss, out_pixels = _masked_nll(
            refined["output_probability"],
            current_gt,
            valid_mask,
        )
        semantic_loss, semantic_pixels = _masked_nll(
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
            targets["rescue"],
            previous_rescue,
            previous_output_correct,
            teacher_full,
        )
        total_loss = (
            LOSS_OUT_WEIGHT * out_loss
            + LOSS_SEM_WEIGHT * semantic_loss
            + LOSS_PERSIST_WEIGHT * persistence_loss
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

        raw_history.insert(0, c_v3_logits.detach())
        raw_history = raw_history[:HISTORY_LENGTH]
        motion_history.insert(0, pending_motion.detach())
        motion_history = motion_history[: HISTORY_LENGTH - 1]

        previous_correction = current_correction.detach()
        previous_rescue = targets["rescue"].detach().cpu()
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


def _train_epoch(
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
    groups,
    optimizer,
    tbptt_steps,
):
    composer.train()
    rows = []
    for samples in groups.values():
        row = _train_sequence(
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
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("No valid C-V6S training sequences")

    frame_total = max(sum(row["supervised_frames"] for row in rows), 1)
    result = {
        "frames": sum(row["frames"] for row in rows),
        "supervised_frames": sum(row["supervised_frames"] for row in rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "out_pixels": sum(row["out_pixels"] for row in rows),
        "semantic_pixels": sum(row["semantic_pixels"] for row in rows),
        "persistence_pixels": sum(row["persistence_pixels"] for row in rows),
        "rescue_frames": sum(row["rescue_frames"] for row in rows),
        "protection_frames": sum(row["protection_frames"] for row in rows),
    }
    for key in (
        "out_loss",
        "semantic_loss",
        "persistence_loss",
        "total_loss",
        "lambda_temp",
        "lambda_final",
        "attenuation",
    ):
        result[key] = sum(
            row[key] * row["supervised_frames"] for row in rows
        ) / frame_total

    rescue_frames = max(sum(row["rescue_frames"] for row in rows), 1)
    protection_frames = max(sum(row["protection_frames"] for row in rows), 1)
    for key in ("lambda_temp_rescue", "lambda_final_rescue"):
        result[key] = sum(
            row[key] * row["rescue_frames"] for row in rows
        ) / rescue_frames
    for key in ("lambda_temp_protection", "lambda_final_protection"):
        result[key] = sum(
            row[key] * row["protection_frames"] for row in rows
        ) / protection_frames
    return result


@torch.inference_mode()
def _evaluate(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    temporal_corrector,
    dynamics,
    composer,
    groups,
    raft,
):
    composer.eval()
    confusion = {
        name: torch.zeros((v5.NUM_CLASSES, v5.NUM_CLASSES), dtype=torch.int64)
        for name in CANDIDATES
    }
    mtc_sum = {name: 0.0 for name in CANDIDATES}
    mtc_count = {name: 0 for name in CANDIDATES}
    vc_sum = {name: {8: 0.0, 16: 0.0} for name in CANDIDATES}
    vc_count = {name: {8: 0, 16: 0} for name in CANDIDATES}
    diagnostics = {
        "decision_frames": 0,
        "c_v6s_vs_c_v6r_disagreement_pixels": 0,
        "c_v6s_vs_c_v3_disagreement_pixels": 0,
        "lambda_temp_sum": 0.0,
        "lambda_final_sum": 0.0,
        "attenuation_sum": 0.0,
        "lambda_temp_rescue_sum": 0.0,
        "lambda_final_rescue_sum": 0.0,
        "lambda_temp_protection_sum": 0.0,
        "lambda_final_protection_sum": 0.0,
        "rescue_frames": 0,
        "protection_frames": 0,
    }

    for sequence, samples in groups.items():
        previous = None
        pending_motion = motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = None
        memory_state = None
        correction_hidden = None
        dynamics_state = None
        raw_history = []
        motion_history = []
        previous_predictions = {}
        seq_vc = {name: v5.VideoConsistency() for name in CANDIDATES}

        for sample in samples:
            image, host_logits, host_low, current_c1, output_size = v5._host_observation(
                model,
                sample,
            )
            host_pred = host_logits.argmax(1)
            gt_cpu = v5.semantic_mask_from_panoptic_png(sample["mask_path"])
            previous_image_for_mtc = previous[0] if previous is not None else None
            teacher_flow = (
                raft.current_to_previous(image, previous_image_for_mtc)
                if previous_image_for_mtc is not None
                else None
            )

            if previous is None:
                e1_pred = c_v3_pred = c_v6r_pred = c_v6s_pred = host_pred
                semantic_state_low = torch.zeros_like(host_low)
                previous = (image, host_low.detach(), current_c1.detach())
            elif pending_motion is None:
                previous_image, previous_low, previous_c1 = previous
                observed = v5._observe_motion(
                    observer,
                    previous_low,
                    previous_c1,
                    host_low,
                    current_c1,
                )
                error = F.softmax(host_low, dim=1) - F.softmax(previous_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(observed, error, None)
                e1_pred = c_v3_pred = c_v6r_pred = c_v6s_pred = host_pred
                previous = (image, host_low.detach(), current_c1.detach())
            else:
                previous_image, previous_low, previous_c1 = previous
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
                memory_row, memory_state, e1_logits, c_v3_logits = v5._frozen_cv3_step(
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
                observed = v5._observe_motion(
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
                e1_pred = e1_logits.argmax(1)
                c_v3_pred = c_v3_logits.argmax(1)

                if not raw_history:
                    c_v6r_pred = c_v6s_pred = c_v3_pred
                    raw_history = [c_v3_logits.detach()]
                else:
                    candidate_rows = v6._build_probability_history_candidates(
                        raw_history,
                        motion_history,
                        pending_motion,
                        HISTORY_LENGTH,
                    )
                    temporal_evidence, temporal_full = _frozen_temporal_step(
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
                    c_v6r_pred = temporal_full["output_probability"].argmax(1)
                    c_v6s_pred = refined["output_probability"].argmax(1)
                    targets = v6._semantic_targets(temporal_full, candidate_rows, gt_cpu)

                    diagnostics["decision_frames"] += 1
                    diagnostics["c_v6s_vs_c_v6r_disagreement_pixels"] += int(
                        (c_v6s_pred != c_v6r_pred).sum().item()
                    )
                    diagnostics["c_v6s_vs_c_v3_disagreement_pixels"] += int(
                        (c_v6s_pred != c_v3_pred).sum().item()
                    )
                    diagnostics["lambda_temp_sum"] += float(
                        refined["lambda_temp"].mean().item()
                    )
                    diagnostics["lambda_final_sum"] += float(
                        refined["lambda_final"].mean().item()
                    )
                    diagnostics["attenuation_sum"] += float(
                        refined["attenuation"].mean().item()
                    )
                    rescue = targets["rescue"].to(refined["lambda_final"].device)
                    protection = targets["protection"].to(refined["lambda_final"].device)
                    if bool(rescue.any()):
                        diagnostics["lambda_temp_rescue_sum"] += float(
                            refined["lambda_temp"][0, 0][rescue].mean().item()
                        )
                        diagnostics["lambda_final_rescue_sum"] += float(
                            refined["lambda_final"][0, 0][rescue].mean().item()
                        )
                        diagnostics["rescue_frames"] += 1
                    if bool(protection.any()):
                        diagnostics["lambda_temp_protection_sum"] += float(
                            refined["lambda_temp"][0, 0][protection].mean().item()
                        )
                        diagnostics["lambda_final_protection_sum"] += float(
                            refined["lambda_final"][0, 0][protection].mean().item()
                        )
                        diagnostics["protection_frames"] += 1

                    raw_history.insert(0, c_v3_logits.detach())
                    raw_history = raw_history[:HISTORY_LENGTH]
                    motion_history.insert(0, pending_motion.detach())
                    motion_history = motion_history[: HISTORY_LENGTH - 1]

                previous = (image, host_low.detach(), current_c1.detach())
                pending_motion = next_motion.detach()
                motion_hidden = next_motion_hidden.detach()

            predictions = {
                "host": host_pred,
                "e1_base": e1_pred,
                "c_v3_base": c_v3_pred,
                "c_v6r_teacher": c_v6r_pred,
                "c_v6s": c_v6s_pred,
            }
            for name, prediction in predictions.items():
                pred_cpu = prediction.squeeze(0).cpu()
                v5.update_confusion_matrix(confusion[name], pred_cpu, gt_cpu)
                seq_vc[name].update(gt_cpu, pred_cpu)

            if teacher_flow is not None:
                for name, prediction in predictions.items():
                    if name in previous_predictions:
                        score = v5._pair_mtc(
                            previous_predictions[name],
                            prediction,
                            teacher_flow,
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
            "mIoU": float(torch.nanmean(v5.compute_iou(confusion[name])).item()),
            "mTC": mtc_sum[name] / mtc_count[name] if mtc_count[name] else float("nan"),
            "mVC8": vc_sum[name][8] / vc_count[name][8] if vc_count[name][8] else float("nan"),
            "mVC16": vc_sum[name][16] / vc_count[name][16] if vc_count[name][16] else float("nan"),
        }
        for name in CANDIDATES
    }
    decisions = max(diagnostics["decision_frames"], 1)
    diagnostics_out = {
        "decision_frames": diagnostics["decision_frames"],
        "c_v6s_vs_c_v6r_disagreement_pixels": diagnostics[
            "c_v6s_vs_c_v6r_disagreement_pixels"
        ],
        "c_v6s_vs_c_v3_disagreement_pixels": diagnostics[
            "c_v6s_vs_c_v3_disagreement_pixels"
        ],
        "lambda_temp": diagnostics["lambda_temp_sum"] / decisions,
        "lambda_final": diagnostics["lambda_final_sum"] / decisions,
        "attenuation": diagnostics["attenuation_sum"] / decisions,
        "lambda_temp_rescue": diagnostics["lambda_temp_rescue_sum"]
        / max(diagnostics["rescue_frames"], 1),
        "lambda_final_rescue": diagnostics["lambda_final_rescue_sum"]
        / max(diagnostics["rescue_frames"], 1),
        "lambda_temp_protection": diagnostics["lambda_temp_protection_sum"]
        / max(diagnostics["protection_frames"], 1),
        "lambda_final_protection": diagnostics["lambda_final_protection_sum"]
        / max(diagnostics["protection_frames"], 1),
        "temporal_branch_frozen": True,
        "dynamics_error_role": "frozen temporal correction-budget evidence only",
        "semantic_refiner_uses_dynamics_error": False,
        "validity_is_hard_semantic_mask": True,
        "controller_output_feedback": False,
    }
    return metrics, diagnostics_out


def _selection_key(metrics):
    candidate = metrics["c_v6s"]
    preserved_mtc = candidate["mTC"] >= MTC_PRESERVATION_FLOOR
    if preserved_mtc:
        return (1, candidate["mIoU"], candidate["mTC"])
    return (0, candidate["mTC"], candidate["mIoU"])


def _delta(candidate, reference):
    return {
        key: candidate[key] - reference[key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=v5.FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=v5.OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=v5.RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--base-checkpoint", default=v5.BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v3-checkpoint", default=v5.C_V3_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v6r-checkpoint", default=CV6R_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--tbptt-steps", type=int, default=TBPTT_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs <= 0 or args.tbptt_steps <= 0:
        raise ValueError("epochs and tbptt-steps must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    v5._validate_bounded_motion_checkpoint(args.residual_checkpoint)
    model = v5.load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = v5._load_frozen_observer(args.observer_checkpoint)
    residual, residual_payload = v5._load_frozen_residual(
        args.residual_checkpoint,
        observer,
    )
    correction, mask_predictor, base_payload = v5._load_frozen_e1_base(
        args.base_checkpoint,
        observer,
    )
    refiner, cv3_payload = v5._load_frozen_cv3_refiner(args.c_v3_checkpoint)
    temporal_corrector, cv6r_payload = _load_frozen_cv6r(args.c_v6r_checkpoint)
    dynamics = EulerDynamicsError(
        tau_e=DYNAMICS_TAU_E,
        k_e=DYNAMICS_K_E,
        dt=DYNAMICS_DT,
    )
    composer = SemanticBudgetRefiner(
        num_classes=v5.NUM_CLASSES,
        history_length=HISTORY_LENGTH,
        score_hidden_channels=SCORE_HIDDEN_CHANNELS,
        refine_hidden_channels=REFINE_HIDDEN_CHANNELS,
    ).cuda()

    for name, module in (
        ("Host", model),
        ("Motion Observer", observer),
        ("Motion Residual", residual),
        ("E1 correction", correction),
        ("E1 transportability mask", mask_predictor),
        ("C-V3 refiner", refiner),
        ("C-V6R temporal corrector", temporal_corrector),
    ):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError(f"{name} must remain frozen in C-V6S")
    if not any(parameter.requires_grad for parameter in composer.parameters()):
        raise RuntimeError("SemanticBudgetRefiner must be trainable")

    raft = v5.FrozenRAFT()
    train = v5.KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = v5.KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = v5.sequence_groups(train)
    all_val_groups = v5.sequence_groups(val)
    missing = [sequence for sequence in v5.FULL9 if sequence not in all_val_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {sequence: all_val_groups[sequence] for sequence in v5.FULL9}

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    zero_groups = {
        sequence: val_groups[sequence]
        for sequence in ZERO_STEP_SEQUENCES
        if sequence in val_groups
    }
    zero_metrics, zero_diag = _evaluate(
        model,
        observer,
        residual,
        correction,
        mask_predictor,
        refiner,
        temporal_corrector,
        dynamics,
        composer,
        zero_groups,
        raft,
    )
    zero_step = {
        "sequences": list(zero_groups),
        "c_v6s_vs_c_v6r_disagreement_pixels": zero_diag[
            "c_v6s_vs_c_v6r_disagreement_pixels"
        ],
        "passed": zero_diag["c_v6s_vs_c_v6r_disagreement_pixels"] == 0,
        "c_v6r_teacher": zero_metrics["c_v6r_teacher"],
        "c_v6s": zero_metrics["c_v6s"],
        "identity_definition": (
            "Delta_sem=0 and attenuation=0 => P_hist_sem=P_hist_temp and "
            "lambda_final=lambda_temp"
        ),
    }
    with (result_output / "zero_step_check.json").open("w") as handle:
        json.dump(zero_step, handle, indent=2)
    if not zero_step["passed"]:
        raise RuntimeError(
            "C-V6S zero-step check failed: untrained C-V6S differs from C-V6R-E1"
        )

    optimizer = torch.optim.AdamW(
        composer.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
            temporal_corrector,
            dynamics,
            composer,
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
            temporal_corrector,
            dynamics,
            composer,
            val_groups,
            raft,
        )
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_c_v6r_e1": _delta(metrics["c_v6s"], metrics["c_v6r_teacher"]),
            "delta_vs_c_v3": _delta(metrics["c_v6s"], metrics["c_v3_base"]),
            "delta_vs_host": _delta(metrics["c_v6s"], metrics["host"]),
            "selection_key": list(_selection_key(metrics)),
        }
        history.append(row)
        with (result_output / f"epoch_{epoch:03d}.json").open("w") as handle:
            json.dump(row, handle, indent=2)

        if best is None or tuple(row["selection_key"]) > tuple(best["selection_key"]):
            best = row
            torch.save(
                {
                    "experiment": "c_v6s_semantic_budget_refinement",
                    "epoch": epoch,
                    "composer_state_dict": composer.state_dict(),
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "train": train_stats,
                    "frozen_c_v6r_checkpoint": args.c_v6r_checkpoint,
                    "frozen_c_v6r_epoch": cv6r_payload.get("epoch"),
                    "architecture": {
                        "history_length": HISTORY_LENGTH,
                        "temporal_branch_frozen": True,
                        "semantic_refinement": (
                            "Normalize(P_hist_temp * exp(Delta_sem))"
                        ),
                        "budget": (
                            "lambda_final=lambda_temp*(1-clamp(d_sem,0,1))"
                        ),
                        "semantic_refiner_uses_dynamics_error": False,
                        "validity_is_hard_mask": True,
                        "age_is_soft_evidence": True,
                        "output_feedback": False,
                    },
                    "loss_weights": {
                        "output_ce": LOSS_OUT_WEIGHT,
                        "semantic_ce": LOSS_SEM_WEIGHT,
                        "persistence": LOSS_PERSIST_WEIGHT,
                    },
                },
                output / "best.pt",
            )
        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V6S produced no epoch result")

    candidate = best["metrics"]["c_v6s"]
    summary = {
        "experiment": "C-V6S Semantic Refinement under Frozen C-V6R-E1 Temporal Budget",
        "source_result_commit": "aa390a2",
        "zero_step_check": zero_step,
        "best": best,
        "history": history,
        "frozen_checkpoints": {
            "fast_b": args.fast_b_checkpoint,
            "observer": args.observer_checkpoint,
            "residual": args.residual_checkpoint,
            "e1_base": args.base_checkpoint,
            "c_v3_base": args.c_v3_checkpoint,
            "c_v6r_e1": args.c_v6r_checkpoint,
            "c_v3_checkpoint_epoch": cv3_payload.get("epoch"),
            "c_v6r_checkpoint_epoch": cv6r_payload.get("epoch"),
            "residual_experiment": residual_payload.get("experiment"),
            "e1_experiment": base_payload.get("experiment"),
        },
        "selection_rule": {
            "temporal_hard_constraint": f"mTC >= {MTC_PRESERVATION_FLOOR}",
            "objective_after_constraint": "maximize mIoU, then mTC",
            "semantic_reference": CV4_E2_MIOU_REFERENCE,
            "c_v6r_e1_mtc_reference": CV6R_E1_MTC_REFERENCE,
        },
        "targets": {
            "mtc_preserved": candidate["mTC"] >= MTC_PRESERVATION_FLOOR,
            "miou_reaches_cv4_e2": candidate["mIoU"] >= CV4_E2_MIOU_REFERENCE,
        },
        "architecture": {
            "temporal_branch": "frozen C-V6R-E1",
            "dynamics_error_role": "frozen temporal correction-budget evidence only",
            "semantic_refinement_anchor": "C-V6R P_hist_temp",
            "class_wise_history_evidence": True,
            "budget_only_attenuates": True,
            "validity_is_hard_mask": True,
            "age_is_soft_evidence": True,
            "recursive_semantic_state": False,
            "controller_output_feedback": False,
        },
        "losses": {
            "output": "active-pixel mean CE on valid GT",
            "semantic": "active-pixel mean CE on valid GT & history-available",
            "persistence": "unchanged strict C-V6R persistence",
            "teacher_kl": False,
            "protection_kl": False,
            "weights": [1.0, 1.0, 1.0],
        },
        "epochs": args.epochs,
        "tbptt_steps": args.tbptt_steps,
    }
    with (result_output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
