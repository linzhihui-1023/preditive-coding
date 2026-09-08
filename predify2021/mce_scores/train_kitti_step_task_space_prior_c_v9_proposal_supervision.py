"""C-V9: C-V8 architecture with direct Rescue-supervised Proposal learning.

中文：C-V9 保持 C-V8 结构不变，仅增加面向 Rescue（救回）的 Proposal（提议）直接监督。

Evidence-driven change / 基于已有证据的唯一变化
------------------------------------------------
C-V8 frozen decomposition showed that the largest capability gap is before
Gate/tanh: error_76 has strong linear semantic decodability, while the trained
C-V8 raw Proposal recovers only a small fraction of Rescue pixels.  Therefore
this experiment changes Proposal learning, not the inference architecture.

Architecture / 推理结构
----------------------
Exactly the reviewed C-V8 DirectErrorProposalCorrector:
History -> Prediction -> Prediction Error -> 76D->19D Proposal
                         + motion-aligned H_err / Dynamics Error -> Gate
Z_final = Z_cur + g * tanh(DeltaZ_proposal)

Training / 训练
--------------
L = L_final_CE + L_proposal_rescue_CE

L_final_CE is unchanged C-V8 final segmentation CE.
L_proposal_rescue_CE is computed on the *raw*, ungated, unbounded proposal path:
    Z_raw = detach(Z_cur) + valid * upsample(DeltaZ_proposal_raw)
and only on Rescue pixels:
    C-V3 wrong AND at least one valid motion-aligned history predicts GT.

The Rescue definition is exactly the formal semantic-decoding diagnostic mask.
GT is used only for supervised training roles/losses; it is never a model input
and creates no inference-time dependency.  Proposal loss weight is fixed to 1.0
and is intentionally not exposed as a sweep parameter.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    sequence_groups,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
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
    train_kitti_step_task_space_prior_c_v8_direct_error_proposal_gate as c_v8,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_direct_error_proposal_corrector import (
    DirectErrorProposalCorrector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)


EXPERIMENT = "c_v9_proposal_supervision"
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v9_proposal_supervision"
)
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v9_proposal_supervision"
MODEL_KEY = "c_v9"
MIOU_HARD_FLOOR = c_v8.MIOU_HARD_FLOOR
G_MAX = c_v8.G_MAX
GATE_BIAS = c_v8.GATE_BIAS
PROPOSAL_LOSS_WEIGHT = 1.0


def _proposal_rescue_ce(
    c_v3_logits,
    delta_z_raw_low,
    any_valid_full,
    current_gt_cpu,
    rescue_mask,
):
    """Directly supervise the raw Proposal on formal Rescue pixels only.

    Returns (loss, z_raw, rescue_pixels).  No Gate, tanh, H_err or reliability
    tensor enters this function, so this auxiliary loss can update Proposal Head
    but cannot directly train Gate/Error Memory.
    """
    z_cur = c_v3_logits.detach()
    full_size = tuple(z_cur.shape[-2:])
    raw_full = F.interpolate(
        delta_z_raw_low,
        size=full_size,
        mode="bilinear",
        align_corners=False,
    )
    if any_valid_full.ndim != 4 or any_valid_full.shape[1] != 1:
        raise ValueError("any_valid_full must be Bx1xHxW")
    raw_full = raw_full * any_valid_full.to(raw_full.dtype)
    z_raw = z_cur + raw_full

    target = current_gt_cpu.to(z_raw.device, non_blocking=True).long().unsqueeze(0)
    if rescue_mask.ndim != 2 or tuple(rescue_mask.shape) != full_size:
        raise ValueError("rescue_mask must be HxW at full resolution")
    rescue = rescue_mask.to(z_raw.device).bool()
    rescue_pixels = int(rescue.sum().item())

    per_pixel = F.cross_entropy(
        z_raw,
        target,
        ignore_index=c_v5.IGNORE_LABEL,
        reduction="none",
    )[0]
    if rescue_pixels:
        loss = per_pixel[rescue].mean()
    else:
        # Keep a graph-connected exact zero so the caller can combine losses
        # without special gradient handling.
        loss = per_pixel.sum() * 0.0
    return loss, z_raw, rescue_pixels


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
        "frames_with_rescue": 0,
        "rescue_pixels": 0,
        "optimizer_steps": 0,
        "segmentation_ce": 0.0,
        "proposal_rescue_ce_sum": 0.0,
        "total_loss": 0.0,
        "prediction_error_abs": 0.0,
        "dynamics_error_abs": 0.0,
        "delta_z_abs": 0.0,
        "gate_mean": 0.0,
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
            model,
            samples[frame_index],
        )
        current_gt = c_v5.semantic_mask_from_panoptic_png(
            samples[frame_index]["mask_path"]
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
            masks = diagnostic._build_diagnostic_masks(
                c_v3_logits,
                candidate_rows,
                current_gt,
            )

        evidence = c_v7._correction_evidence(
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
        ).long().unsqueeze(0)
        segmentation_ce = F.cross_entropy(
            evidence["final_logits_full"],
            target,
            ignore_index=c_v5.IGNORE_LABEL,
        )
        proposal_rescue_ce, z_raw, rescue_pixels = _proposal_rescue_ce(
            c_v3_logits,
            evidence["row"]["delta_z_raw"],
            evidence["any_valid_full"],
            current_gt,
            masks["rescue"],
        )
        total_loss = segmentation_ce + PROPOSAL_LOSS_WEIGHT * proposal_rescue_ce
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite C-V9 training loss")
        buffered_losses.append(total_loss)

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
            totals["total_loss"] += float(total_loss.detach().item())
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

            if rescue_pixels:
                totals["frames_with_rescue"] += 1
                totals["rescue_pixels"] += rescue_pixels
                totals["proposal_rescue_ce_sum"] += (
                    float(proposal_rescue_ce.detach().item()) * rescue_pixels
                )
                gt_gpu = current_gt.to(z_raw.device, non_blocking=True).long()
                raw_pred = z_raw.argmax(1)[0]
                rescue = masks["rescue"].to(raw_pred.device).bool()
                totals["proposal_raw_rescue_recovered"] += int(
                    (rescue & (raw_pred == gt_gpu)).sum().item()
                )

            current_correct = masks["current_correct"].to(z_raw.device).bool()
            n_correct = int(current_correct.sum().item())
            totals["proposal_raw_current_correct_pixels"] += n_correct
            if n_correct:
                gt_gpu = current_gt.to(z_raw.device, non_blocking=True).long()
                raw_pred = z_raw.argmax(1)[0]
                totals["proposal_raw_current_correct_damaged"] += int(
                    (current_correct & (raw_pred != gt_gpu)).sum().item()
                )

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
        "total_loss",
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

    rescue_den = max(totals["rescue_pixels"], 1)
    totals["proposal_rescue_ce"] = totals.pop("proposal_rescue_ce_sum") / rescue_den
    totals["proposal_raw_rescue_recovery_rate"] = (
        totals["proposal_raw_rescue_recovered"] / rescue_den
    )
    correct_den = max(totals["proposal_raw_current_correct_pixels"], 1)
    totals["proposal_raw_current_correct_damage_rate"] = (
        totals["proposal_raw_current_correct_damaged"] / correct_den
    )
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
        raise RuntimeError("No valid C-V9 training sequences")

    frame_total = max(sum(row["frames"] for row in rows), 1)
    rescue_total = max(sum(row["rescue_pixels"] for row in rows), 1)
    correct_total = max(
        sum(row["proposal_raw_current_correct_pixels"] for row in rows),
        1,
    )
    result = {
        "frames": sum(row["frames"] for row in rows),
        "frames_with_rescue": sum(row["frames_with_rescue"] for row in rows),
        "rescue_pixels": sum(row["rescue_pixels"] for row in rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "proposal_loss_weight": PROPOSAL_LOSS_WEIGHT,
    }
    for key in (
        "segmentation_ce",
        "total_loss",
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

    result["proposal_rescue_ce"] = sum(
        row["proposal_rescue_ce"] * row["rescue_pixels"] for row in rows
    ) / rescue_total
    result["proposal_raw_rescue_recovery_rate"] = sum(
        row["proposal_raw_rescue_recovered"] for row in rows
    ) / rescue_total
    result["proposal_raw_current_correct_pixels"] = sum(
        row["proposal_raw_current_correct_pixels"] for row in rows
    )
    result["proposal_raw_current_correct_damage_rate"] = sum(
        row["proposal_raw_current_correct_damaged"] for row in rows
    ) / correct_total
    return result


def _rename_metrics(raw_metrics):
    if "c_v7" not in raw_metrics:
        raise KeyError("shared C-V7 evaluator did not return its candidate slot")
    return {
        "host": raw_metrics["host"],
        "c_v3_base": raw_metrics["c_v3_base"],
        MODEL_KEY: raw_metrics["c_v7"],
    }


def _selection_key(metrics):
    candidate = metrics[MODEL_KEY]
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

    # Fresh C-V8 architecture initialization: do not continue from trained C-V8.
    # This isolates the supervision change from continuation/pretraining effects.
    corrector = DirectErrorProposalCorrector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=c_v5.HISTORY_LENGTH,
        hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
        g_max=G_MAX,
        gate_bias=GATE_BIAS,
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

    zero_step = c_v8._zero_step_equality_check(
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

    # Same C-V8 evaluation protocol; RAFT remains metric-only.
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
        raw_metrics, diagnostics = c_v7._evaluate(
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
        metrics = _rename_metrics(raw_metrics)
        diagnostics.pop("raw_history_enters_correction_head", None)
        diagnostics.update(
            {
                "semantic_proposal_source": "concat validity-gated e1..e4",
                "proposal_head": "unchanged C-V8 bias-free zero-init 1x1 Conv 76D->19D",
                "proposal_supervision": True,
                "proposal_supervision_scope": "formal Rescue pixels only",
                "proposal_supervision_path": "Z_raw = Z_cur + upsample(DeltaZ_proposal_raw)",
                "proposal_loss_weight": PROPOSAL_LOSS_WEIGHT,
                "proposal_loss_gate_bypass": True,
                "proposal_loss_tanh_bypass": True,
                "proposal_loss_gt_model_input": False,
                "proposal_aware_gate": True,
                "proposal_gate_input_detached": False,
                "joint_final_ce_gradient": True,
                "tanh_bound_retained": True,
                "g_max": corrector.g_max,
                "fixed_gate_operating_point": True,
            }
        )

        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_host": c_v7._delta_metrics(metrics[MODEL_KEY], metrics["host"]),
            "delta_vs_c_v3": c_v7._delta_metrics(
                metrics[MODEL_KEY], metrics["c_v3_base"]
            ),
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
                        "base_architecture": "C-V8 Direct Error Proposal + Temporal Error Memory + Proposal-Aware Gate",
                        "architecture_changed_from_c_v8": False,
                        "history_length": c_v5.HISTORY_LENGTH,
                        "history_source": "raw detached frozen C-V3 logits",
                        "causal_path": "History -> Prediction -> Prediction Error -> Proposal -> Gated Correction",
                        "prediction_error": "V_k * (P_cur - P_hist_k)",
                        "semantic_proposal_source": "concat(e1..e4)",
                        "proposal_head": "bias-free zero-init 1x1 Conv",
                        "proposal_input_channels": c_v5.NUM_CLASSES * c_v5.HISTORY_LENGTH,
                        "proposal_output_channels": c_v5.NUM_CLASSES,
                        "error_hidden_role": "motion-aligned temporal context for Gate",
                        "proposal_aware_gate": True,
                        "gate_channels": 1,
                        "g_max": corrector.g_max,
                        "gate_bias_init": corrector.gate_bias,
                        "z_cur_detached": True,
                        "output": "Z_final = Z_cur + g * tanh(DeltaZ_proposal)",
                        "validity_order": "upsample first, full-resolution mask second",
                    },
                    "training_supervision": {
                        "final_loss": "full-image segmentation CE",
                        "proposal_loss": "raw Proposal CE on formal Rescue pixels only",
                        "proposal_loss_weight": PROPOSAL_LOSS_WEIGHT,
                        "proposal_loss_path": "Z_raw = Z_cur + valid * upsample(DeltaZ_proposal_raw)",
                        "rescue_definition": "C-V3 wrong AND exists valid aligned history whose argmax equals GT",
                        "gate_target": False,
                        "temporal_loss": False,
                        "raft_training": False,
                        "gt_inference_input": False,
                    },
                    "zero_step": zero_step,
                },
                output / "best.pt",
            )
        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V9 produced no epoch result")

    summary = {
        "experiment": "C-V9 Rescue-Supervised Direct Prediction-Error Proposal",
        "base_architecture": "unchanged C-V8",
        "best": best,
        "history": history,
        "zero_step": zero_step,
        "selection_rule": {
            "hard_constraint": f"C-V9 mIoU >= fixed C-V4 E2 floor {MIOU_HARD_FLOOR:.16f}",
            "objective_after_constraint": "maximize C-V9 mTC, then mIoU",
            "fallback_if_all_fail_floor": "highest mIoU, then mTC",
        },
        "training_supervision": {
            "total_loss": "L_final_CE + L_proposal_rescue_CE",
            "proposal_loss_weight": PROPOSAL_LOSS_WEIGHT,
            "proposal_loss_weight_sweep": False,
            "proposal_loss_scope": "formal Rescue pixels only",
            "proposal_loss_bypasses_gate": True,
            "proposal_loss_bypasses_tanh": True,
            "gate_supervision_changed": False,
            "temporal_loss": False,
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
