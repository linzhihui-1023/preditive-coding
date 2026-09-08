"""C-V7 Task-Aware Prediction-Error Residual Correction.

中文：C-V7 任务感知预测误差残差修正训练。

This entry keeps the validated C-V7 architecture unchanged and changes only
training supervision.  The trainable path remains:

    History -> motion-aligned Prediction -> Prediction Error
            -> motion-aligned Error Memory -> bounded DeltaZ + single gate
            -> Z_final = detach(Z_C-V3) + g * DeltaZ

Compared with the CE-only C-V7 experiment, this version adds two training-only
pixel roles without telling the model which history age to choose:

1. Rescue pixels（救回像素）:
   frozen C-V3 is wrong, while at least one valid aligned historical hypothesis
   predicts the GT class.
2. Protection pixels（保护像素）:
   frozen C-V3 is correct, while at least one valid historical hypothesis
   conflicts with the current C-V3 prediction.

GT/history correctness is used only to define training masks.  It never enters
C-V7 inference and never creates a t-1/t-2/t-3/t-4 selector target.

Loss:
    L_seg = CE(Z_final, GT)
    L_rescue  = mean_R softplus(m_cur - m_final)
    L_protect = mean_P relu(m_cur - m_final)
    L_task = L_rescue + L_protect
    L = L_seg + lambda_task * L_task

lambda_task is calibrated once from trainable-parameter gradient norms using the
same strategy previously used by C-V3.  No temporal loss and no RAFT teacher are
used for training.  RAFT remains metric-only for Full9 mTC evaluation.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    sequence_groups,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_multihypothesis_error_residual_corrector import (
    MultiHypothesisErrorResidualCorrector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)

from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v7_error_residual_correction as c_v7,
)


EXPERIMENT = "c_v7_task_aware_correction"
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v7_task_aware_correction"
)
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v7_task_aware_correction"
GRAD_EPS = 1e-12


@torch.no_grad()
def _build_task_masks(c_v3_logits, candidate_rows, current_gt_cpu):
    """Build training-only Rescue/Protection masks at full resolution.

    No history age is selected.  The mask only answers whether usable semantic
    evidence exists somewhere in the valid K-frame hypothesis bank.
    """
    gt = current_gt_cpu.to(c_v3_logits.device, non_blocking=True)
    valid_gt = gt != c_v5.IGNORE_LABEL
    current_pred = c_v3_logits.detach().argmax(1)[0]
    current_correct = valid_gt & (current_pred == gt)

    any_history_valid = torch.zeros_like(valid_gt)
    history_can_rescue = torch.zeros_like(valid_gt)
    history_conflict = torch.zeros_like(valid_gt)
    rescue_by_age = [0] * len(candidate_rows)

    for index, row in enumerate(candidate_rows):
        valid_history = row["valid_full"][0].bool() & valid_gt
        history_pred = row["logits"].detach().argmax(1)[0]
        any_history_valid |= valid_history

        correct_history = valid_history & (history_pred == gt)
        history_can_rescue |= correct_history
        history_conflict |= valid_history & (history_pred != current_pred)

        rescue_k = (~current_correct) & correct_history
        rescue_by_age[index] = int(rescue_k.sum().item())

    rescue = valid_gt & (~current_correct) & history_can_rescue
    protect = valid_gt & current_correct & history_conflict

    return {
        "valid_gt": valid_gt,
        "current_correct": current_correct,
        "any_history_valid": any_history_valid,
        "history_can_rescue": history_can_rescue,
        "history_conflict": history_conflict,
        "rescue": rescue,
        "protect": protect,
        "rescue_by_age_nonexclusive": rescue_by_age,
    }


def _true_class_margin(logits, current_gt_cpu):
    """Return Z(y) - max_{c!=y} Z(c) for every pixel."""
    gt = current_gt_cpu.to(logits.device, non_blocking=True)
    safe_gt = gt.clamp(0, c_v5.NUM_CLASSES - 1)
    index = safe_gt.unsqueeze(0).unsqueeze(0)
    true_logit = logits.gather(1, index)[:, 0]

    other_logits = logits.clone()
    other_logits.scatter_(1, index, float("-inf"))
    best_other = other_logits.max(dim=1).values
    return true_logit - best_other


def _masked_mean(values, mask, reference):
    if bool(mask.any()):
        return values[mask].mean()
    return reference.sum() * 0.0


def _task_aware_losses(c_v3_logits, final_logits, current_gt_cpu, candidate_rows):
    """Compute CE + role-aware margin losses without an Oracle history target."""
    masks = _build_task_masks(c_v3_logits, candidate_rows, current_gt_cpu)
    target = current_gt_cpu.to(final_logits.device, non_blocking=True).unsqueeze(0)
    segmentation_ce = F.cross_entropy(
        final_logits,
        target,
        ignore_index=c_v5.IGNORE_LABEL,
    )

    current_margin = _true_class_margin(c_v3_logits.detach(), current_gt_cpu)[0]
    final_margin = _true_class_margin(final_logits, current_gt_cpu)[0]
    margin_delta = final_margin - current_margin

    rescue_loss = _masked_mean(
        F.softplus(current_margin - final_margin),
        masks["rescue"],
        final_logits,
    )
    protection_loss = _masked_mean(
        F.relu(current_margin - final_margin),
        masks["protect"],
        final_logits,
    )
    task_loss = rescue_loss + protection_loss

    with torch.no_grad():
        valid_count = int(masks["valid_gt"].sum().item())
        rescue_count = int(masks["rescue"].sum().item())
        protect_count = int(masks["protect"].sum().item())
        any_history_count = int(masks["any_history_valid"].sum().item())
        current_correct_count = int(masks["current_correct"].sum().item())
        rescue_margin_gain = (
            float(margin_delta[masks["rescue"]].mean().item())
            if rescue_count
            else 0.0
        )
        protection_margin_gain = (
            float(margin_delta[masks["protect"]].mean().item())
            if protect_count
            else 0.0
        )

    diagnostics = {
        "valid_pixels": valid_count,
        "current_correct_pixels": current_correct_count,
        "any_history_valid_pixels": any_history_count,
        "rescue_pixels": rescue_count,
        "protection_pixels": protect_count,
        "rescue_fraction_of_valid": rescue_count / max(valid_count, 1),
        "protection_fraction_of_valid": protect_count / max(valid_count, 1),
        "rescue_margin_gain": rescue_margin_gain,
        "protection_margin_gain": protection_margin_gain,
        "rescue_by_age_nonexclusive": masks["rescue_by_age_nonexclusive"],
    }
    return segmentation_ce, rescue_loss, protection_loss, task_loss, diagnostics


def _gradient_norm(loss, parameters):
    """C-V3 gradient-norm calibration strategy, reused unchanged."""
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    total = None
    for grad in grads:
        if grad is None:
            continue
        value = grad.detach().float().square().sum()
        total = value if total is None else total + value
    if total is None:
        return 0.0
    return float(total.sqrt().item())


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
    task_scale_state,
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

    buffered_seg = []
    buffered_task = []
    frames_in_window = 0
    trainable = [parameter for parameter in corrector.parameters() if parameter.requires_grad]

    totals = {
        "frames": 0,
        "optimizer_steps": 0,
        "segmentation_ce": 0.0,
        "rescue_loss": 0.0,
        "protection_loss": 0.0,
        "task_loss": 0.0,
        "total_loss": 0.0,
        "prediction_error_abs": 0.0,
        "dynamics_error_abs": 0.0,
        "delta_z_abs": 0.0,
        "gate_mean": 0.0,
        "applied_correction_abs": 0.0,
        "error_reliability_mean": 0.0,
        "sign_agreement_mean": 0.0,
        "valid_fraction_mean": 0.0,
        "valid_pixels": 0,
        "current_correct_pixels": 0,
        "any_history_valid_pixels": 0,
        "rescue_pixels": 0,
        "protection_pixels": 0,
        "rescue_margin_gain_sum": 0.0,
        "protection_margin_gain_sum": 0.0,
        "rescue_by_age_nonexclusive": [0] * corrector.history_length,
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

        (
            segmentation_ce,
            rescue_loss,
            protection_loss,
            task_loss,
            task_diag,
        ) = _task_aware_losses(
            c_v3_logits,
            evidence["final_logits_full"],
            current_gt,
            candidate_rows,
        )
        if not all(
            bool(torch.isfinite(loss).item())
            for loss in (segmentation_ce, rescue_loss, protection_loss, task_loss)
        ):
            raise FloatingPointError("Non-finite C-V7 task-aware loss")

        buffered_seg.append(segmentation_ce)
        buffered_task.append(task_loss)

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
            totals["rescue_loss"] += float(rescue_loss.detach().item())
            totals["protection_loss"] += float(protection_loss.detach().item())
            totals["task_loss"] += float(task_loss.detach().item())
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

            for key in (
                "valid_pixels",
                "current_correct_pixels",
                "any_history_valid_pixels",
                "rescue_pixels",
                "protection_pixels",
            ):
                totals[key] += int(task_diag[key])
            totals["rescue_margin_gain_sum"] += (
                task_diag["rescue_margin_gain"] * task_diag["rescue_pixels"]
            )
            totals["protection_margin_gain_sum"] += (
                task_diag["protection_margin_gain"] * task_diag["protection_pixels"]
            )
            for index, value in enumerate(task_diag["rescue_by_age_nonexclusive"]):
                if index < corrector.history_length:
                    totals["rescue_by_age_nonexclusive"][index] += int(value)

        frames_in_window += 1
        boundary = frames_in_window >= tbptt_steps or frame_index == len(samples) - 1
        if boundary:
            window_seg = torch.stack(buffered_seg).mean()
            window_task = torch.stack(buffered_task).mean()

            if task_scale_state["value"] is None:
                g_seg = _gradient_norm(window_seg, trainable)
                g_task = _gradient_norm(window_task, trainable)
                if g_seg > GRAD_EPS and g_task > GRAD_EPS:
                    task_scale_state["value"] = g_seg / g_task
                    task_scale_state["seg_grad_norm"] = g_seg
                    task_scale_state["task_grad_norm"] = g_task
                    task_scale_state["calibration_sequence_length"] = len(samples)
                    task_scale_state["calibration_frame_index"] = frame_index

            lambda_task = task_scale_state["value"]
            if lambda_task is None:
                lambda_task = 0.0
            window_loss = window_seg + float(lambda_task) * window_task
            optimizer.zero_grad(set_to_none=True)
            window_loss.backward()
            optimizer.step()
            totals["optimizer_steps"] += 1
            totals["total_loss"] += float(window_loss.detach().item())

            buffered_seg = []
            buffered_task = []
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
    windows = max(totals["optimizer_steps"], 1)
    for key in (
        "segmentation_ce",
        "rescue_loss",
        "protection_loss",
        "task_loss",
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
    totals["total_loss"] /= windows
    totals["rescue_fraction_of_valid"] = totals["rescue_pixels"] / max(
        totals["valid_pixels"], 1
    )
    totals["protection_fraction_of_valid"] = totals["protection_pixels"] / max(
        totals["valid_pixels"], 1
    )
    totals["rescue_margin_gain"] = totals["rescue_margin_gain_sum"] / max(
        totals["rescue_pixels"], 1
    )
    totals["protection_margin_gain"] = totals["protection_margin_gain_sum"] / max(
        totals["protection_pixels"], 1
    )
    totals.pop("rescue_margin_gain_sum")
    totals.pop("protection_margin_gain_sum")
    totals["lambda_task"] = (
        float(task_scale_state["value"])
        if task_scale_state["value"] is not None
        else 0.0
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
    task_scale_state,
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
            task_scale_state,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("No valid C-V7 task-aware training sequences")

    frame_total = max(sum(row["frames"] for row in rows), 1)
    window_total = max(sum(row["optimizer_steps"] for row in rows), 1)
    result = {
        "frames": sum(row["frames"] for row in rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "valid_pixels": sum(row["valid_pixels"] for row in rows),
        "current_correct_pixels": sum(row["current_correct_pixels"] for row in rows),
        "any_history_valid_pixels": sum(row["any_history_valid_pixels"] for row in rows),
        "rescue_pixels": sum(row["rescue_pixels"] for row in rows),
        "protection_pixels": sum(row["protection_pixels"] for row in rows),
        "rescue_by_age_nonexclusive": [
            sum(row["rescue_by_age_nonexclusive"][index] for row in rows)
            for index in range(corrector.history_length)
        ],
    }
    for key in (
        "segmentation_ce",
        "rescue_loss",
        "protection_loss",
        "task_loss",
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
    result["total_loss"] = sum(
        row["total_loss"] * row["optimizer_steps"] for row in rows
    ) / window_total
    result["rescue_fraction_of_valid"] = result["rescue_pixels"] / max(
        result["valid_pixels"], 1
    )
    result["protection_fraction_of_valid"] = result["protection_pixels"] / max(
        result["valid_pixels"], 1
    )
    result["rescue_margin_gain"] = sum(
        row["rescue_margin_gain"] * row["rescue_pixels"] for row in rows
    ) / max(result["rescue_pixels"], 1)
    result["protection_margin_gain"] = sum(
        row["protection_margin_gain"] * row["protection_pixels"] for row in rows
    ) / max(result["protection_pixels"], 1)
    result["lambda_task"] = (
        float(task_scale_state["value"])
        if task_scale_state["value"] is not None
        else 0.0
    )
    result["task_scale_calibration"] = task_scale_state.copy()
    return result


def _selection_key(metrics):
    return c_v7._selection_key(metrics)


def _delta_metrics(candidate, reference):
    return c_v7._delta_metrics(candidate, reference)


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
    parser.add_argument("--g-max", type=float, default=c_v7.G_MAX)
    parser.add_argument("--gate-bias", type=float, default=c_v7.GATE_BIAS)
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

    zero_step = c_v7._zero_step_equality_check(
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

    task_scale_state = {
        "value": None,
        "seg_grad_norm": None,
        "task_grad_norm": None,
        "calibration_sequence_length": None,
        "calibration_frame_index": None,
    }

    # Metric-only RAFT. It is never passed into _train_epoch.
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
            task_scale_state,
        )
        metrics, diagnostics = c_v7._evaluate(
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
                    "task_scale_calibration": task_scale_state.copy(),
                    "architecture": {
                        "history_length": c_v5.HISTORY_LENGTH,
                        "history_source": "raw detached frozen C-V3 logits",
                        "causal_path": "History -> Prediction -> Prediction Error -> Residual Correction",
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
                    },
                    "training": {
                        "base_loss": "final segmentation CE",
                        "rescue_mask": "C-V3 wrong AND at least one valid history predicts GT",
                        "protection_mask": "C-V3 correct AND at least one valid history conflicts with C-V3",
                        "rescue_loss": "softplus(m_cur - m_final) on Rescue pixels",
                        "protection_loss": "relu(m_cur - m_final) on Protection pixels",
                        "history_age_target": False,
                        "task_weight": "one-time C-V3-style gradient-norm calibration",
                        "temporal_loss": False,
                        "raft_training": False,
                        "raft_inference_decision": False,
                    },
                    "zero_step": zero_step,
                },
                output / "best.pt",
            )
        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V7 task-aware training produced no epoch result")

    summary = {
        "experiment": "C-V7 Task-Aware Multi-Hypothesis Prediction-Error Residual Correction",
        "best": best,
        "history": history,
        "zero_step": zero_step,
        "selection_rule": {
            "hard_constraint": (
                f"C-V7 mIoU >= fixed C-V4 E2 floor {c_v7.MIOU_HARD_FLOOR:.16f}"
            ),
            "objective_after_constraint": "maximize C-V7 mTC, then mIoU",
            "fallback_if_all_fail_floor": "highest mIoU, then mTC",
        },
        "architecture": {
            "architecture_changed_vs_c_v7_ce_only": False,
            "history_length": c_v5.HISTORY_LENGTH,
            "causal_path": "History -> Prediction -> Prediction Error -> Residual Correction",
            "raw_history_enters_correction_head": False,
            "error_hidden_motion_aligned": True,
            "error_hidden_reliability_gated": True,
            "z_cur_detached": True,
            "bounded_logit_residual": True,
            "g_max": corrector.g_max,
            "gate_bias_init": corrector.gate_bias,
            "correction_head_zero_initialized": True,
        },
        "training": {
            "base_loss": "final segmentation CE",
            "rescue_mask": "C-V3 wrong AND at least one valid history predicts GT",
            "protection_mask": "C-V3 correct AND at least one valid history conflicts with C-V3",
            "rescue_loss": "softplus(m_cur - m_final)",
            "protection_loss": "relu(m_cur - m_final)",
            "history_age_target": False,
            "task_scale_calibration": task_scale_state.copy(),
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
