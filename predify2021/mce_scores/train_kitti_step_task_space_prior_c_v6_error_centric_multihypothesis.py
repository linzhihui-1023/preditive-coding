"""C-V6: Error-Centric Multi-Hypothesis Temporal Predictive Coding.

中文：C-V6 误差中心多假设时序预测编码。

This branch preserves the frozen C-V5 foundation (Host, motion observer,
motion residual predictor, E1 correction, C-V3 semantic memory and K=4
candidate bank) while replacing the top-level generic semantic selector with
an error-only controller.

Hard constraint / 硬约束：
- raw Current / History semantic probabilities never enter the C-V6 controller;
- every historical candidate first becomes a strict validity-gated prediction
  error e_k = V_k * (P_current - P_history_k);
- V_k requires both accumulated low-resolution motion-path validity and
  conservative full-resolution warp validity;
- the t-1 error continues to drive the explicit Euler Dynamics Error state;
- full semantic candidates remain outside the controller and are used only
  after the selector chooses Current / t-1 / ... / t-K;
- frozen C-V5 is used only as a training-time decision teacher and is absent at
  C-V6 inference.

The frozen upstream boundary is detached. Therefore this stage changes only the
new C-V6 selector optimization and cannot alter C-V3, motion or Host weights.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import load_fast_b_model
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_residual import (
    _load_frozen_observer,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask import (
    _load_frozen_residual,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask_r2 import (
    _validate_bounded_motion_checkpoint,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_utility_from_e1_r2 import (
    _load_frozen_e1_base,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v3_temporal_semantic_prediction import (
    _load_frozen_cv3_refiner,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v4_stateful_semantic_hysteresis_main import (
    _warp_low_state_zero_invalid,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_multihypothesis_error_selector import (
    MultiHypothesisErrorSelector,
    build_multihypothesis_error_evidence,
    strict_controller_validity,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)

# Reuse the validated C-V5 candidate bank, target construction and evaluation.
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)


EXPERIMENT = "c_v6_error_centric_multihypothesis"
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v6_error_centric_multihypothesis"
)
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v6_error_centric_multihypothesis"
C_V5_TEACHER_CHECKPOINT_DEFAULT = str(Path(c_v5.OUTPUT_DEFAULT) / "best.pt")
# The first C-V6 experiment is a clean causal baseline: the selector is
# trained only from the GT/RAFT task target.  A frozen C-V5 teacher is an
# optional preservation aid, enabled only when a positive weight is supplied.
DISTILL_WEIGHT = 0.0
DISTILL_TEMPERATURE = 1.0


def _strict_history_validities(
    current_probability,
    candidate_rows,
    low_path_validities,
    history_length,
):
    """Return controller validities using low-path AND full-warp validity."""
    low_size = tuple(current_probability.shape[-2:])
    strict = []
    for index in range(history_length):
        if index < len(candidate_rows):
            strict.append(
                strict_controller_validity(
                    low_path_validities[index],
                    candidate_rows[index]["valid_full"],
                    low_size,
                ).to(current_probability.dtype)
            )
        else:
            strict.append(torch.zeros_like(current_probability[:, :1]))
    return strict


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
    """Build C-V6 controller evidence without raw semantic shortcut inputs."""
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
    history_validities = _strict_history_validities(
        current_probability,
        candidate_rows,
        low_path_validities,
        selector.history_length,
    )

    multi = build_multihypothesis_error_evidence(
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
        warped_dynamics, _ = _warp_low_state_zero_invalid(
            dynamics_state,
            pending_motion,
        )
        warped_dynamics = (
            warped_dynamics
            * history1_valid
            * transportability_low.detach().clamp(0.0, 1.0)
        )

    dynamics_state = dynamics.step(
        primary_error,
        warped_dynamics,
    ).detach()

    row = selector(
        prediction_errors,
        dynamics_state,
        multi["current_margin"],
        multi["history_margins"],
        transportability_low.detach(),
        memory_reliability_low.detach(),
        [validity.detach() for validity in history_validities],
        selector_hidden,
    )

    selector_logits_full = F.interpolate(
        row["selector_logits"],
        size=tuple(c_v3_logits.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )
    for index in range(selector.history_length):
        channel = index + 1
        if index < len(candidate_rows):
            valid = candidate_rows[index]["valid_full"]
            selector_logits_full[:, channel][~valid] = -1e4
        else:
            selector_logits_full[:, channel].fill_(-1e4)

    return {
        "row": row,
        "selector_logits_full": selector_logits_full,
        "prediction_error": primary_error,
        "prediction_errors": prediction_errors,
        "dynamics_state": dynamics_state,
        "history_validities": history_validities,
    }


def _load_c_v5_teacher(checkpoint_path, device="cuda"):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"C-V5 teacher checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if "selector_state_dict" not in payload:
        raise KeyError("C-V5 teacher checkpoint lacks selector_state_dict")
    teacher = c_v5.MultiFrameSemanticSelector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=c_v5.HISTORY_LENGTH,
        hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
    ).to(device)
    teacher.load_state_dict(payload["selector_state_dict"], strict=True)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher, payload


def _distillation_kl_sum(student_logits, teacher_logits, temperature):
    """KL teacher->student summed over selected pixels."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must share shape")
    if student_logits.numel() == 0:
        return student_logits.sum() * 0.0
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("distillation temperature must be positive")
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    return (
        F.kl_div(student_log_prob, teacher_prob, reduction="sum")
        * (temperature * temperature)
    )


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
    previous_image, previous_host_logits, previous_low, previous_c1, _ = frame1

    raw_history = [previous_host_logits.detach()]
    motion_history = []
    previous_cv3_logits = previous_host_logits.detach()

    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = None
    selector_hidden = None
    dynamics_state = None
    teacher_hidden = None
    teacher_dynamics_state = None

    task_loss_sums = []
    task_loss_counts = []
    distill_loss_sums = []
    distill_loss_counts = []
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
    }
    target_totals = c_v5._new_target_totals(selector.history_length)

    for frame_index in range(2, len(samples)):
        current_image, host_logits, host_low, current_c1, output_size = c_v5._host_observation(
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

        teacher_evidence = None
        if teacher_selector is not None and distill_weight > 0.0:
            with torch.no_grad():
                teacher_evidence = teacher_evidence_fn(
                    teacher_selector,
                    teacher_dynamics,
                    c_v3_logits,
                    candidate_rows,
                    pending_motion,
                    e1["transportability_low"],
                    memory_row["memory_reliability"],
                    teacher_hidden,
                    teacher_dynamics_state,
                )
                teacher_hidden = teacher_evidence["row"]["hidden"]
                teacher_dynamics_state = teacher_evidence["dynamics_state"]

            teacher_full = raft.current_to_previous(current_image, previous_image)
            (
                _,
                temporal_target,
                supervised,
                decision,
                target_diag,
            ) = c_v5._build_multiframe_target(
                c_v3_logits,
                candidate_rows,
                previous_cv3_logits,
                current_gt,
                teacher_full,
            )
            c_v5._add_target_totals(target_totals, target_diag)

        if bool(supervised.any()):
            logits = evidence["selector_logits_full"][0].permute(1, 2, 0)[supervised]
            target = temporal_target[supervised]
            task_loss_sum = F.cross_entropy(logits, target, reduction="sum")
            count = int(supervised.sum().item())
            task_loss_sums.append(task_loss_sum)
            task_loss_counts.append(count)
            totals["supervised_frames"] += 1
            totals["supervised_pixels"] += count
            totals["selector_ce_per_pixel"] += float(task_loss_sum.detach().item())

        if teacher_evidence is not None and distill_weight > 0.0 and bool(decision.any()):
            student_logits = evidence["selector_logits_full"][0].permute(1, 2, 0)[decision]
            teacher_logits = teacher_evidence["selector_logits_full"][0].permute(1, 2, 0)[decision]
            distill_loss_sum = _distillation_kl_sum(
                student_logits,
                teacher_logits,
                distill_temperature,
            )
            count = int(decision.sum().item())
            distill_loss_sums.append(distill_loss_sum)
            distill_loss_counts.append(count)
            totals["distill_pixels"] += count
            totals["distill_kl_per_pixel"] += float(distill_loss_sum.detach().item())
            with torch.no_grad():
                teacher_choice = teacher_logits.argmax(dim=-1)
                student_choice = student_logits.argmax(dim=-1)
                totals["teacher_student_agree_pixels"] += int(
                    (teacher_choice == student_choice).sum().item()
                )
                totals["teacher_student_compare_pixels"] += count

        with torch.no_grad():
            hard_selection = evidence["selector_logits_full"].argmax(1)[0]
            for index in range(selector.history_length + 1):
                totals["selected_candidate_counts"][index] += int(
                    (hard_selection == index).sum().item()
                )
            totals["prediction_error_abs"] += float(
                evidence["prediction_error"].abs().mean().item()
            )
            totals["dynamics_error_abs"] += float(
                dynamics_state.abs().mean().item()
            )

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
            loss_terms = []
            if task_loss_sums:
                task_count = max(sum(task_loss_counts), 1)
                loss_terms.append(torch.stack(task_loss_sums).sum() / float(task_count))
            if distill_weight > 0.0 and distill_loss_sums:
                distill_count = max(sum(distill_loss_counts), 1)
                loss_terms.append(
                    float(distill_weight)
                    * torch.stack(distill_loss_sums).sum()
                    / float(distill_count)
                )
            if loss_terms:
                window_loss = torch.stack(loss_terms).sum()
                optimizer.zero_grad(set_to_none=True)
                window_loss.backward()
                optimizer.step()
                totals["optimizer_steps"] += 1

            task_loss_sums = []
            task_loss_counts = []
            distill_loss_sums = []
            distill_loss_counts = []
            frames_in_window = 0
            if selector_hidden is not None:
                selector_hidden = selector_hidden.detach()
            if teacher_hidden is not None:
                teacher_hidden = teacher_hidden.detach()
            if memory_state is not None:
                memory_state = memory_state.detach()
            if dynamics_state is not None:
                dynamics_state = dynamics_state.detach()
            if teacher_dynamics_state is not None:
                teacher_dynamics_state = teacher_dynamics_state.detach()

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
    totals["selector_ce_per_pixel"] /= max(totals["supervised_pixels"], 1)
    totals["distill_kl_per_pixel"] /= max(totals["distill_pixels"], 1)
    totals["teacher_student_agreement"] = (
        totals["teacher_student_agree_pixels"]
        / max(totals["teacher_student_compare_pixels"], 1)
    )
    totals["prediction_error_abs"] /= frames
    totals["dynamics_error_abs"] /= frames
    totals["targets"] = c_v5._target_rates(target_totals)
    return totals


def _train_epoch_distilled(
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
    if teacher_selector is not None:
        teacher_selector.eval()
    refiner.eval()
    correction.eval()
    mask_predictor.eval()
    rows = []
    target_totals = c_v5._new_target_totals(selector.history_length)

    for samples in groups.values():
        row = _train_sequence_distilled(
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
        raise RuntimeError("No valid C-V6 training sequences")

    frame_total = max(sum(row["frames"] for row in rows), 1)
    supervised_total = max(sum(row["supervised_pixels"] for row in rows), 1)
    distill_total = max(sum(row["distill_pixels"] for row in rows), 1)
    compare_total = max(
        sum(row["teacher_student_compare_pixels"] for row in rows),
        1,
    )
    result = {
        "frames": sum(row["frames"] for row in rows),
        "supervised_frames": sum(row["supervised_frames"] for row in rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "supervised_pixels": sum(row["supervised_pixels"] for row in rows),
        "distill_pixels": sum(row["distill_pixels"] for row in rows),
        "selector_ce_per_pixel": sum(
            row["selector_ce_per_pixel"] * row["supervised_pixels"]
            for row in rows
        ) / supervised_total,
        "distill_kl_per_pixel": sum(
            row["distill_kl_per_pixel"] * row["distill_pixels"]
            for row in rows
        ) / distill_total,
        "teacher_student_agreement": sum(
            row["teacher_student_agree_pixels"] for row in rows
        ) / compare_total,
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
        "targets": c_v5._target_rates(target_totals),
        "distillation": {
            "teacher": "frozen C-V5 selector",
            "weight": float(distill_weight),
            "temperature": float(distill_temperature),
            "mask": "GT-valid decision pixels where at least one candidate differs",
            "teacher_inference": False,
        },
    }
    return result


def _rename_metrics(metrics):
    metrics = dict(metrics)
    metrics["c_v6"] = metrics.pop("c_v5")
    return metrics


def _annotate_diagnostics(diagnostics):
    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "controller_semantic_input": (
                "strict-validity-gated class-wise prediction errors e1..eK only"
            ),
            "raw_current_probability_in_controller": False,
            "raw_history_probability_in_controller": False,
            "prediction_error_reference": "t-1..t-K frozen C-V3 hypotheses",
            "deep_history_error_validity_gated": True,
            "controller_validity": (
                "accumulated low-path validity AND conservative full-warp validity"
            ),
            "dynamics_error_reference": "strict-validity-gated t-1 prediction error",
            "upstream_error_boundary_detached": True,
            "teacher_inference": False,
        }
    )
    return diagnostics


def _evaluate(*args, **kwargs):
    metrics, diagnostics = c_v5._evaluate(*args, **kwargs)
    return _rename_metrics(metrics), _annotate_diagnostics(diagnostics)


def _selection_key(metrics):
    base = metrics["c_v3_base"]
    candidate = metrics["c_v6"]
    preserved = candidate["mIoU"] >= base["mIoU"]
    if preserved:
        return (1, candidate["mTC"], candidate["mIoU"])
    return (0, candidate["mIoU"], candidate["mTC"])


def _delta_metrics(candidate, reference):
    return {
        key: candidate[key] - reference[key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=c_v5.FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=c_v5.OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=c_v5.RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--base-checkpoint", default=c_v5.BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v3-checkpoint", default=c_v5.C_V3_CHECKPOINT_DEFAULT)
    parser.add_argument(
        "--c-v5-teacher-checkpoint",
        default=C_V5_TEACHER_CHECKPOINT_DEFAULT,
    )
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=c_v5.EPOCHS)
    parser.add_argument("--tbptt-steps", type=int, default=c_v5.TBPTT_STEPS)
    parser.add_argument("--lr", type=float, default=c_v5.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=c_v5.WEIGHT_DECAY)
    parser.add_argument("--distill-weight", type=float, default=DISTILL_WEIGHT)
    parser.add_argument(
        "--distill-temperature",
        type=float,
        default=DISTILL_TEMPERATURE,
    )
    parser.add_argument("--dynamics-tau-e", type=float, default=c_v5.DYNAMICS_TAU_E)
    parser.add_argument("--dynamics-k-e", type=float, default=c_v5.DYNAMICS_K_E)
    parser.add_argument("--dynamics-dt", type=float, default=c_v5.DYNAMICS_DT)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--seed", type=int, default=c_v5.SEED)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.distill_weight < 0.0:
        raise ValueError("distill-weight must be non-negative")
    if args.distill_temperature <= 0.0:
        raise ValueError("distill-temperature must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    _validate_bounded_motion_checkpoint(args.residual_checkpoint)
    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = _load_frozen_observer(args.observer_checkpoint)
    residual, residual_payload = _load_frozen_residual(
        args.residual_checkpoint,
        observer,
    )
    correction, mask_predictor, base_payload = _load_frozen_e1_base(
        args.base_checkpoint,
        observer,
    )
    refiner, cv3_payload = _load_frozen_cv3_refiner(args.c_v3_checkpoint)

    selector = MultiHypothesisErrorSelector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=c_v5.HISTORY_LENGTH,
        hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
    ).cuda()
    dynamics = EulerDynamicsError(
        tau_e=args.dynamics_tau_e,
        k_e=args.dynamics_k_e,
        dt=args.dynamics_dt,
    )
    zero_step = c_v5._zero_step_check(selector)

    # Patch only the evidence builder used by the validated C-V5 evaluator.
    # Training uses the explicit distilled C-V6 loop below.
    original_selector_evidence = c_v5._selector_evidence
    c_v5._selector_evidence = _selector_evidence

    optimizer = torch.optim.AdamW(
        selector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    raft = FrozenRAFT()

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train)
    all_val_groups = sequence_groups(val)
    missing = [sequence for sequence in c_v5.FULL9 if sequence not in all_val_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {sequence: all_val_groups[sequence] for sequence in c_v5.FULL9}

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    try:
        oracle_metrics, oracle_diagnostics = _evaluate(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            selector,
            dynamics,
            val_groups,
            raft,
        )
        oracle_precheck = {
            "metrics": oracle_metrics,
            "diagnostics": oracle_diagnostics,
            "delta_multiframe_oracle_vs_c_v3": _delta_metrics(
                oracle_metrics["multiframe_semantic_temporal_oracle"],
                oracle_metrics["c_v3_base"],
            ),
            "single_frame_semantic_temporal_oracle_mTC": c_v5.SINGLE_FRAME_ORACLE_MTC,
            "multiframe_oracle_exceeds_single_frame_oracle_mTC": (
                oracle_metrics["multiframe_semantic_temporal_oracle"]["mTC"]
                > c_v5.SINGLE_FRAME_ORACLE_MTC
            ),
        }
        with (result_output / "oracle_precheck.json").open("w") as handle:
            json.dump(oracle_precheck, handle, indent=2)
        print(json.dumps({"oracle_precheck": oracle_precheck}, indent=2), flush=True)

        if args.oracle_only:
            return

        # C-V5 is optional.  With the default zero weight, train directly from
        # the GT/RAFT task target and do not require a teacher checkpoint.
        teacher_selector = None
        teacher_payload = None
        teacher_dynamics = None
        if args.distill_weight > 0.0:
            teacher_selector, teacher_payload = _load_c_v5_teacher(
                args.c_v5_teacher_checkpoint
            )
            teacher_dynamics = EulerDynamicsError(
                tau_e=args.dynamics_tau_e,
                k_e=args.dynamics_k_e,
                dt=args.dynamics_dt,
            )

        history = []
        best = None
        for epoch in range(1, args.epochs + 1):
            train_stats = _train_epoch_distilled(
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
                original_selector_evidence,
                raft,
                train_groups,
                optimizer,
                args.tbptt_steps,
                args.distill_weight,
                args.distill_temperature,
            )
            metrics, diagnostics = _evaluate(
                model,
                observer,
                residual,
                correction,
                mask_predictor,
                refiner,
                selector,
                dynamics,
                val_groups,
                raft,
            )

            row = {
                "epoch": epoch,
                "train": train_stats,
                "metrics": metrics,
                "diagnostics": diagnostics,
                "delta_vs_c_v3_base": {
                    name: _delta_metrics(metrics[name], metrics["c_v3_base"])
                    for name in (
                        "c_v6",
                        "multiframe_semantic_oracle",
                        "multiframe_semantic_temporal_oracle",
                    )
                },
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
                        "selector_state_dict": selector.state_dict(),
                        "metrics": metrics,
                        "diagnostics": diagnostics,
                        "train": train_stats,
                        "dynamics": dynamics.config(),
                        "distillation": {
                            "teacher_checkpoint": (
                                args.c_v5_teacher_checkpoint
                                if teacher_payload is not None
                                else None
                            ),
                            "teacher_epoch": (
                                teacher_payload.get("epoch")
                                if teacher_payload is not None
                                else None
                            ),
                            "weight": args.distill_weight,
                            "temperature": args.distill_temperature,
                            "teacher_inference": False,
                        },
                        "architecture": {
                            "history_length": c_v5.HISTORY_LENGTH,
                            "history_source": "raw detached frozen C-V3 logits",
                            "candidate_bank_semantics_enter_controller": False,
                            "controller_semantic_input": "strict-validity-gated e1..eK",
                            "signed_error_encoding": "[ReLU(e), ReLU(-e)]",
                            "controller_validity": (
                                "low-path validity AND conservative full-warp validity"
                            ),
                            "dynamics_error_reference": "strict-validity-gated e1",
                            "upstream_error_boundary_detached": True,
                            "history_logits_resampling": "one final warp per candidate",
                            "controller_output_feedback": False,
                            "selector_hidden_channels": c_v5.CONTROLLER_HIDDEN_CHANNELS,
                            "target_priority": "semantic correctness then temporal consistency",
                            "raft_inference": False,
                        },
                    },
                    output / "best.pt",
                )
            print(json.dumps(row, indent=2), flush=True)

        if best is None:
            raise RuntimeError("C-V6 produced no epoch result")

        base_metrics = best["metrics"]["c_v3_base"]
        candidate = best["metrics"]["c_v6"]
        summary = {
            "experiment": "C-V6 Error-Centric Multi-Hypothesis Temporal Predictive Coding",
            "oracle_precheck": oracle_precheck,
            "best": best,
            "history": history,
            "zero_step": zero_step,
            "dynamics": dynamics.config(),
            "distillation": {
                "teacher_checkpoint": (
                    args.c_v5_teacher_checkpoint
                    if teacher_payload is not None
                    else None
                ),
                "teacher_epoch": (
                    teacher_payload.get("epoch")
                    if teacher_payload is not None
                    else None
                ),
                "weight": args.distill_weight,
                "temperature": args.distill_temperature,
                "mask": "GT-valid decision pixels where at least one candidate differs",
                "teacher_inference": False,
            },
            "frozen_checkpoints": {
                "fast_b": args.fast_b_checkpoint,
                "observer": args.observer_checkpoint,
                "residual": args.residual_checkpoint,
                "e1_base": args.base_checkpoint,
                "c_v3_base": args.c_v3_checkpoint,
                "c_v5_teacher": (
                    args.c_v5_teacher_checkpoint
                    if teacher_payload is not None
                    else None
                ),
                "c_v3_checkpoint_epoch": cv3_payload.get("epoch"),
                "residual_experiment": residual_payload.get("experiment"),
                "e1_experiment": base_payload.get("experiment"),
            },
            "architecture": {
                "history_length": c_v5.HISTORY_LENGTH,
                "history_source": "raw detached frozen C-V3 logits",
                "candidate_bank_semantics_enter_controller": False,
                "controller_semantic_input": "strict-validity-gated class-wise e1..eK",
                "signed_error_encoding": "[ReLU(e), ReLU(-e)]",
                "controller_validity": (
                    "low-path validity AND conservative full-warp validity"
                ),
                "deep_history_error_validity_gated": True,
                "dynamics_error_role": (
                    "motion-compensated T-masked explicit error state from strict e1"
                ),
                "upstream_error_boundary_detached": True,
                "history_logits_resampling": "one final warp per candidate",
                "controller_output_feedback": False,
                "training_loss": (
                    "task CE + optional frozen C-V5 selector decision KL distillation"
                    if args.distill_weight > 0.0
                    else "GT/RAFT task CE only; no teacher distillation"
                ),
                "teacher_inference": False,
                "raft_inference": False,
            },
            "selection_rule": {
                "hard_constraint": "C-V6 mIoU >= frozen C-V3 Base mIoU",
                "objective_after_constraint": "maximize C-V6 mTC, then mIoU",
            },
            "target": {
                "reference_c_v3_mIoU": base_metrics["mIoU"],
                "expected_reference_c_v3_mIoU": c_v5.MIOU_REFERENCE,
                "mIoU_preserved": candidate["mIoU"] >= base_metrics["mIoU"],
                "mTC_target_low": c_v5.MTC_TARGET_LOW,
                "mTC_target_high": c_v5.MTC_TARGET_HIGH,
                "mTC_reached_74": candidate["mTC"] >= c_v5.MTC_TARGET_LOW,
            },
            "tbptt_steps": args.tbptt_steps,
            "epochs": args.epochs,
        }
        with (result_output / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        c_v5._selector_evidence = original_selector_evidence


if __name__ == "__main__":
    main()
