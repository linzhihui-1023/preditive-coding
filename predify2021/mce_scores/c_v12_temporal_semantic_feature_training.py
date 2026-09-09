"""Training/evaluation helpers for C-V12 Temporal Semantic Feature Correction.

中文：C-V12 时序条件语义特征修正训练与评测辅助模块。

C-V12 keeps a frozen C-V4 Stateful Semantic Hysteresis（有状态语义滞回）
as the temporal baseline. A new K=4 Prediction-Error（预测误差）branch predicts
only a c4 feature residual. The semantic feature effect is decoded by the frozen
Host Decoder（宿主解码器）and added on top of the frozen C-V4 baseline logits.
Thus zero feature correction reproduces C-V4 exactly.
"""

from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import VideoConsistency
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v4_stateful_semantic_hysteresis_main as c_v4,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_only import (
    FULL9,
    IGNORE_LABEL,
    NUM_CLASSES,
    _pair_mtc,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    StatefulSemanticHysteresisController,
)


HISTORY_LENGTH = 4
SEMANTIC_CHANNELS = 128
C4_CHANNELS = 2048
C_V4_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v4_stateful_semantic_hysteresis_main/best.pt"
)
C_V4_REFERENCE_EPOCH = 2
C_V4_REFERENCE_MIOU = 0.6637739071008685
RESCUE_LOSS_WEIGHT = 1.0


@torch.no_grad()
def host_feature_observation(model, sample):
    image = load_image(sample)
    raw = model.extract_backbone_features(image)
    output_size = tuple(image.shape[-2:])
    host_feature = HostFeature(raw.c4, raw.c1, output_size)
    logits = model.decode_from_host_feature(host_feature)
    logits_low = F.interpolate(
        logits,
        size=tuple(raw.c1.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )
    return {
        "image": image,
        "host_logits": logits,
        "host_low": logits_low,
        "c1": raw.c1.detach(),
        "c4": raw.c4.detach(),
        "output_size": output_size,
    }


def legacy_observation(row):
    return (
        row["image"],
        row["host_logits"],
        row["host_low"],
        row["c1"],
        row["output_size"],
    )


def load_frozen_c_v4_controller(path=C_V4_CHECKPOINT_DEFAULT):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("experiment") != "c_v4_stateful_semantic_hysteresis_main":
        raise RuntimeError("Expected the corrected C-V4 Stateful Semantic Hysteresis checkpoint")
    if "controller_state_dict" not in payload:
        raise RuntimeError("C-V4 checkpoint missing controller_state_dict")
    if int(payload.get("epoch", -1)) != C_V4_REFERENCE_EPOCH:
        raise RuntimeError(
            f"C-V12 requires C-V4 balanced-best Epoch {C_V4_REFERENCE_EPOCH}; "
            f"got Epoch {payload.get('epoch')}"
        )
    controller = StatefulSemanticHysteresisController(
        num_classes=NUM_CLASSES,
        hidden_channels=c_v4.CONTROLLER_HIDDEN_CHANNELS,
    ).cuda()
    controller.load_state_dict(payload["controller_state_dict"], strict=True)
    controller.eval().requires_grad_(False)
    return controller, payload


@torch.no_grad()
def build_prediction_error_and_temporal_state(
    controller,
    dynamics,
    c_v3_logits,
    candidate_rows,
    pending_motion,
    transportability_low,
    memory_reliability_low,
    controller_hidden,
    dynamics_state,
    history_length=HISTORY_LENGTH,
):
    """Build K errors and update the exact frozen C-V4 temporal core from t-1."""
    low_size = tuple(transportability_low.shape[-2:])
    current_low = F.interpolate(
        c_v3_logits.detach(),
        size=low_size,
        mode="bilinear",
        align_corners=False,
    )
    current_probability = F.softmax(current_low, dim=1)
    history_probabilities, history_validities = c_v5._pad_history_for_controller(
        current_probability,
        candidate_rows,
        history_length,
    )
    prediction_errors = [
        (current_probability - probability) * validity
        for probability, validity in zip(history_probabilities, history_validities)
    ]

    if dynamics_state is None:
        warped_dynamics = None
    else:
        warped_dynamics, _ = c_v4._warp_low_state_zero_invalid(
            dynamics_state,
            pending_motion,
        )
    dynamics_state = dynamics.step(
        prediction_errors[0],
        warped_dynamics,
    ).detach()

    temporal_row = controller(
        current_probability.detach(),
        history_probabilities[0].detach(),
        prediction_errors[0].detach(),
        dynamics_state.detach(),
        transportability_low.detach(),
        memory_reliability_low.detach(),
        history_validities[0].detach(),
        controller_hidden,
    )
    temporal_hidden = temporal_row["hidden"].detach()
    return {
        "prediction_errors": [error.detach() for error in prediction_errors],
        "history_validities_low": [value.detach() for value in history_validities],
        "current_probability": current_probability.detach(),
        "history_probabilities": [value.detach() for value in history_probabilities],
        "temporal_row": temporal_row,
        "temporal_hidden": temporal_hidden,
        "dynamics_state": dynamics_state,
    }


@torch.no_grad()
def cv4_baseline_logits(c_v3_logits, candidate_rows, temporal_row):
    """Construct logits whose argmax exactly follows frozen C-V4 hard hysteresis."""
    if not candidate_rows:
        return c_v3_logits.detach(), torch.zeros_like(c_v3_logits[:, :1], dtype=torch.bool)
    history_logits = candidate_rows[0]["logits"].detach()
    valid = candidate_rows[0]["valid_full"].unsqueeze(1).bool()
    keep_logit = F.interpolate(
        temporal_row["keep_logit"],
        size=tuple(c_v3_logits.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )
    current_pred = c_v3_logits.argmax(1, keepdim=True)
    history_pred = history_logits.argmax(1, keepdim=True)
    conflict = valid & current_pred.ne(history_pred)
    keep = (keep_logit > 0.0) & conflict
    baseline = torch.where(keep, history_logits, c_v3_logits.detach())
    return baseline, keep


@torch.no_grad()
def formal_rescue_mask(baseline_logits, candidate_rows, gt_cpu):
    gt = gt_cpu.to(baseline_logits.device, non_blocking=True).long()
    valid_gt = gt.ne(IGNORE_LABEL)
    baseline_pred = baseline_logits.argmax(1)[0]
    rescue = valid_gt & baseline_pred.ne(gt)
    any_history_correct = torch.zeros_like(rescue)
    for candidate in candidate_rows:
        history_pred = candidate["logits"].argmax(1)[0]
        any_history_correct |= candidate["valid_full"][0].bool() & history_pred.eq(gt)
    return rescue & any_history_correct


def decode_feature_correction(
    model,
    corrector,
    observation,
    baseline_logits,
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
    corrected_host_logits = model.decode_from_host_feature(
        HostFeature(
            row["corrected_c4"],
            observation["c1"],
            observation["output_size"],
        )
    )
    # Decoder-derived semantic effect of Delta-c4. At Delta-c4=0 this is exactly 0.
    feature_delta_logits = corrected_host_logits - observation["host_logits"].detach()
    final_logits = baseline_logits.detach() + feature_delta_logits
    return final_logits, feature_delta_logits, row


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
    memory_state = None
    controller_hidden = None
    dynamics_state = None

    optimizer.zero_grad(set_to_none=True)
    frames_in_window = 0
    totals = {
        "frames": 0,
        "optimizer_steps": 0,
        "final_ce": 0.0,
        "rescue_ce": 0.0,
        "rescue_pixels": 0,
        "rescue_recovered": 0,
        "baseline_correct_pixels": 0,
        "baseline_correct_damaged": 0,
        "delta_c4_abs": 0.0,
        "delta_c4_rms_ratio": 0.0,
        "feature_delta_logits_abs": 0.0,
        "temporal_gain_mean": 0.0,
        "temporal_gain_std": 0.0,
    }

    for frame_index in range(2, len(samples)):
        observation = host_feature_observation(model, samples[frame_index])
        gt_cpu = semantic_mask_from_panoptic_png(samples[frame_index]["mask_path"])

        with torch.no_grad():
            prior_low, _ = c_v5.warp_low_logits(previous_low, pending_motion)
            e1 = c_v5._frozen_e1_step(
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
            memory_row, memory_state, _, c_v3_logits = c_v5._frozen_cv3_step(
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
            candidate_rows = c_v5._build_history_candidates(
                raw_history,
                motion_history,
                pending_motion,
                corrector.history_length,
            )
            error_row = build_prediction_error_and_temporal_state(
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
            baseline_logits, _ = cv4_baseline_logits(
                c_v3_logits,
                candidate_rows,
                error_row["temporal_row"],
            )
            rescue = formal_rescue_mask(baseline_logits, candidate_rows, gt_cpu)

        final_logits, feature_delta_logits, correction_row = decode_feature_correction(
            model,
            corrector,
            observation,
            baseline_logits,
            error_row,
            e1["transportability_low"],
            memory_row["memory_reliability"],
        )
        target = gt_cpu.to(final_logits.device, non_blocking=True).long().unsqueeze(0)
        pixel_ce = F.cross_entropy(
            final_logits,
            target,
            ignore_index=IGNORE_LABEL,
            reduction="none",
        )[0]
        valid = gt_cpu.to(final_logits.device, non_blocking=True).ne(IGNORE_LABEL)
        final_ce = pixel_ce[valid].mean()
        rescue_pixels = int(rescue.sum().item())
        rescue_ce = pixel_ce[rescue].mean() if rescue_pixels else final_ce * 0.0
        loss = final_ce + RESCUE_LOSS_WEIGHT * rescue_ce
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite C-V12 feature-correction loss")
        loss.backward()
        frames_in_window += 1

        with torch.no_grad():
            final_pred = final_logits.argmax(1)[0]
            baseline_pred = baseline_logits.argmax(1)[0]
            gt_gpu = target[0]
            baseline_correct = valid & baseline_pred.eq(gt_gpu)
            totals["frames"] += 1
            totals["final_ce"] += float(final_ce.item())
            totals["rescue_ce"] += float(rescue_ce.item()) if rescue_pixels else 0.0
            totals["rescue_pixels"] += rescue_pixels
            totals["rescue_recovered"] += int((rescue & final_pred.eq(gt_gpu)).sum().item())
            totals["baseline_correct_pixels"] += int(baseline_correct.sum().item())
            totals["baseline_correct_damaged"] += int(
                (baseline_correct & final_pred.ne(gt_gpu)).sum().item()
            )
            delta = correction_row["delta_c4"]
            current = observation["c4"]
            totals["delta_c4_abs"] += float(delta.abs().mean().item())
            totals["delta_c4_rms_ratio"] += float(
                delta.square().mean().sqrt().div(current.square().mean().sqrt().clamp_min(1e-8)).item()
            )
            totals["feature_delta_logits_abs"] += float(feature_delta_logits.abs().mean().item())
            totals["temporal_gain_mean"] += float(correction_row["temporal_gain"].mean().item())
            totals["temporal_gain_std"] += float(correction_row["temporal_gain"].std().item())

            observed_motion = c_v5._observe_motion(
                observer,
                previous_low,
                previous_c1,
                observation["host_low"],
                observation["c1"],
            )
            motion_error = F.softmax(observation["host_low"], dim=1) - F.softmax(prior_low, dim=1)
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
            # Match a mean-over-frames window while releasing every frame graph immediately.
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
        "final_ce",
        "rescue_ce",
        "delta_c4_abs",
        "delta_c4_rms_ratio",
        "feature_delta_logits_abs",
        "temporal_gain_mean",
        "temporal_gain_std",
    ):
        totals[key] /= frames
    totals["rescue_recovery_rate"] = totals["rescue_recovered"] / max(totals["rescue_pixels"], 1)
    totals["baseline_correct_damage_rate"] = (
        totals["baseline_correct_damaged"] / max(totals["baseline_correct_pixels"], 1)
    )
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
    refiner.eval(); correction.eval(); mask_predictor.eval(); c_v4_controller.eval()
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
        raise RuntimeError("No valid C-V12 training sequences")

    frame_total = max(sum(row["frames"] for row in rows), 1)
    result = {
        "frames": sum(row["frames"] for row in rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "rescue_pixels": sum(row["rescue_pixels"] for row in rows),
        "rescue_recovered": sum(row["rescue_recovered"] for row in rows),
        "baseline_correct_pixels": sum(row["baseline_correct_pixels"] for row in rows),
        "baseline_correct_damaged": sum(row["baseline_correct_damaged"] for row in rows),
        "rescue_loss_weight": RESCUE_LOSS_WEIGHT,
    }
    for key in (
        "final_ce", "rescue_ce", "delta_c4_abs", "delta_c4_rms_ratio",
        "feature_delta_logits_abs", "temporal_gain_mean", "temporal_gain_std",
    ):
        result[key] = sum(row[key] * row["frames"] for row in rows) / frame_total
    result["rescue_recovery_rate"] = result["rescue_recovered"] / max(result["rescue_pixels"], 1)
    result["baseline_correct_damage_rate"] = (
        result["baseline_correct_damaged"] / max(result["baseline_correct_pixels"], 1)
    )
    return result


@torch.inference_mode()
def evaluate(
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
    raft_metric,
):
    names = ("host", "c_v3_base", "c_v4_frozen", "c_v12")
    confusion = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in names}
    mtc_sum = {name: 0.0 for name in names}
    mtc_count = {name: 0 for name in names}
    vc_sum = {name: {8: 0.0, 16: 0.0} for name in names}
    vc_count = {name: {8: 0, 16: 0} for name in names}
    diag = {
        "correction_frames": 0,
        "delta_c4_abs": 0.0,
        "delta_c4_rms_ratio": 0.0,
        "feature_delta_logits_abs": 0.0,
        "temporal_gain_mean": 0.0,
        "temporal_gain_std": 0.0,
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

        # C-V4 itself uses Host fallback for the first two frames.
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
        memory_state = None
        controller_hidden = None
        dynamics_state = None

        for frame_index in range(2, len(samples)):
            observation = host_feature_observation(model, samples[frame_index])
            gt_cpu = semantic_mask_from_panoptic_png(samples[frame_index]["mask_path"])
            prior_low, _ = c_v5.warp_low_logits(previous_low, pending_motion)
            e1 = c_v5._frozen_e1_step(
                correction, mask_predictor, observation["c1"], observation["host_low"],
                prior_low, pending_motion, semantic_state_low,
                transport_hidden, semantic_hidden, mask_hidden,
            )
            transport_hidden = e1["transport_hidden"]; semantic_hidden = e1["semantic_hidden"]
            mask_hidden = e1["mask_hidden"]; semantic_state_low = e1["semantic_state_low"]
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
            final_logits, feature_delta_logits, correction_row = decode_feature_correction(
                model, corrector, observation, c_v4_logits, error_row,
                e1["transportability_low"], memory_row["memory_reliability"],
            )

            predictions = {
                "host": observation["host_logits"].argmax(1),
                "c_v3_base": c_v3_logits.argmax(1),
                "c_v4_frozen": c_v4_logits.argmax(1),
                "c_v12": final_logits.argmax(1),
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
            diag["correction_frames"] += 1
            diag["rescue_pixels"] += int(rescue.sum().item())
            diag["rescue_recovered"] += int((rescue & final_pred.eq(gt_gpu)).sum().item())
            diag["baseline_correct_pixels"] += int(baseline_correct.sum().item())
            diag["baseline_correct_damaged"] += int((baseline_correct & final_pred.ne(gt_gpu)).sum().item())
            delta = correction_row["delta_c4"]
            diag["delta_c4_abs"] += float(delta.abs().mean().item())
            diag["delta_c4_rms_ratio"] += float(
                delta.square().mean().sqrt().div(observation["c4"].square().mean().sqrt().clamp_min(1e-8)).item()
            )
            diag["feature_delta_logits_abs"] += float(feature_delta_logits.abs().mean().item())
            diag["temporal_gain_mean"] += float(correction_row["temporal_gain"].mean().item())
            diag["temporal_gain_std"] += float(correction_row["temporal_gain"].std().item())

            observed_motion = c_v5._observe_motion(
                observer, previous_low, previous_c1,
                observation["host_low"], observation["c1"]
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
            pending_motion = next_motion.detach(); motion_hidden = next_motion_hidden.detach()
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
    for key in ("delta_c4_abs", "delta_c4_rms_ratio", "feature_delta_logits_abs", "temporal_gain_mean", "temporal_gain_std"):
        diag[key] /= frames
    diag["rescue_recovery_rate"] = diag["rescue_recovered"] / max(diag["rescue_pixels"], 1)
    diag["baseline_correct_damage_rate"] = diag["baseline_correct_damaged"] / max(diag["baseline_correct_pixels"], 1)
    diag.update({
        "temporal_base": "frozen C-V4 E2 Stateful Semantic Hysteresis",
        "history_length": corrector.history_length,
        "history_source": "raw detached frozen C-V3 logits",
        "semantic_correction_target": "Host c4 2048D feature",
        "semantic_correction_source": "concat validity-gated e1..e4 only",
        "feature_effect": "Decoder(c4 + Delta-c4) - Decoder(c4)",
        "final_composition": "frozen C-V4 logits + decoded feature effect",
        "controller_output_feedback": False,
        "corrected_output_history_feedback": False,
        "raft_training": False,
        "raft_metric_only": True,
    })
    return metrics, diag
