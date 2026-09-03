"""Predify Z4-only Stage C: prediction-error-driven semantic correction.

Stage P's trained Z4 predictor is frozen. Stage C trains only:
  - semantic_error_encoder
  - semantic_state_cell
  - semantic_restoration_head

The frozen Host, Z1 path, Z4 predictor, C4 adapter and C4 writeback remain in
forward for inference compatibility where required, but only the Z4 predictive
coding path participates in temporal prediction.

Training objective:
  L = L_seg + lambda_tc * L_TC + lambda_safe * L_safe

RAFT is training/evaluation-only. It is not part of inference.
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import (
    VideoConsistency,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
    sequence_groups,
)
from predify2021.mce_scores.evaluate_kitti_step_raft_error_memory import FrozenRAFT
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    load_components,
    residual_writeback_host_feature,
)
from predify2021.mce_scores.train_kitti_step_z4_only import (
    FAST_B_DEFAULT,
    z4_predict_next,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorRegulatedSemanticRestorationPredictor,
    HostFeature,
    UnifiedFeatures,
)

SEED = 0
NUM_CLASSES = 19
IGNORE = 255
TBPTT = 16
SEGMENTATION_POSITIONS = (4, 8, 12, 16)
MAX_EPOCHS = 15
PATIENCE = 3
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
HOST_CONFIDENCE = 0.70
TARGET_TC_GRAD_RATIO = 0.40
TARGET_SAFE_GRAD_RATIO = 0.20
MVC16_FLOOR_DELTA = -0.002
DEV3 = ("0002", "0010", "0018")

STAGE_P_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_z4_only_stage_p/best.pt"
)
OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_z4_only_stage_c"
RESULT_DEFAULT = "results/kitti_step_z4_only_stage_c"


def encode_clean(model, sample):
    image = load_image(sample)
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        observation = model.encode_backbone_features(raw)
    return image, observation, raw, tuple(image.shape[-2:])


def prediction_state(observation, pending_z4):
    """Build the restore_current interface without invoking any Z1 predictor."""
    return UnifiedFeatures(
        observation.z1,
        observation.z2,
        observation.z3,
        pending_z4,
    )


def corrected_logits(model, raw, observation, restored, output_size):
    """Use the frozen validated C4 adapter/writeback and frozen Host decoder."""
    zero_z1 = torch.zeros_like(observation.z1)
    zero_z2 = torch.zeros_like(observation.z2)
    zero_z3 = torch.zeros_like(observation.z3)
    delta = UnifiedFeatures(
        zero_z1,
        zero_z2,
        zero_z3,
        restored.z4 - observation.z4,
    )
    host_feature = residual_writeback_host_feature(
        model, raw, delta, output_size
    )
    if torch.is_grad_enabled():
        return checkpoint(
            lambda high: model.decode_from_host_feature(
                HostFeature(high, host_feature.low_level, host_feature.output_size)
            ),
            host_feature.tensor,
            use_reentrant=False,
        )
    return model.decode_from_host_feature(host_feature)


def host_logits(model, raw, output_size):
    with torch.no_grad():
        return model.decode_from_host_feature(
            HostFeature(raw.c4, raw.c1, output_size)
        )


def flow_grid(backward_flow, height, width):
    """Current-to-previous RAFT flow -> grid_sample coordinates."""
    flow = backward_flow
    source_h, source_w = flow.shape[-2:]
    if (source_h, source_w) != (height, width):
        flow = F.interpolate(
            flow, size=(height, width), mode="bilinear", align_corners=True
        )
        flow = flow.clone()
        flow[:, 0].mul_(width / source_w)
        flow[:, 1].mul_(height / source_h)
    y, x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    source_x = x.unsqueeze(0) + flow[:, 0]
    source_y = y.unsqueeze(0) + flow[:, 1]
    valid = (
        (source_x >= 0)
        & (source_x <= width - 1)
        & (source_y >= 0)
        & (source_y <= height - 1)
    )
    grid = torch.stack(
        (
            2.0 * source_x / max(width - 1, 1) - 1.0,
            2.0 * source_y / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    )
    return grid, valid


def temporal_loss(
    current_logits,
    previous_logits,
    backward_flow,
    previous_mask,
    current_mask,
):
    """RAFT-aligned probability consistency on valid same-label pixels."""
    height, width = current_logits.shape[-2:]
    grid, valid = flow_grid(backward_flow, height, width)
    current_prob = current_logits.softmax(1)
    previous_prob = previous_logits.detach().softmax(1)
    warped_previous = F.grid_sample(
        previous_prob,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    warped_previous_mask = F.grid_sample(
        previous_mask.float()[None, None],
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0].long()
    previous_confident = previous_prob.amax(1, keepdim=True)
    warped_confident = F.grid_sample(
        (previous_confident > HOST_CONFIDENCE).float(),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0] > 0.5
    keep = (
        valid[0]
        & (warped_previous_mask != IGNORE)
        & (current_mask != IGNORE)
        & (warped_previous_mask == current_mask)
        & warped_confident
    )
    if not keep.any():
        return current_logits.sum() * 0.0, 0.0
    per_pixel = F.kl_div(
        current_prob.clamp_min(1e-8).log(),
        warped_previous.clamp_min(1e-8),
        reduction="none",
    ).sum(1)[0]
    return per_pixel[keep].mean(), float(keep.float().mean().item())


def safe_host_loss(student_logits, frozen_host_logits, target):
    """Preserve pixels the frozen Host already predicts correctly/confidently."""
    host_prob = frozen_host_logits.detach().softmax(1)
    host_confidence, host_prediction = host_prob.max(1)
    safe = (
        (target != IGNORE)
        & (host_prediction[0] == target)
        & (host_confidence[0] > HOST_CONFIDENCE)
    )
    if not safe.any():
        return student_logits.sum() * 0.0, 0.0
    per_pixel = F.kl_div(
        student_logits.log_softmax(1),
        host_prob,
        reduction="none",
    ).sum(1)[0]
    return per_pixel[safe].mean(), float(safe.float().mean().item())


def semantic_parameters(predictor):
    modules = (
        predictor.semantic_error_encoder,
        predictor.semantic_state_cell,
        predictor.semantic_restoration_head,
    )
    return [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def assert_stage_c_contract(model, predictor):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Stage C Host/C4 interface must remain frozen")
    allowed = {
        "semantic_error_encoder",
        "semantic_state_cell",
        "semantic_restoration_head",
    }
    unexpected = [
        name
        for name, parameter in predictor.named_parameters()
        if parameter.requires_grad and name.split(".", 1)[0] not in allowed
    ]
    if unexpected:
        raise RuntimeError(
            f"Unexpected Stage C trainable predictor parameters: {unexpected[:8]}"
        )
    required = {
        name.split(".", 1)[0]
        for name, parameter in predictor.named_parameters()
        if parameter.requires_grad
    }
    if required != allowed:
        raise RuntimeError(
            f"Stage C trainable modules mismatch: expected={sorted(allowed)}, "
            f"observed={sorted(required)}"
        )
    for name in ("z4_dyn_recurrent", "z4_dyn_delta", "z1_dyn_recurrent", "z1_dyn_delta"):
        if any(parameter.requires_grad for parameter in getattr(predictor, name).parameters()):
            raise RuntimeError(f"{name} must be frozen in Stage C")


def load_stage_c(args):
    model, _ = load_components(
        STATIC_CHECKPOINT_DEFAULT,
        ADAPTER_CHECKPOINT_DEFAULT,
        args.dynamics_checkpoint,
        WRITEBACK_CHECKPOINT_DEFAULT,
    )
    fast_b_payload = torch.load(
        args.fast_b_checkpoint, map_location="cpu", weights_only=False
    )
    model.multi_layer_adapter.output_adapters[3].load_state_dict(
        fast_b_payload["c4_output_adapter_state_dict"], strict=True
    )
    model.host_conditioned_writebacks["3"].load_state_dict(
        fast_b_payload["c4_writeback_state_dict"], strict=True
    )

    stage_p_payload = torch.load(
        args.stage_p_checkpoint, map_location="cpu", weights_only=False
    )
    predictor = ErrorRegulatedSemanticRestorationPredictor(
        use_error_temporal_stats=True
    ).cuda()
    predictor.load_state_dict(stage_p_payload["model_state_dict"], strict=True)

    model.requires_grad_(False)
    predictor.requires_grad_(False)
    for module in (
        predictor.semantic_error_encoder,
        predictor.semantic_state_cell,
        predictor.semantic_restoration_head,
    ):
        module.requires_grad_(True)

    model.eval()
    predictor.train()
    assert_stage_c_contract(model, predictor)
    return model, predictor, fast_b_payload, stage_p_payload


def _grad_norm(loss, parameters):
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    values = [
        grad.detach().float().square().sum()
        for grad in grads
        if grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt().item())


def _clip_loss_probe(model, predictor, raft, samples):
    """Real Stage C forward/backward paths for gradient-ratio calibration."""
    clip = samples[: min(len(samples), TBPTT + 1)]
    if len(clip) < 3:
        raise RuntimeError("Stage C gradient probe requires at least 3 frames")

    previous_image, observation, raw, output_size = encode_clean(model, clip[0])
    previous_mask = semantic_mask_from_panoptic_png(
        clip[0]["mask_path"]
    ).cuda()
    previous_student_logits = host_logits(model, raw, output_size).detach()

    with torch.no_grad():
        pending_z4, h4 = z4_predict_next(
            predictor,
            observation.z4,
            torch.zeros_like(observation.z4),
            None,
        )
    semantic_hidden = predictor.initial_semantic_state(observation)
    error_stats = predictor.initial_error_temporal_statistics()

    seg_losses = []
    tc_losses = []
    safe_losses = []

    for local, sample in enumerate(clip[1:], 1):
        image, observation, raw, output_size = encode_clean(model, sample)
        mask = semantic_mask_from_panoptic_png(sample["mask_path"]).cuda()
        prior = prediction_state(observation, pending_z4.detach())
        restored, semantic_hidden, diagnostics = predictor.restore_current(
            observation,
            prior,
            semantic_hidden,
            error_temporal_state=error_stats,
        )
        error_stats = diagnostics["error_temporal_state"]
        student_logits = corrected_logits(
            model, raw, observation, restored, output_size
        )
        frozen_host_logits = host_logits(model, raw, output_size)
        backward_flow = raft.backward_flow(image, previous_image)

        if local in SEGMENTATION_POSITIONS or local == len(clip) - 1:
            seg_losses.append(
                F.cross_entropy(
                    student_logits,
                    mask.unsqueeze(0),
                    ignore_index=IGNORE,
                )
            )
            safe, _ = safe_host_loss(
                student_logits, frozen_host_logits, mask
            )
            safe_losses.append(safe)

        tc, _ = temporal_loss(
            student_logits,
            previous_student_logits,
            backward_flow,
            previous_mask,
            mask,
        )
        tc_losses.append(tc)

        prediction_error_z4 = observation.z4 - pending_z4
        with torch.no_grad():
            pending_z4, h4 = z4_predict_next(
                predictor,
                observation.z4,
                prediction_error_z4,
                h4,
            )
        previous_image = image
        previous_mask = mask
        previous_student_logits = student_logits.detach()

    return (
        torch.stack(seg_losses).mean(),
        torch.stack(tc_losses).mean(),
        torch.stack(safe_losses).mean(),
    )


def calibrate_loss_weights(model, predictor, raft, samples):
    parameters = semantic_parameters(predictor)
    predictor.zero_grad(set_to_none=True)
    lseg, ltc, lsafe = _clip_loss_probe(
        model, predictor, raft, samples
    )
    gseg = _grad_norm(lseg, parameters)
    gtc = _grad_norm(ltc, parameters)
    gsafe = _grad_norm(lsafe, parameters)
    predictor.zero_grad(set_to_none=True)
    if gseg <= 0 or gtc <= 0 or gsafe <= 0:
        raise RuntimeError(
            "Stage C gradient calibration failed: "
            f"gseg={gseg}, gtc={gtc}, gsafe={gsafe}"
        )
    lambda_tc = TARGET_TC_GRAD_RATIO * gseg / gtc
    lambda_safe = TARGET_SAFE_GRAD_RATIO * gseg / gsafe
    lambda_tc = float(min(max(lambda_tc, 1e-3), 100.0))
    lambda_safe = float(min(max(lambda_safe, 1e-3), 100.0))
    return {
        "raw_losses": {
            "Lseg": float(lseg.detach().item()),
            "LTC": float(ltc.detach().item()),
            "Lsafe": float(lsafe.detach().item()),
        },
        "raw_gradient_norms": {
            "Lseg": gseg,
            "LTC": gtc,
            "Lsafe": gsafe,
        },
        "target_gradient_ratios": {
            "TC_vs_seg": TARGET_TC_GRAD_RATIO,
            "safe_vs_seg": TARGET_SAFE_GRAD_RATIO,
        },
        "lambda_tc": lambda_tc,
        "lambda_safe": lambda_safe,
    }


def train_sequence(
    model,
    predictor,
    raft,
    samples,
    optimizer,
    lambda_tc,
    lambda_safe,
):
    totals = {
        "windows": 0,
        "frames": 0,
        "Lseg": 0.0,
        "LTC": 0.0,
        "Lsafe": 0.0,
        "total": 0.0,
        "prediction_error_abs": 0.0,
        "delta_z4_abs": 0.0,
        "semantic_gain": 0.0,
        "tc_valid_ratio": 0.0,
        "safe_pixel_ratio": 0.0,
    }
    if len(samples) < 2:
        return totals

    previous_image, observation, raw, output_size = encode_clean(
        model, samples[0]
    )
    previous_mask = semantic_mask_from_panoptic_png(
        samples[0]["mask_path"]
    ).cuda()
    previous_student_logits = host_logits(
        model, raw, output_size
    ).detach()

    with torch.no_grad():
        pending_z4, h4 = z4_predict_next(
            predictor,
            observation.z4,
            torch.zeros_like(observation.z4),
            None,
        )
    semantic_hidden = predictor.initial_semantic_state(observation)
    error_stats = predictor.initial_error_temporal_statistics()

    seg_losses = []
    tc_losses = []
    safe_losses = []
    error_values = []
    delta_values = []
    gain_values = []
    tc_ratios = []
    safe_ratios = []

    for frame_index, sample in enumerate(samples[1:], 1):
        image, observation, raw, output_size = encode_clean(model, sample)
        mask = semantic_mask_from_panoptic_png(sample["mask_path"]).cuda()

        prior = prediction_state(observation, pending_z4.detach())
        restored, semantic_hidden, diagnostics = predictor.restore_current(
            observation,
            prior,
            semantic_hidden,
            error_temporal_state=error_stats,
        )
        error_stats = diagnostics["error_temporal_state"]
        student_logits = corrected_logits(
            model, raw, observation, restored, output_size
        )
        frozen_host_logits = host_logits(model, raw, output_size)
        backward_flow = raft.backward_flow(image, previous_image)

        local_position = ((frame_index - 1) % TBPTT) + 1
        if (
            local_position in SEGMENTATION_POSITIONS
            or frame_index == len(samples) - 1
        ):
            seg_losses.append(
                F.cross_entropy(
                    student_logits,
                    mask.unsqueeze(0),
                    ignore_index=IGNORE,
                )
            )
            safe, safe_ratio = safe_host_loss(
                student_logits,
                frozen_host_logits,
                mask,
            )
            safe_losses.append(safe)
            safe_ratios.append(safe_ratio)

        tc, tc_ratio = temporal_loss(
            student_logits,
            previous_student_logits,
            backward_flow,
            previous_mask,
            mask,
        )
        tc_losses.append(tc)
        tc_ratios.append(tc_ratio)

        prediction_error_z4 = observation.z4 - pending_z4
        error_values.append(prediction_error_z4.detach().abs().mean())
        delta_values.append(
            (restored.z4 - observation.z4).detach().abs().mean()
        )
        gain_values.append(
            diagnostics["semantic_update_gain"].detach().mean()
        )

        with torch.no_grad():
            pending_z4, h4 = z4_predict_next(
                predictor,
                observation.z4,
                prediction_error_z4,
                h4,
            )

        previous_image = image
        previous_mask = mask
        previous_student_logits = student_logits.detach()
        totals["frames"] += 1

        window_end = (
            local_position == TBPTT
            or frame_index == len(samples) - 1
        )
        if not window_end:
            continue

        lseg = (
            torch.stack(seg_losses).mean()
            if seg_losses
            else student_logits.sum() * 0.0
        )
        ltc = torch.stack(tc_losses).mean()
        lsafe = (
            torch.stack(safe_losses).mean()
            if safe_losses
            else student_logits.sum() * 0.0
        )
        total = lseg + lambda_tc * ltc + lambda_safe * lsafe
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite Stage C objective")

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        totals["windows"] += 1
        totals["Lseg"] += float(lseg.detach().item())
        totals["LTC"] += float(ltc.detach().item())
        totals["Lsafe"] += float(lsafe.detach().item())
        totals["total"] += float(total.detach().item())
        totals["prediction_error_abs"] += float(
            torch.stack(error_values).mean().item()
        )
        totals["delta_z4_abs"] += float(
            torch.stack(delta_values).mean().item()
        )
        totals["semantic_gain"] += float(
            torch.stack(gain_values).mean().item()
        )
        totals["tc_valid_ratio"] += sum(tc_ratios) / max(len(tc_ratios), 1)
        totals["safe_pixel_ratio"] += (
            sum(safe_ratios) / max(len(safe_ratios), 1)
        )

        semantic_hidden = semantic_hidden.detach()
        error_stats = (
            error_stats.detach()
            if error_stats is not None
            else None
        )
        pending_z4 = pending_z4.detach()
        h4 = h4.detach() if h4 is not None else None

        seg_losses = []
        tc_losses = []
        safe_losses = []
        error_values = []
        delta_values = []
        gain_values = []
        tc_ratios = []
        safe_ratios = []

    for key in (
        "Lseg",
        "LTC",
        "Lsafe",
        "total",
        "prediction_error_abs",
        "delta_z4_abs",
        "semantic_gain",
        "tc_valid_ratio",
        "safe_pixel_ratio",
    ):
        totals[key] /= max(totals["windows"], 1)
    return totals


def train_epoch(
    model,
    predictor,
    raft,
    groups,
    optimizer,
    lambda_tc,
    lambda_safe,
):
    predictor.train()
    aggregate = {
        "sequences": 0,
        "windows": 0,
        "frames": 0,
        "Lseg": 0.0,
        "LTC": 0.0,
        "Lsafe": 0.0,
        "total": 0.0,
        "prediction_error_abs": 0.0,
        "delta_z4_abs": 0.0,
        "semantic_gain": 0.0,
        "tc_valid_ratio": 0.0,
        "safe_pixel_ratio": 0.0,
    }
    for samples in groups.values():
        row = train_sequence(
            model,
            predictor,
            raft,
            samples,
            optimizer,
            lambda_tc,
            lambda_safe,
        )
        aggregate["sequences"] += 1
        aggregate["windows"] += row["windows"]
        aggregate["frames"] += row["frames"]
        for key in (
            "Lseg",
            "LTC",
            "Lsafe",
            "total",
            "prediction_error_abs",
            "delta_z4_abs",
            "semantic_gain",
            "tc_valid_ratio",
            "safe_pixel_ratio",
        ):
            aggregate[key] += row[key] * row["windows"]

    for key in (
        "Lseg",
        "LTC",
        "Lsafe",
        "total",
        "prediction_error_abs",
        "delta_z4_abs",
        "semantic_gain",
        "tc_valid_ratio",
        "safe_pixel_ratio",
    ):
        aggregate[key] /= max(aggregate["windows"], 1)
    return aggregate


def pair_mtc(previous_prediction, current_prediction, backward_flow):
    height, width = current_prediction.shape[-2:]
    grid, valid = flow_grid(backward_flow, height, width)
    warped = F.grid_sample(
        previous_prediction.float().unsqueeze(1),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0].long()
    keep = valid[0]
    a = warped[keep].cpu()
    b = current_prediction[0][keep].cpu()
    if not a.numel():
        return float("nan")
    confusion = torch.bincount(
        NUM_CLASSES * a + b,
        minlength=NUM_CLASSES ** 2,
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    return float(torch.nanmean(compute_iou(confusion)).item())


@torch.inference_mode()
def evaluate(model, predictor, groups, raft):
    predictor.eval()
    confusion = {
        "host": torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64),
        "ours": torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64),
    }
    mvc_sums = {
        name: {8: 0.0, 16: 0.0}
        for name in confusion
    }
    mvc_counts = {
        name: {8: 0, 16: 0}
        for name in confusion
    }
    mtc_sum = {name: 0.0 for name in confusion}
    mtc_count = {name: 0 for name in confusion}
    per_sequence = {}

    for sequence, samples in groups.items():
        if not samples:
            continue
        sequence_confusion = {
            name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
            for name in confusion
        }
        vc = {
            "host": VideoConsistency(),
            "ours": VideoConsistency(),
        }
        sequence_mtc_sum = {name: 0.0 for name in confusion}
        sequence_mtc_count = {name: 0 for name in confusion}

        previous_image, observation, raw, output_size = encode_clean(
            model, samples[0]
        )
        mask = semantic_mask_from_panoptic_png(samples[0]["mask_path"])
        base_logits = host_logits(model, raw, output_size)
        base_prediction = base_logits.argmax(1)
        predictions = {
            "host": base_prediction,
            "ours": base_prediction,
        }
        for name, prediction in predictions.items():
            prediction_cpu = prediction[0].cpu()
            update_confusion_matrix(
                confusion[name], prediction_cpu, mask
            )
            update_confusion_matrix(
                sequence_confusion[name], prediction_cpu, mask
            )
            vc[name].update(mask, prediction_cpu)

        previous_predictions = {
            name: prediction.detach()
            for name, prediction in predictions.items()
        }
        pending_z4, h4 = z4_predict_next(
            predictor,
            observation.z4,
            torch.zeros_like(observation.z4),
            None,
        )
        semantic_hidden = predictor.initial_semantic_state(observation)
        error_stats = predictor.initial_error_temporal_statistics()

        for sample in samples[1:]:
            image, observation, raw, output_size = encode_clean(
                model, sample
            )
            mask = semantic_mask_from_panoptic_png(sample["mask_path"])
            base_logits = host_logits(model, raw, output_size)
            host_prediction = base_logits.argmax(1)

            prior = prediction_state(observation, pending_z4)
            restored, semantic_hidden, diagnostics = predictor.restore_current(
                observation,
                prior,
                semantic_hidden,
                error_temporal_state=error_stats,
            )
            error_stats = diagnostics["error_temporal_state"]
            ours_logits = corrected_logits(
                model, raw, observation, restored, output_size
            )
            ours_prediction = ours_logits.argmax(1)
            predictions = {
                "host": host_prediction,
                "ours": ours_prediction,
            }

            backward_flow = raft.backward_flow(image, previous_image)
            for name, prediction in predictions.items():
                prediction_cpu = prediction[0].cpu()
                update_confusion_matrix(
                    confusion[name], prediction_cpu, mask
                )
                update_confusion_matrix(
                    sequence_confusion[name], prediction_cpu, mask
                )
                vc[name].update(mask, prediction_cpu)
                mtc = pair_mtc(
                    previous_predictions[name],
                    prediction,
                    backward_flow,
                )
                if math.isfinite(mtc):
                    mtc_sum[name] += mtc
                    mtc_count[name] += 1
                    sequence_mtc_sum[name] += mtc
                    sequence_mtc_count[name] += 1

            prediction_error_z4 = observation.z4 - pending_z4
            pending_z4, h4 = z4_predict_next(
                predictor,
                observation.z4,
                prediction_error_z4,
                h4,
            )
            previous_image = image
            previous_predictions = {
                name: prediction.detach()
                for name, prediction in predictions.items()
            }

        sequence_row = {}
        for name in confusion:
            values = vc[name].values()
            sequence_row[name] = {
                "mIoU": float(
                    torch.nanmean(
                        compute_iou(sequence_confusion[name])
                    ).item()
                ),
                "mVC8": values[8],
                "mVC16": values[16],
                "mTC": (
                    sequence_mtc_sum[name]
                    / max(sequence_mtc_count[name], 1)
                ),
                "valid_frame_pairs": sequence_mtc_count[name],
            }
            stats = vc[name].stats()
            for length in (8, 16):
                mvc_sums[name][length] += stats[length]["sum"]
                mvc_counts[name][length] += stats[length]["count"]
        per_sequence[sequence] = sequence_row

    metrics = {}
    for name in confusion:
        metrics[name] = {
            "mIoU": float(
                torch.nanmean(compute_iou(confusion[name])).item()
            ),
            "mVC8": mvc_sums[name][8] / max(mvc_counts[name][8], 1),
            "mVC16": mvc_sums[name][16] / max(mvc_counts[name][16], 1),
            "mTC": mtc_sum[name] / max(mtc_count[name], 1),
            "valid_frame_pairs": mtc_count[name],
        }
    metrics["delta"] = {
        key: metrics["ours"][key] - metrics["host"][key]
        for key in ("mIoU", "mVC8", "mVC16", "mTC")
    }
    metrics["per_sequence"] = per_sequence
    return metrics


def gate_pass(metrics):
    delta = metrics["delta"]
    return bool(
        delta["mIoU"] >= 0.0
        and delta["mTC"] > 0.0
        and delta["mVC16"] >= MVC16_FLOOR_DELTA
    )


def progress_rank(metrics):
    delta = metrics["delta"]
    passed = gate_pass(metrics)
    if passed:
        return (
            1,
            delta["mTC"],
            delta["mIoU"],
            delta["mVC16"],
        )
    return (
        0,
        min(delta["mIoU"], 0.0),
        delta["mTC"],
        delta["mVC16"],
    )


def save_checkpoint(
    path,
    model,
    predictor,
    epoch,
    metrics,
    calibration,
    args,
):
    torch.save(
        {
            "experiment": "z4_only_stage_c",
            "epoch": epoch,
            "model_state_dict": predictor.state_dict(),
            "c4_output_adapter_state_dict":
                model.multi_layer_adapter.output_adapters[3].state_dict(),
            "c4_writeback_state_dict":
                model.host_conditioned_writebacks["3"].state_dict(),
            "stage_p_checkpoint": args.stage_p_checkpoint,
            "source_fast_b_checkpoint": args.fast_b_checkpoint,
            "loss_calibration": calibration,
            "metrics": metrics,
            "joint_c4_training": False,
            "z4_only": True,
        },
        path,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predify Z4-only Stage C semantic correction training."
    )
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument(
        "--fast-b-checkpoint", default=FAST_B_DEFAULT
    )
    parser.add_argument(
        "--stage-p-checkpoint", default=STAGE_P_DEFAULT
    )
    parser.add_argument(
        "--dynamics-checkpoint",
        default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    )
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--smoke-check-only", action="store_true")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("Stage C requires CUDA")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model, predictor, _, stage_p_payload = load_stage_c(args)
    raft = FrozenRAFT()

    train_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root), "train"
    )
    train_groups = sequence_groups(train_dataset)
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root), "val"
    )
    all_val_groups = sequence_groups(val_dataset)
    dev3 = {sequence: all_val_groups[sequence] for sequence in DEV3}

    calibration = calibrate_loss_weights(
        model,
        predictor,
        raft,
        next(iter(train_groups.values())),
    )
    print(
        json.dumps(
            {"stage_c_loss_calibration": calibration},
            sort_keys=True,
        ),
        flush=True,
    )
    if args.smoke_check_only:
        return

    initial = evaluate(model, predictor, dev3, raft)
    print(
        json.dumps(
            {"stage_c_initial_dev3": initial},
            sort_keys=True,
        ),
        flush=True,
    )

    parameters = semantic_parameters(predictor)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    history = []
    best_progress = None
    best_gate = None
    stale = 0

    for epoch in range(1, args.epochs + 1):
        train = train_epoch(
            model,
            predictor,
            raft,
            train_groups,
            optimizer,
            calibration["lambda_tc"],
            calibration["lambda_safe"],
        )
        metrics = evaluate(model, predictor, dev3, raft)
        record = {
            "epoch": epoch,
            "stage": "C",
            "train": train,
            "dev3": metrics,
            "gate_pass": gate_pass(metrics),
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

        save_checkpoint(
            output / f"epoch_{epoch:03d}.pt",
            model,
            predictor,
            epoch,
            metrics,
            calibration,
            args,
        )

        rank = progress_rank(metrics)
        if best_progress is None or rank > best_progress["rank"]:
            best_progress = {
                "epoch": epoch,
                "rank": rank,
                "metrics": metrics,
            }
            stale = 0
            save_checkpoint(
                output / "best_progress.pt",
                model,
                predictor,
                epoch,
                metrics,
                calibration,
                args,
            )
        else:
            stale += 1

        if gate_pass(metrics):
            gate_rank = (
                metrics["delta"]["mTC"],
                metrics["delta"]["mIoU"],
                metrics["delta"]["mVC16"],
            )
            if best_gate is None or gate_rank > best_gate["rank"]:
                best_gate = {
                    "epoch": epoch,
                    "rank": gate_rank,
                    "metrics": metrics,
                }
                save_checkpoint(
                    output / "best.pt",
                    model,
                    predictor,
                    epoch,
                    metrics,
                    calibration,
                    args,
                )

        if stale >= args.patience:
            break

    result = {
        "experiment": "Predify Z4-only Stage C",
        "stage_p_checkpoint": args.stage_p_checkpoint,
        "stage_p_epoch": stage_p_payload.get("epoch"),
        "source_fast_b_checkpoint": args.fast_b_checkpoint,
        "trainable_modules": [
            "semantic_error_encoder",
            "semantic_state_cell",
            "semantic_restoration_head",
        ],
        "frozen_modules": [
            "DeepLabV3+ Host",
            "Z1 predictor",
            "Z4 predictor",
            "C4 output adapter",
            "C4 host-conditioned writeback",
        ],
        "tbptt": TBPTT,
        "segmentation_supervision_positions": SEGMENTATION_POSITIONS,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "loss_calibration": calibration,
        "initial_dev3": initial,
        "history": history,
        "best_progress": best_progress,
        "best_gate": best_gate,
        "gate": {
            "passed": best_gate is not None,
            "requirements": {
                "delta_mIoU_gte": 0.0,
                "delta_mTC_gt": 0.0,
                "delta_mVC16_gte": MVC16_FLOOR_DELTA,
            },
        },
    }
    (result_output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (result_output / "README.md").write_text(
        "# Z4-only Stage C\n"
        "Stage C freezes the Stage-P Z4 predictor and trains only the semantic "
        "error encoder, semantic state cell and restoration head. RAFT is used "
        "only for training/evaluation temporal supervision. The C4 interface "
        "and DeepLabV3+ Host remain frozen.\n"
    )


if __name__ == "__main__":
    main()
