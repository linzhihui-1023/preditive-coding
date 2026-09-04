"""Temporal Joint Training V1 for the existing FAST-B architecture.

This experiment keeps the model information flow unchanged.  It trains the
Semantic V3 branch, C4 adapter/writeback, and the previously trained dynamics
branch on ordered clean KITTI-STEP clips.  RAFT is privileged supervision only:
it is frozen, used to build a differentiable probability warp during training,
and is not part of the saved predictor or inference path.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

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
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import (
    load_writeback_checkpoint,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    error_state,
    load_components,
    residual_writeback_host_feature,
    zero_state,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorRegulatedSemanticRestorationPredictor,
    HostFeature,
    UnifiedFeatures,
)


SEED = 0
EPOCHS = 3
TBPTT_STEPS = 8
LR_SEMANTIC = 1e-4
LR_DYNAMICS = 1e-5
WEIGHT_DECAY = 0.01
LAMBDA_SEGMENTATION = 3e-4
LAMBDA_PREDICTION = 1.0
LAMBDA_Z1_PREDICTION = 0.25
LAMBDA_DELTA = 1e-5
LAMBDA_TC_VALUES = (1e-4, 5e-4, 1e-3, 2e-3, 5e-3)
DEV3_SEQUENCES = ("0002", "0010", "0018")
# Host dev3 metrics were computed once with the same clean frame-wise protocol
# before the causal comparison.  Reuse them for A/B/C instead of rerunning the
# Host baseline for every diagnostic cell.
HOST_DEV3_REFERENCE = {
    "mIoU": 0.5799865130187902,
    "mTC": 0.7009239766436142,
    "mVC8": 0.9345485965278791,
    "mVC16": 0.9273142366723673,
}
NUM_CLASSES = 19
IGNORE_LABEL = 255
C4_ADAPTER_INDEX = 3
C4_WRITEBACK_KEY = "3"
FAST_B_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_semantic_v3_joint_c4_fast_ab_5754714/"
    "fast_b_joint_c4_weak_z4/best.pt"
)
OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_temporal_joint_v1"
RESULT_DEFAULT = "results/kitti_step_temporal_joint_v1"


class FrozenRAFT:
    def __init__(self):
        self.weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.weights, progress=True).cuda().eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def current_to_previous(self, current, previous):
        height, width = current.shape[-2:]
        pad_height = (-height) % 8
        pad_width = (-width) % 8
        current = F.pad(current, (0, pad_width, 0, pad_height), mode="replicate")
        previous = F.pad(previous, (0, pad_width, 0, pad_height), mode="replicate")
        current, previous = self.weights.transforms()(current, previous)
        return self.model(current, previous)[-1][..., :height, :width]


def configure_trainable_path(model, predictor, unfreeze_dynamics=True):
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    for module in (
        predictor.semantic_error_encoder,
        predictor.semantic_state_cell,
        predictor.semantic_restoration_head,
    ):
        module.requires_grad_(True)
    if unfreeze_dynamics:
        for name in predictor.DYNAMICS_MODULES:
            getattr(predictor, name).requires_grad_(True)
    model.multi_layer_adapter.output_adapters[C4_ADAPTER_INDEX].requires_grad_(True)
    model.host_conditioned_writebacks[C4_WRITEBACK_KEY].requires_grad_(True)
    model.eval()
    predictor.train()


def trainable_parameters(model, predictor):
    semantic = [p for p in predictor.semantic_parameters() if p.requires_grad]
    dynamics = [
        p
        for name in predictor.DYNAMICS_MODULES
        for p in getattr(predictor, name).parameters()
        if p.requires_grad
    ]
    c4 = [
        p
        for module in (
            model.multi_layer_adapter.output_adapters[C4_ADAPTER_INDEX],
            model.host_conditioned_writebacks[C4_WRITEBACK_KEY],
        )
        for p in module.parameters()
        if p.requires_grad
    ]
    return semantic, dynamics, c4


def assert_freeze_contract(model, predictor, unfreeze_dynamics=True):
    frozen_modules = {
        "backbone": model.backbone,
        "decoder": model.decode_head,
        "c1_output_adapter": model.multi_layer_adapter.output_adapters[0],
        "c1_writeback": model.host_conditioned_writebacks["0"],
    }
    bad = [
        name
        for name, module in frozen_modules.items()
        if any(parameter.requires_grad for parameter in module.parameters())
    ]
    if bad:
        raise RuntimeError(f"Frozen-module contract violated: {bad}")
    dynamics = [
        parameter
        for name in predictor.DYNAMICS_MODULES
        for parameter in getattr(predictor, name).parameters()
    ]
    if unfreeze_dynamics and (not dynamics or not all(parameter.requires_grad for parameter in dynamics)):
        raise RuntimeError("Dynamics branch must be trainable in Temporal Joint V1")
    if not unfreeze_dynamics and any(parameter.requires_grad for parameter in dynamics):
        raise RuntimeError("Dynamics branch must be frozen for this diagnostic cell")
    if not all(
        parameter.requires_grad
        for module in (
            predictor.semantic_error_encoder,
            predictor.semantic_state_cell,
            predictor.semantic_restoration_head,
            model.multi_layer_adapter.output_adapters[3],
            model.host_conditioned_writebacks["3"],
        )
        for parameter in module.parameters()
    ):
        raise RuntimeError("Semantic/C4 trainable path is incomplete")


def load_training_components(args):
    model, source = load_components(
        STATIC_CHECKPOINT_DEFAULT,
        ADAPTER_CHECKPOINT_DEFAULT,
        args.dynamics_checkpoint,
        WRITEBACK_CHECKPOINT_DEFAULT,
    )
    fast_payload = torch.load(
        args.fast_b_checkpoint, map_location="cpu", weights_only=False
    )
    model.multi_layer_adapter.output_adapters[3].load_state_dict(
        fast_payload["c4_output_adapter_state_dict"], strict=True
    )
    model.host_conditioned_writebacks["3"].load_state_dict(
        fast_payload["c4_writeback_state_dict"], strict=True
    )
    predictor = ErrorRegulatedSemanticRestorationPredictor(
        use_error_temporal_stats=True
    ).cuda()
    predictor.load_state_dict(fast_payload["model_state_dict"], strict=True)
    # The FAST-B payload is authoritative for the complete initial predictor;
    # source is only loaded by load_components to preserve the existing chain.
    del source
    configure_trainable_path(model, predictor, args.dynamics_mode == "unfrozen")
    assert_freeze_contract(model, predictor, args.dynamics_mode == "unfrozen")
    return model, predictor, fast_payload


def encode_clean(model, sample):
    image = load_image(sample)
    # Backbone is frozen, but the C4 input adapter remains part of the
    # observation graph so its gradients can reach the trainable C4 path.
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
    observation = model.encode_backbone_features(raw)
    return image, observation, raw, tuple(image.shape[-2:])


def decode_records(model, records):
    host_features = []
    masks = []
    for raw, observation, restored, output_size, mask in records:
        zero = zero_state(observation)
        delta = UnifiedFeatures(
            zero.z1,
            zero.z2,
            zero.z3,
            restored.z4 - observation.z4,
        )
        host_features.append(residual_writeback_host_feature(model, raw, delta, output_size))
        masks.append(mask)
    batched = HostFeature(
        torch.cat([feature.tensor for feature in host_features], dim=0),
        torch.cat([feature.low_level for feature in host_features], dim=0),
        host_features[0].output_size,
    )
    # Decoder parameters are frozen, but its input still needs gradients.  A
    # non-reentrant checkpoint keeps the TBPTT graph while avoiding retention of
    # the large DeepLab decoder activations for every frame in the window.
    logits = checkpoint(
        lambda tensor: model.decode_from_host_feature(
            HostFeature(tensor, batched.low_level, batched.output_size)
        ),
        batched.tensor,
        use_reentrant=False,
    )
    return logits, torch.stack(masks).to(batched.tensor.device)


def flow_grid(backward_flow, height, width):
    flow = F.interpolate(
        backward_flow, size=(height, width), mode="bilinear", align_corners=True
    )
    flow_x = flow[:, 0] * (width / backward_flow.shape[-1])
    flow_y = flow[:, 1] * (height / backward_flow.shape[-2])
    y, x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    source_x = x.unsqueeze(0) + flow_x
    source_y = y.unsqueeze(0) + flow_y
    valid = (
        (source_x >= 0)
        & (source_x <= width - 1)
        & (source_y >= 0)
        & (source_y <= height - 1)
    )
    grid = torch.stack(
        (
            2.0 * source_x / (width - 1) - 1.0,
            2.0 * source_y / (height - 1) - 1.0,
        ),
        dim=-1,
    )
    return grid, valid


def temporal_kl(current_logits, previous_logits, backward_flow, previous_mask, current_mask):
    height, width = current_logits.shape[-2:]
    grid, flow_valid = flow_grid(backward_flow, height, width)
    current_prob = current_logits.softmax(dim=1)
    previous_prob = previous_logits.detach().softmax(dim=1)
    warped_prob = F.grid_sample(
        previous_prob,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).detach()
    previous_mask = previous_mask.to(current_logits.device, dtype=torch.float32)
    current_mask = current_mask.to(current_logits.device, dtype=torch.int64)
    warped_mask = F.grid_sample(
        previous_mask.unsqueeze(0).unsqueeze(0),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0).squeeze(0).to(torch.int64)
    valid = flow_valid.squeeze(0)
    valid &= previous_mask.squeeze(0).to(torch.int64) != IGNORE_LABEL
    valid &= current_mask != IGNORE_LABEL
    valid &= warped_mask == current_mask
    if not bool(valid.any()):
        return current_logits.sum() * 0.0
    per_pixel = F.kl_div(
        current_prob.clamp_min(1e-8).log(),
        warped_prob.clamp_min(1e-8),
        reduction="none",
    ).sum(dim=1)
    return per_pixel[0][valid].mean()


def correction_magnitude(restored, observation):
    return (restored.z4 - observation.z4).abs().mean()


def detach_state(state):
    return UnifiedFeatures(*(value.detach() for value in state.as_tuple()))


def train_sequence(model, predictor, samples, optimizer, raft, args, max_steps=0):
    if len(samples) < 2:
        return {"steps": 0, "windows": 0, "segmentation_loss": 0.0, "prediction_loss": 0.0, "temporal_loss": 0.0, "delta_loss": 0.0, "total_loss": 0.0}
    steps = min(len(samples) - 1, max_steps) if max_steps else len(samples) - 1
    previous_image, first_observation, first_raw, output_size = encode_clean(model, samples[0])
    pending, h4, h1 = predictor.predict_next(
        first_observation, zero_state(first_observation), None, None
    )
    semantic_hidden = predictor.initial_semantic_state(first_observation)
    error_stats = predictor.initial_error_temporal_statistics()
    with torch.no_grad():
        first_host_logits = model.decode_from_host_feature(
            HostFeature(first_raw.c4, first_raw.c1, output_size)
        )
    previous_logits = first_host_logits.detach()
    previous_mask = semantic_mask_from_panoptic_png(samples[0]["mask_path"]).cuda()
    records = []
    prediction_window_losses = []
    totals = {"steps": 0, "windows": 0, "segmentation_loss": 0.0, "prediction_loss": 0.0, "temporal_loss": 0.0, "delta_loss": 0.0, "total_loss": 0.0}
    for offset in range(steps):
        frame_index = offset + 1
        image, observation, raw, output_size = encode_clean(model, samples[frame_index])
        prediction_error = error_state(observation, pending)
        restored, semantic_hidden, diagnostics = predictor.restore_current(
            observation,
            pending,
            semantic_hidden,
            error_temporal_state=error_stats,
        )
        error_stats = diagnostics["error_temporal_state"]
        prediction_loss = F.smooth_l1_loss(pending.z4, observation.z4.detach()) + LAMBDA_Z1_PREDICTION * F.smooth_l1_loss(pending.z1, observation.z1.detach())
        prediction_window_losses.append(prediction_loss)
        mask = semantic_mask_from_panoptic_png(samples[frame_index]["mask_path"]).cuda()
        with torch.no_grad():
            flow = raft.current_to_previous(image, previous_image) if args.lambda_tc > 0 else None
        records.append((raw, observation, restored, output_size, mask, flow, previous_mask, previous_logits))
        previous_image = image
        previous_mask = mask
        previous_logits = None
        pending, h4, h1 = predictor.predict_next(
            observation, prediction_error, h4, h1
        )
        totals["steps"] += 1
        window_end = len(records) == args.tbptt_steps or frame_index == steps
        if not window_end:
            continue

        decode_input = [record[:5] for record in records]
        logits, masks = decode_records(model, decode_input)
        segmentation_loss = F.cross_entropy(logits[-1:], masks[-1:], ignore_index=IGNORE_LABEL)
        if args.lambda_tc > 0:
            temporal_losses = []
            for index, record in enumerate(records):
                previous_for_loss = record[7] if index == 0 else logits[index - 1:index]
                temporal_losses.append(
                    temporal_kl(logits[index:index + 1], previous_for_loss, record[5], record[6], record[4])
                )
            temporal_loss = torch.stack(temporal_losses).mean()
        else:
            temporal_loss = logits.sum() * 0.0
        prediction_loss_window = torch.stack(prediction_window_losses).mean()
        delta_loss = torch.stack([
            correction_magnitude(record[2], record[1]) for record in records
        ]).mean()
        total_loss = (
            LAMBDA_SEGMENTATION * segmentation_loss
            + LAMBDA_PREDICTION * prediction_loss_window
            + args.lambda_tc * temporal_loss
            + LAMBDA_DELTA * delta_loss
        )
        if not all(torch.isfinite(value).item() for value in (segmentation_loss, prediction_loss_window, temporal_loss, delta_loss, total_loss)):
            raise FloatingPointError("Non-finite Temporal Joint V1 loss")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        totals["windows"] += 1
        for name, value in (("segmentation_loss", segmentation_loss), ("prediction_loss", prediction_loss_window), ("temporal_loss", temporal_loss), ("delta_loss", delta_loss), ("total_loss", total_loss)):
            totals[name] += float(value.detach().item())
        semantic_hidden = semantic_hidden.detach()
        h4, h1 = h4.detach(), h1.detach()
        pending = detach_state(pending)
        error_stats = error_stats.detach() if error_stats is not None else None
        previous_logits = logits[-1:].detach()
        records = []
        prediction_window_losses = []
        # Rebuild the next-frame prediction from detached boundary inputs so
        # the first prediction loss in the next TBPTT window still trains the
        # Dynamics branch without retaining the prior optimizer-step graph.
        if frame_index < steps:
            pending, h4, h1 = predictor.predict_next(
                detach_state(observation),
                detach_state(prediction_error),
                h4,
                h1,
            )
    return totals


def _train_sequence_with_prediction_capture(model, predictor, samples, optimizer, raft, args, max_steps=0):
    return train_sequence(model, predictor, samples, optimizer, raft, args, max_steps)


def evaluate_clean(model, predictor, groups, raft, include_host=True):
    names = ("host", "ours") if include_host else ("ours",)
    confusion = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in names}
    vc = {name: {} for name in names}
    mtc_sum = {name: 0.0 for name in names}
    mtc_count = 0
    mvc_sums = {name: {8: 0.0, 16: 0.0} for name in names}
    mvc_counts = {name: {8: 0, 16: 0} for name in names}
    per_sequence = {}
    with torch.inference_mode():
        for sequence, samples in groups.items():
            if len(samples) < 2:
                continue
            host_vc, ours_vc = VideoConsistency(), VideoConsistency()
            seq_confusion = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in names}
            seq_mtc = {name: 0.0 for name in names}
            seq_pairs = 0
            image, observation, raw, output_size = encode_clean(model, samples[0])
            mask = semantic_mask_from_panoptic_png(samples[0]["mask_path"])
            host_prediction = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size)).argmax(1)
            previous_host = previous_ours = host_prediction
            if include_host:
                update_confusion_matrix(confusion["host"], host_prediction.squeeze(0).cpu(), mask)
                host_vc.update(mask, host_prediction)
                update_confusion_matrix(seq_confusion["host"], host_prediction.squeeze(0).cpu(), mask)
            ours_vc.update(mask, host_prediction)
            update_confusion_matrix(confusion["ours"], host_prediction.squeeze(0).cpu(), mask)
            update_confusion_matrix(seq_confusion["ours"], host_prediction.squeeze(0).cpu(), mask)
            pending, h4, h1 = predictor.predict_next(observation, zero_state(observation), None, None)
            semantic_hidden = predictor.initial_semantic_state(observation)
            error_stats = predictor.initial_error_temporal_statistics()
            previous_image = image
            for sample in samples[1:]:
                image, observation, raw, output_size = encode_clean(model, sample)
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                if include_host:
                    host_prediction = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size)).argmax(1)
                prediction_error = error_state(observation, pending)
                restored, semantic_hidden, diagnostics = predictor.restore_current(observation, pending, semantic_hidden, error_temporal_state=error_stats)
                error_stats = diagnostics["error_temporal_state"]
                ours_prediction = model.decode_from_host_feature(residual_writeback_host_feature(model, raw, UnifiedFeatures(torch.zeros_like(observation.z1), torch.zeros_like(observation.z2), torch.zeros_like(observation.z3), restored.z4 - observation.z4), output_size)).argmax(1)
                prediction_trackers = (("host", host_prediction, host_vc), ("ours", ours_prediction, ours_vc)) if include_host else (("ours", ours_prediction, ours_vc),)
                for name, prediction, tracker in prediction_trackers:
                    update_confusion_matrix(confusion[name], prediction.squeeze(0).cpu(), mask)
                    update_confusion_matrix(seq_confusion[name], prediction.squeeze(0).cpu(), mask)
                    tracker.update(mask, prediction)
                flow = raft.current_to_previous(image, previous_image)
                # Evaluation mTC uses nearest-neighbor class-ID warping; reuse
                # the probability warp grid with hard labels for consistency.
                tc_predictions = (("host", previous_host, host_prediction), ("ours", previous_ours, ours_prediction)) if include_host else (("ours", previous_ours, ours_prediction),)
                for name, previous_prediction, current_prediction in tc_predictions:
                    grid, valid = flow_grid(flow, current_prediction.shape[-2], current_prediction.shape[-1])
                    warped = F.grid_sample(previous_prediction.float().unsqueeze(0), grid, mode="nearest", padding_mode="zeros", align_corners=True).squeeze(0).squeeze(0).to(torch.int64)
                    keep = valid.squeeze(0)
                    a, b = warped[keep].cpu(), current_prediction.squeeze(0)[keep].cpu()
                    pair_confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
                    if a.numel():
                        pair_confusion += torch.bincount(NUM_CLASSES * a + b, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
                    score = float(torch.nanmean(compute_iou(pair_confusion)).item())
                    if math.isfinite(score):
                        mtc_sum[name] += score; seq_mtc[name] += score
                mtc_count += 1; seq_pairs += 1
                previous_image, previous_ours = image, ours_prediction
                if include_host:
                    previous_host = host_prediction
                pending, h4, h1 = predictor.predict_next(observation, prediction_error, h4, h1)
            trackers = (("host", host_vc), ("ours", ours_vc)) if include_host else (("ours", ours_vc),)
            for name, tracker in trackers:
                stats = tracker.stats(); values = tracker.values()
                for length in (8, 16):
                    mvc_sums[name][length] += stats[length]["sum"]
                    mvc_counts[name][length] += stats[length]["count"]
                iou = compute_iou(seq_confusion[name])
                per_sequence.setdefault(sequence, {})[name] = {
                    "mIoU": float(torch.nanmean(iou).item()),
                    "mVC8": values[8], "mVC16": values[16],
                    "mTC": seq_mtc[name] / seq_pairs if seq_pairs else float("nan"),
                }
            per_sequence[sequence]["valid_frame_pairs"] = seq_pairs
    metrics = {}
    for name in names:
        iou = compute_iou(confusion[name])
        metrics[name] = {
            "mIoU": float(torch.nanmean(iou).item()),
            "mVC8": mvc_sums[name][8] / mvc_counts[name][8] if mvc_counts[name][8] else float("nan"),
            "mVC16": mvc_sums[name][16] / mvc_counts[name][16] if mvc_counts[name][16] else float("nan"),
            "mTC": mtc_sum[name] / mtc_count if mtc_count else float("nan"),
        }
    metrics["valid_frame_pairs"] = mtc_count
    metrics["per_sequence"] = per_sequence
    return metrics


def gradient_smoke(model, predictor, train_samples, args):
    semantic, dynamics, c4 = trainable_parameters(model, predictor)
    optimizer = torch.optim.AdamW(
        [{"params": semantic + c4, "lr": LR_SEMANTIC}, {"params": dynamics, "lr": LR_DYNAMICS}],
        weight_decay=WEIGHT_DECAY,
    )
    raft = FrozenRAFT()
    # A short ordered clip exercises both the segmentation/temporal graph and
    # the detached TBPTT state boundary.
    row = train_sequence(model, predictor, train_samples[: min(len(train_samples), 10)], optimizer, raft, args)
    def grad_norm(modules):
        values = [p.grad.detach().norm() for module in modules for p in module.parameters() if p.grad is not None]
        return float(torch.stack(values).norm().item()) if values else 0.0
    result = {
        "semantic_grad_norm": grad_norm((predictor.semantic_error_encoder, predictor.semantic_state_cell, predictor.semantic_restoration_head)),
        "dynamics_grad_norm": grad_norm(tuple(getattr(predictor, name) for name in predictor.DYNAMICS_MODULES)),
        "c4_grad_norm": grad_norm((model.multi_layer_adapter.output_adapters[3], model.host_conditioned_writebacks["3"])),
        "backbone_grad_norm": grad_norm((model.backbone,)),
        "decoder_grad_norm": grad_norm((model.decode_head,)),
        "c1_grad_norm": grad_norm((model.multi_layer_adapter.output_adapters[0], model.host_conditioned_writebacks["0"])),
        "loss_finite": all(math.isfinite(row[key]) for key in ("segmentation_loss", "prediction_loss", "temporal_loss", "delta_loss", "total_loss")),
    }
    dynamics_expected = any(parameter.requires_grad for name in predictor.DYNAMICS_MODULES for parameter in getattr(predictor, name).parameters())
    if result["semantic_grad_norm"] <= 0 or result["c4_grad_norm"] <= 0 or (dynamics_expected and result["dynamics_grad_norm"] <= 0) or (not dynamics_expected and result["dynamics_grad_norm"] != 0.0):
        raise RuntimeError(f"Temporal Joint V1 smoke gradient failed: {result}")
    if any(result[key] != 0.0 for key in ("backbone_grad_norm", "decoder_grad_norm", "c1_grad_norm")):
        raise RuntimeError(f"Frozen branch received gradient: {result}")
    model.zero_grad(set_to_none=True); predictor.zero_grad(set_to_none=True)
    return result


def checkpoint_payload(model, predictor, args, epoch, lambda_tc, metrics):
    return {
        "experiment": "kitti_step_temporal_joint_v1",
        "model_state_dict": predictor.state_dict(),
        "c4_output_adapter_state_dict": model.multi_layer_adapter.output_adapters[3].state_dict(),
        "c4_writeback_state_dict": model.host_conditioned_writebacks["3"].state_dict(),
        "source_fast_b_checkpoint": args.fast_b_checkpoint,
        "source_dynamics_checkpoint": args.dynamics_checkpoint,
        "epoch": epoch,
        "use_error_temporal_stats": True,
        "joint_c4_training": True,
        "temporal_joint_training": True,
        "lambda_tc": lambda_tc,
        "metrics": metrics,
    }


def run_lambda(args, lambda_tc, train_groups, val_groups, host_reference):
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model, predictor, _ = load_training_components(args)
    args.lambda_tc = lambda_tc
    semantic, dynamics, c4 = trainable_parameters(model, predictor)
    optimizer = torch.optim.AdamW(
        [{"params": semantic + c4, "lr": LR_SEMANTIC}, {"params": dynamics, "lr": LR_DYNAMICS}],
        weight_decay=args.weight_decay,
    )
    raft = FrozenRAFT()
    output = Path(args.output) / f"lambda_tc_{lambda_tc:g}"
    output.mkdir(parents=True, exist_ok=True)
    history = []; best = None
    for epoch in range(1, args.epochs + 1):
        train_totals = {"steps": 0, "windows": 0, "segmentation_loss": 0.0, "prediction_loss": 0.0, "temporal_loss": 0.0, "delta_loss": 0.0, "total_loss": 0.0}
        for samples in train_groups.values():
            row = _train_sequence_with_prediction_capture(model, predictor, samples, optimizer, raft, args)
            for key in train_totals: train_totals[key] += row[key]
        for key in ("segmentation_loss", "prediction_loss", "temporal_loss", "delta_loss", "total_loss"):
            train_totals[key] /= max(train_totals["windows"], 1)
        metrics = evaluate_clean(model, predictor, val_groups, raft, include_host=False)
        record = {"epoch": epoch, "lambda_tc": lambda_tc, "train": train_totals, "val": metrics}
        history.append(record); print(json.dumps(record, sort_keys=True), flush=True)
        torch.save(checkpoint_payload(model, predictor, args, epoch, lambda_tc, metrics), output / f"epoch_{epoch:03d}.pt")
        ours = metrics["ours"]
        eligible = ours["mIoU"] >= host_reference["mIoU"] - 0.002
        key = (ours["mTC"], ours["mVC16"], ours["mVC8"], ours["mIoU"])
        if eligible and (best is None or key > best["selection_key"]):
            best = {"epoch": epoch, "selection_key": key, "metrics": metrics, "train": train_totals}
            torch.save(checkpoint_payload(model, predictor, args, epoch, lambda_tc, metrics), output / "best.pt")
    summary = {
        "experiment": "kitti_step_temporal_joint_v1",
        "lambda_tc": lambda_tc,
        "epochs": args.epochs,
        "tbptt_steps": args.tbptt_steps,
        "lambda_segmentation": LAMBDA_SEGMENTATION,
        "lambda_prediction": LAMBDA_PREDICTION,
        "lambda_z1_prediction": LAMBDA_Z1_PREDICTION,
        "lambda_delta": LAMBDA_DELTA,
        "lr_semantic": LR_SEMANTIC,
        "lr_dynamics": LR_DYNAMICS,
        "trainable_parameter_count": sum(p.numel() for p in semantic + dynamics + c4),
        "dynamics_trainable_parameter_count": sum(p.numel() for p in dynamics),
        "history": history,
        "best": best,
        "host_dev3_reference": host_reference,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--dynamics-checkpoint", default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--tbptt-steps", type=int, default=TBPTT_STEPS)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lambda-tc", type=float, default=None)
    parser.add_argument("--dynamics-mode", choices=("frozen", "unfrozen"), default="unfrozen")
    parser.add_argument("--smoke-check-only", action="store_true")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    train_groups = sequence_groups(dataset)
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    val_all = sequence_groups(val_dataset)
    val_groups = {key: val_all[key] for key in DEV3_SEQUENCES}
    model, predictor, _ = load_training_components(args)
    smoke_args = argparse.Namespace(**vars(args))
    smoke_args.lambda_tc = args.lambda_tc if args.lambda_tc is not None else 1e-3
    smoke = gradient_smoke(model, predictor, next(iter(train_groups.values())), smoke_args)
    if args.smoke_check_only:
        print(json.dumps({"smoke_check": smoke}, sort_keys=True)); return
    lambdas = (args.lambda_tc,) if args.lambda_tc is not None else LAMBDA_TC_VALUES
    host_reference = dict(HOST_DEV3_REFERENCE)
    results = {}
    for lambda_tc in lambdas:
        results[f"{lambda_tc:g}"] = run_lambda(args, lambda_tc, train_groups, val_groups, host_reference)
    combined = {"experiment": "kitti_step_temporal_joint_v1", "host_dev3_reference": host_reference, "lambda_results": results, "smoke_check": smoke}
    result_output = Path(args.result_output); result_output.mkdir(parents=True, exist_ok=True)
    (result_output / "search_summary.json").write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result_output": str(result_output / 'search_summary.json'), "lambdas": list(results)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
