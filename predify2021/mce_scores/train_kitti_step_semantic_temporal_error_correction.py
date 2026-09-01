import json
import os
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    pil_rgb_to_unit_tensor,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.role_separated_direct_state_correction import (
    error_state,
    load_image,
    load_role_components,
    make_paths,
    next_role_prediction,
    zero_state,
)
from predify2021.mce_scores.kitti_step_persistent_blur import (
    BLUR_KERNEL_SIZE,
    BLUR_SIGMA_LEVELS,
    BLUR_SIGMA_MAX,
    BLUR_WARMUP_FRACTION,
    persistent_gaussian_blur,
    warmup_frame_count,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    residual_writeback_host_feature,
)
from predify2021.mce_scores.semantic_temporal_error_step import (
    DYNAMIC_ERROR_GAIN,
    DYNAMIC_ERROR_SAMPLE_TIME,
    DYNAMIC_ERROR_TIME_CONSTANT,
    detach_error_state,
    semantic_temporal_error_step,
    zero_semantic_temporal_state,
)
from predify2021.mce_scores.video_metrics import VideoConsistency, weighted_iou
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    BackboneFeatures,
    HostFeature,
    UnifiedFeatures,
)
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import (
    build_semantic_temporal_corrections,
)


SEED = int(os.environ.get("PREDIFY_SEED", "0"))
EPOCHS = int(os.environ.get("PREDIFY_EPOCHS", "15"))
EARLY_STOPPING_PATIENCE = int(
    os.environ.get("PREDIFY_EARLY_STOPPING_PATIENCE", "3")
)
EARLY_STOPPING_MIN_MIOU_IMPROVEMENT = 1e-4
MIOU_TIE_TOLERANCE = 1e-6
TRUNCATED_BPTT = int(os.environ.get("PREDIFY_TRUNCATED_BPTT", "4"))
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
DISTILL_WEIGHT = 0.5
TEMPORAL_PREDICTION_LOSS_WEIGHT = float(
    os.environ.get("PREDIFY_TEMPORAL_PREDICTION_LOSS_WEIGHT", "0.1")
)
DYNAMIC_DIAGNOSTICS = (
    "historical_dynamic_component",
    "first_order_model_residual",
    "gate_modulation",
    "update_gate_delta",
    "reset_gate_delta",
)
GATE_DIAGNOSTICS = (
    "mean_update_gate",
    "mean_reset_gate",
    "gate_saturation_rate",
    "corr_abs_residual_update_delta",
    "corr_abs_residual_reset_delta",
    "mean_abs_dynamic_error",
)
FROZEN_ENCODE_BATCH_SIZE = int(
    os.environ.get("PREDIFY_FROZEN_ENCODE_BATCH_SIZE", "8")
)
LOADER_WORKERS = int(os.environ.get("PREDIFY_LOADER_WORKERS", "8"))
LOADER_PREFETCH_FACTOR = int(
    os.environ.get("PREDIFY_LOADER_PREFETCH_FACTOR", "2")
)
TRAIN_SEQUENCE_LIMIT = int(os.environ.get("PREDIFY_TRAIN_SEQUENCE_LIMIT", "0"))
VAL_SEQUENCE_LIMIT = int(os.environ.get("PREDIFY_VAL_SEQUENCE_LIMIT", "0"))
FRAMES_PER_SEQUENCE_LIMIT = int(
    os.environ.get("PREDIFY_FRAMES_PER_SEQUENCE_LIMIT", "0")
)


def parameter_efficiency_metrics(model, predictor, corrections):
    modules = (model, predictor, corrections)
    total_params = sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
    )
    trainable_params = sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    state_adapter_params = sum(
        parameter.numel() for parameter in model.multi_layer_adapter.parameters()
    )
    writeback_params = sum(
        parameter.numel()
        for parameter in model.host_conditioned_writebacks.parameters()
    )
    predictor_params = sum(parameter.numel() for parameter in predictor.parameters())
    correction_params = sum(parameter.numel() for parameter in corrections.parameters())
    additional_inference_params = (
        state_adapter_params
        + writeback_params
        + predictor_params
        + correction_params
    )
    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_params / total_params,
        "trainable_ratio_percent": 100.0 * trainable_params / total_params,
        "additional_inference_params": additional_inference_params,
        "additional_inference_components": {
            "state_adapter": state_adapter_params,
            "host_conditioned_writeback": writeback_params,
            "predictor": predictor_params,
            "correction": correction_params,
        },
    }


def corrected_host_feature(model, raw_features, observation, posterior, output_size):
    delta = UnifiedFeatures(
        posterior.z1 - observation.z1,
        torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3),
        posterior.z4 - observation.z4,
    )
    return residual_writeback_host_feature(model, raw_features, delta, output_size)


def absolute_pearson_correlation(left, right):
    left = left.detach().abs().float().reshape(-1)
    right = right.detach().abs().float().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if denominator <= 1e-12:
        return left.new_zeros(())
    return (left * right).sum() / denominator


class SequenceChunkDataset(Dataset):
    """Ordered per-sequence chunks for parallel CPU image loading."""

    def __init__(self, groups, chunk_size):
        self.chunks = []
        for sequence_id, samples in groups.items():
            for start in range(0, len(samples), chunk_size):
                self.chunks.append((sequence_id, start, samples[start : start + chunk_size], len(samples)))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        sequence_id, start, samples, total_frames = self.chunks[index]
        images = []
        masks = []
        for sample in samples:
            image = Image.open(sample["image_path"]).convert("RGB")
            images.append(pil_rgb_to_unit_tensor(image))
            masks.append(semantic_mask_from_panoptic_png(sample["mask_path"]))
        return {
            "images": torch.stack(images),
            "masks": torch.stack(masks),
            "sequence_id": sequence_id,
            "start": start,
            "total_frames": total_frames,
        }


def collate_sequence_chunk(chunks):
    return chunks[0]


def make_sequence_chunk_loader(groups):
    dataset = SequenceChunkDataset(groups, FROZEN_ENCODE_BATCH_SIZE)
    loader_options = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": False,
        "num_workers": LOADER_WORKERS,
        "pin_memory": True,
        "collate_fn": collate_sequence_chunk,
    }
    if LOADER_WORKERS > 0:
        loader_options.update(
            persistent_workers=True,
            prefetch_factor=LOADER_PREFETCH_FACTOR,
        )
    return DataLoader(
        **loader_options,
    )


def limit_sequence_groups(groups, sequence_limit=0, frame_limit=0):
    items = list(groups.items())
    if sequence_limit > 0:
        items = items[:sequence_limit]
    return {
        sequence_id: samples[:frame_limit] if frame_limit > 0 else samples
        for sequence_id, samples in items
    }


def stack_backbone_features(features):
    return BackboneFeatures(*(torch.cat([getattr(item, name) for item in features], dim=0) for name in ("c1", "c2", "c3", "c4")))


def split_backbone_features(features, size):
    return tuple(BackboneFeatures(*(getattr(features, name).split(size, dim=0)[index] for name in ("c1", "c2", "c3", "c4"))) for index in range(2))


def select_backbone_features(features, index):
    return BackboneFeatures(*(getattr(features, name)[index : index + 1] for name in ("c1", "c2", "c3", "c4")))


def stack_unified_features(features):
    return UnifiedFeatures(*(torch.cat([getattr(item, name) for item in features], dim=0) for name in ("z1", "z2", "z3", "z4")))


def split_unified_features(features, size):
    return tuple(UnifiedFeatures(*(getattr(features, name).split(size, dim=0)[index] for name in ("z1", "z2", "z3", "z4"))) for index in range(2))


def select_unified_features(features, index):
    return UnifiedFeatures(*(getattr(features, name)[index : index + 1] for name in ("z1", "z2", "z3", "z4")))


def encode_frozen_sequence_chunk(model, clean_images, frame_indices, total_frames):
    corrupted_images = torch.cat(
        [
            persistent_gaussian_blur(clean_images[index : index + 1], frame_index, total_frames)
            for index, frame_index in enumerate(frame_indices)
        ],
        dim=0,
    )
    packed_images = torch.cat((clean_images, corrupted_images), dim=0)
    with torch.no_grad():
        packed_raw = model.extract_backbone_features(packed_images)
        packed_states = model.encode_backbone_features(packed_raw)
        clean_logits = model.decode_from_host_feature(
            HostFeature(packed_raw.c4[: len(clean_images)], packed_raw.c1[: len(clean_images)], tuple(clean_images.shape[-2:]))
        )
    clean_raw, corrupted_raw = split_backbone_features(packed_raw, len(clean_images))
    clean_state, observation = split_unified_features(packed_states, len(clean_images))
    return clean_raw, corrupted_raw, clean_state, observation, clean_logits


def semantic_losses_batch(model, raw_features, observation, posterior, clean_logits, masks, output_size):
    host_feature = corrected_host_feature(model, raw_features, observation, posterior, output_size)
    logits = model.decode_from_host_feature(host_feature)
    semantic = F.cross_entropy(logits, masks.cuda(non_blocking=True), ignore_index=255)
    teacher_probability = F.softmax(clean_logits.detach(), dim=1)
    distillation = F.kl_div(
        F.log_softmax(logits, dim=1), teacher_probability, reduction="none"
    ).sum(dim=1).mean()
    return semantic, distillation, logits


def flush_bptt_batch(
    model,
    records,
    output_size,
    optimizer,
    totals,
    confusion,
    video_consistency,
    collect_video_metrics,
):
    raw_features = stack_backbone_features([record["raw"] for record in records])
    observation = stack_unified_features([record["observation"] for record in records])
    posterior = stack_unified_features([record["posterior"] for record in records])
    clean_logits = torch.cat([record["clean_logits"] for record in records], dim=0)
    masks = torch.cat([record["mask"] for record in records], dim=0)
    semantic, distillation, logits = semantic_losses_batch(
        model, raw_features, observation, posterior, clean_logits, masks, output_size
    )
    temporal_predictions = [
        record.get("temporal_prediction")
        for record in records
        if record.get("temporal_prediction") is not None
    ]
    temporal_targets = [
        record["temporal_target"]
        for record in records
        if record.get("temporal_prediction") is not None
    ]
    persistence_predictions = [
        record["temporal_persistence"]
        for record in records
        if record.get("temporal_prediction") is not None
    ]
    if temporal_predictions:
        target_z1 = torch.cat([value[0] for value in temporal_targets], dim=0).detach()
        target_z4 = torch.cat([value[1] for value in temporal_targets], dim=0).detach()
        temporal_loss = 0.5 * (
            F.smooth_l1_loss(
                torch.cat([value[0] for value in temporal_predictions], dim=0),
                target_z1,
            )
            + F.smooth_l1_loss(
                torch.cat([value[1] for value in temporal_predictions], dim=0),
                target_z4,
            )
        )
        zero_temporal_loss = 0.5 * (
            F.smooth_l1_loss(torch.zeros_like(target_z1), target_z1)
            + F.smooth_l1_loss(torch.zeros_like(target_z4), target_z4)
        )
        persistence_temporal_loss = 0.5 * (
            F.smooth_l1_loss(
                torch.cat([value[0] for value in persistence_predictions], dim=0),
                target_z1,
            )
            + F.smooth_l1_loss(
                torch.cat([value[1] for value in persistence_predictions], dim=0),
                target_z4,
            )
        )
    else:
        temporal_loss = semantic.new_zeros(())
        zero_temporal_loss = semantic.new_zeros(())
        persistence_temporal_loss = semantic.new_zeros(())
    total = semantic + DISTILL_WEIGHT * distillation + TEMPORAL_PREDICTION_LOSS_WEIGHT * temporal_loss
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
    totals["segmentation_cross_entropy"].add_(semantic.detach() * len(records))
    totals["distillation_kl"].add_(distillation.detach() * len(records))
    if "temporal_prediction_loss" in totals:
        temporal_pair_count = len(temporal_predictions)
        totals["temporal_pair_count"].add_(temporal_pair_count)
        totals["temporal_prediction_loss"].add_(
            temporal_loss.detach() * temporal_pair_count
        )
        totals["zero_temporal_loss"].add_(
            zero_temporal_loss.detach() * temporal_pair_count
        )
        totals["persistence_temporal_loss"].add_(
            persistence_temporal_loss.detach() * temporal_pair_count
        )
    for level in ("z1", "z4"):
        for name in DYNAMIC_DIAGNOSTICS:
            value = records[0]["values"].get(f"{name}_{level}")
            if value is not None and f"mean_abs_{name}_{level}" in totals:
                total_value = sum(
                    record["values"][f"{name}_{level}"].abs().mean()
                    for record in records
                )
                totals[f"mean_abs_{name}_{level}"].add_(total_value.detach())
        value = records[0]["values"].get(f"gate_modulation_{level}")
        if value is not None and f"std_gate_modulation_{level}" in totals:
            total_value = sum(
                record["values"][f"gate_modulation_{level}"].std()
                for record in records
            )
            totals[f"std_gate_modulation_{level}"].add_(total_value.detach())
        if f"mean_update_gate_{level}" in totals:
            level_index = 0 if level == "z1" else 1
            for record in records:
                values = record["values"]
                update = values[f"update_gate_{level}"]
                reset = values[f"reset_gate_{level}"]
                residual = values[f"first_order_model_residual_{level}"]
                update_delta = values[f"update_gate_delta_{level}"]
                reset_delta = values[f"reset_gate_delta_{level}"]
                totals[f"mean_update_gate_{level}"].add_(update.detach().mean())
                totals[f"mean_reset_gate_{level}"].add_(reset.detach().mean())
                saturation = 0.5 * (
                    ((update < 0.05) | (update > 0.95)).float().mean()
                    + ((reset < 0.05) | (reset > 0.95)).float().mean()
                )
                totals[f"gate_saturation_rate_{level}"].add_(saturation.detach())
                totals[f"corr_abs_residual_update_delta_{level}"].add_(
                    absolute_pearson_correlation(residual, update_delta)
                )
                totals[f"corr_abs_residual_reset_delta_{level}"].add_(
                    absolute_pearson_correlation(residual, reset_delta)
                )
                totals[f"mean_abs_dynamic_error_{level}"].add_(
                    record["hidden"].dynamic_error[level_index].detach().abs().mean()
                )
    totals["total_loss"].add_(total.detach() * len(records))
    finite_tensors = [logits, total]
    for record in records:
        values = record["values"]
        hidden = record["hidden"]
        finite_tensors.extend(
            (
                record["prediction_error"].z1,
                record["prediction_error"].z4,
                values["aligned_error_z1"],
                values["aligned_error_z4"],
                values["task_error_z1"],
                values["task_error_z4"],
                hidden.hidden[0],
                hidden.hidden[1],
                hidden.dynamic_error[0],
                hidden.dynamic_error[1],
                record["posterior"].z1,
                record["posterior"].z4,
            )
        )
        for level in ("z1", "z4"):
            for name in DYNAMIC_DIAGNOSTICS + ("predicted_next_error",):
                value = values.get(f"{name}_{level}")
                if value is not None:
                    finite_tensors.append(value)
    finite = bool(torch.stack([torch.isfinite(value).all() for value in finite_tensors]).all().item())
    if confusion is not None:
        for index, record in enumerate(records):
            prediction = logits[index].argmax(dim=0).cpu().to(torch.int64)
            mask = record["mask"].squeeze(0)
            update_confusion_matrix(confusion, prediction, mask)
            if collect_video_metrics:
                video_consistency.append(mask, {"current_model": prediction})
    return finite


def run_epoch(
    model,
    predictor,
    corrections,
    sequence_loader,
    optimizer=None,
    collect_validation_metrics=False,
    collect_video_metrics=False,
):
    training = optimizer is not None
    corrections.train(training)
    total_keys = ["segmentation_cross_entropy", "distillation_kl", "total_loss"]
    if corrections[0].temporal_prediction is not None:
        total_keys.extend(
            (
                "temporal_pair_count",
                "temporal_prediction_loss",
                "zero_temporal_loss",
                "persistence_temporal_loss",
            )
        )
    if corrections[0].use_dynamic_error:
        for level in ("z1", "z4"):
            for name in DYNAMIC_DIAGNOSTICS:
                total_keys.append(f"mean_abs_{name}_{level}")
            total_keys.append(f"std_gate_modulation_{level}")
            total_keys.extend(f"{name}_{level}" for name in GATE_DIAGNOSTICS)
    totals = {key: torch.zeros((), device="cuda") for key in total_keys}
    frame_count = 0
    finite = True
    confusion = torch.zeros((19, 19), dtype=torch.int64) if collect_validation_metrics else None
    video_consistency = VideoConsistency(("current_model",)) if collect_video_metrics else None
    hidden = None
    predictor_hidden = None
    pending_dynamics = None
    pending_semantic = None
    previous_sequence = None
    previous_temporal_prediction = None
    previous_prediction_error = None
    records = []
    for chunk in sequence_loader:
        sequence_id = chunk["sequence_id"]
        start = chunk["start"]
        total_frames = chunk["total_frames"]
        if sequence_id != previous_sequence:
            hidden = None
            predictor_hidden = predictor.initial_state()
            pending_dynamics = None
            pending_semantic = None
            previous_temporal_prediction = None
            previous_prediction_error = None
            previous_sequence = sequence_id
            if video_consistency is not None:
                video_consistency.reset_sequence()
        clean_images = chunk["images"].cuda(non_blocking=True)
        masks = chunk["masks"]
        frame_indices = range(start, start + len(clean_images))
        clean_raw, corrupted_raw, _, observation_batch, clean_logits_batch = encode_frozen_sequence_chunk(
            model, clean_images, frame_indices, total_frames
        )
        output_size = tuple(clean_images.shape[-2:])
        for index, frame_index in enumerate(frame_indices):
            raw = select_backbone_features(corrupted_raw, index)
            observation = select_unified_features(observation_batch, index)
            clean_logits = clean_logits_batch[index : index + 1]
            mask = masks[index : index + 1]
            if hidden is None:
                hidden = zero_semantic_temporal_state(observation)
            if frame_index == 0:
                with torch.no_grad():
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor, observation, zero_state(observation), predictor_hidden
                    )
                continue
            prediction_error = error_state(observation, pending_dynamics)
            if frame_index == 1:
                with torch.no_grad():
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor, observation, prediction_error, predictor_hidden
                    )
                continue
            if frame_index < warmup_frame_count(total_frames):
                with torch.no_grad():
                    _, hidden, _ = semantic_temporal_error_step(
                        corrections, observation, pending_dynamics, pending_semantic, hidden
                    )
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor, observation, prediction_error, predictor_hidden
                    )
                hidden = detach_error_state(hidden)
                previous_temporal_prediction = None
                previous_prediction_error = None
                continue
            posterior, hidden, values = semantic_temporal_error_step(
                corrections, observation, pending_dynamics, pending_semantic, hidden
            )
            records.append(
                {
                    "raw": raw,
                    "observation": observation,
                    "posterior": posterior,
                    "clean_logits": clean_logits,
                    "mask": mask,
                    "prediction_error": prediction_error,
                    "values": values,
                    "hidden": hidden,
                    "temporal_prediction": previous_temporal_prediction,
                    "temporal_target": (
                        prediction_error.z1.detach(),
                        prediction_error.z4.detach(),
                    ),
                    "temporal_persistence": previous_prediction_error,
                }
            )
            if "predicted_next_error_z1" in values:
                previous_temporal_prediction = (
                    values["predicted_next_error_z1"],
                    values["predicted_next_error_z4"],
                )
            else:
                previous_temporal_prediction = None
            previous_prediction_error = (
                prediction_error.z1.detach(),
                prediction_error.z4.detach(),
            )
            frame_count += 1
            with torch.no_grad():
                pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                    predictor, observation, prediction_error, predictor_hidden
                )
            if len(records) == TRUNCATED_BPTT:
                finite = flush_bptt_batch(
                    model, records, output_size, optimizer, totals, confusion, video_consistency, collect_video_metrics
                ) and finite
                records.clear()
                hidden = detach_error_state(hidden)
                previous_temporal_prediction = None
                previous_prediction_error = None
            elif not training:
                hidden = detach_error_state(hidden)
        if start + len(clean_images) == total_frames and records:
            finite = flush_bptt_batch(
                model, records, output_size, optimizer, totals, confusion, video_consistency, collect_video_metrics
            ) and finite
            records.clear()
            hidden = detach_error_state(hidden)
            previous_temporal_prediction = None
            previous_prediction_error = None
    temporal_pair_count = int(totals.get("temporal_pair_count", torch.zeros(())).item())
    metrics = {}
    for key, value in totals.items():
        if key == "temporal_pair_count":
            metrics[key] = temporal_pair_count
        elif key in {
            "temporal_prediction_loss",
            "zero_temporal_loss",
            "persistence_temporal_loss",
        }:
            metrics[key] = value.item() / max(temporal_pair_count, 1)
        else:
            metrics[key] = value.item() / frame_count
    if temporal_pair_count:
        persistence = metrics["persistence_temporal_loss"]
        metrics["learned_vs_persistence_relative_improvement"] = (
            1.0 - metrics["temporal_prediction_loss"] / persistence
            if persistence > 0
            else 0.0
        )
    if collect_validation_metrics:
        metrics.update(
            {
                "miou": float(torch.nanmean(compute_iou(confusion)).item()),
                "wiou": weighted_iou(confusion),
            }
        )
    if collect_video_metrics:
        mvc = video_consistency.means()
        metrics.update({"mvc8": mvc[8]["current_model"], "mvc16": mvc[16]["current_model"]})
    return metrics, frame_count, finite


def gates(model, predictor, corrections, sample):
    image = load_image(sample)
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        observation = model.encode_backbone_features(raw)
        zero = zero_state(observation)
        clean_host = HostFeature(raw.c4, raw.c1, tuple(image.shape[-2:]))
        identity_host = corrected_host_feature(model, raw, observation, observation, tuple(image.shape[-2:]))
        identity_difference = float(
            (model.decode_from_host_feature(clean_host) - model.decode_from_host_feature(identity_host)).abs().max().item()
        )
        hidden = zero_semantic_temporal_state(observation)
        zero_dynamics = observation
        _, _, zero_values = semantic_temporal_error_step(
            corrections,
            observation,
            zero_dynamics,
            observation,
            hidden,
        )
        zero_delta = max(
            float(zero_values["delta_z1"].abs().max().item()),
            float(zero_values["delta_z4"].abs().max().item()),
        )
        error_identity = max(
            float(zero_values["error_z1"].sub(observation.z1 - zero_dynamics.z1).abs().max().item()),
            float(zero_values["error_z4"].sub(observation.z4 - zero_dynamics.z4).abs().max().item()),
        )
        zero_error_driven = max(
            float(zero_values[key].abs().max().item())
            for key in (
                "task_error_z1",
                "task_error_z4",
                "hidden_z1",
                "hidden_z4",
                "dynamic_error_z1",
                "dynamic_error_z4",
                "error_backbone_z1",
                "error_backbone_z4",
                "base_error_z1",
                "base_error_z4",
            )
        )
        local_correlation_finite = all(
            torch.isfinite(zero_values[key]).all().item()
            for key in (
                "aligned_error_z1",
                "aligned_error_z4",
                "max_attention_weight_z1",
                "max_attention_weight_z4",
                "attention_entropy_z1",
                "attention_entropy_z4",
            )
        )
    only_new_parameters = all(parameter.requires_grad for parameter in corrections.parameters())
    frozen_base = not any(parameter.requires_grad for parameter in model.parameters()) and not any(parameter.requires_grad for parameter in predictor.parameters())
    return {
        "prediction_error_identity_max_abs": error_identity,
        "prediction_error_identity_passed": error_identity == 0.0,
        "zero_hidden_delta_max_abs": zero_delta,
        "full_zero_error_gate_max_abs": zero_error_driven,
        "error_driven_correction_passed": zero_error_driven <= 1e-7,
        "local_correlation_finite": local_correlation_finite,
        "residual_writeback_identity_max_abs_logit": identity_difference,
        "residual_writeback_identity_passed": identity_difference <= 1e-6,
        "only_new_parameters_trainable": only_new_parameters and frozen_base,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Semantic temporal error correction requires CUDA")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    paths = make_paths()
    model, predictor = load_role_components(paths["static"], paths["adapter"], paths["predictor"], paths["writeback"])
    corrections = build_semantic_temporal_corrections()
    default_output = (
        "results/kitti_step_semantic_temporal_error_dynamic"
        if corrections[0].use_dynamic_error
        else (
            "results/kitti_step_semantic_temporal_prediction_baseline"
            if corrections[0].temporal_prediction is not None
            else "results/kitti_step_semantic_temporal_error_correction"
        )
    )
    output = Path(
        os.environ.get("PREDIFY_SEMANTIC_TEMPORAL_ERROR_OUTPUT_DIR", default_output)
    )
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    corrections.requires_grad_(True)
    optimizer = torch.optim.AdamW(corrections.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    parameter_efficiency = parameter_efficiency_metrics(model, predictor, corrections)
    train = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_groups = sequence_groups(train)
    val_groups = sequence_groups(val)
    if len(train_groups) != 12 or len(train.samples) != 5027 or len(val_groups) != 9 or len(val.samples) != 2981:
        raise RuntimeError("KITTI-STEP protocol mismatch")
    train_groups = limit_sequence_groups(
        train_groups,
        sequence_limit=TRAIN_SEQUENCE_LIMIT,
        frame_limit=FRAMES_PER_SEQUENCE_LIMIT,
    )
    val_groups = limit_sequence_groups(
        val_groups,
        sequence_limit=VAL_SEQUENCE_LIMIT,
        frame_limit=FRAMES_PER_SEQUENCE_LIMIT,
    )
    if not train_groups or not val_groups:
        raise ValueError("Quick-run limits removed every train or validation sequence")
    gate = gates(model, predictor, corrections, train.samples[0])
    if not all((value if isinstance(value, bool) else value <= 1e-6) for key, value in gate.items() if key.endswith("passed")):
        raise RuntimeError("semantic temporal error correction gate failed")
    train_loader = make_sequence_chunk_loader(train_groups)
    val_loader = make_sequence_chunk_loader(val_groups)
    history = []
    best = None
    early_stopping_best_miou = None
    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch = None
    for epoch in range(1, EPOCHS + 1):
        torch.cuda.synchronize()
        train_start = time.perf_counter()
        train_metrics, train_frames, train_finite = run_epoch(
            model, predictor, corrections, train_loader, optimizer
        )
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_start
        train_throughput = train_frames / train_seconds
        torch.cuda.synchronize()
        val_start = time.perf_counter()
        with torch.no_grad():
            val_metrics, val_frames, val_finite = run_epoch(
                model,
                predictor,
                corrections,
                val_loader,
                collect_validation_metrics=True,
            )
        torch.cuda.synchronize()
        val_seconds = time.perf_counter() - val_start
        row = {
            "epoch": epoch,
            "train": {
                **train_metrics,
                "effective_frame_count": train_frames,
                "finite": train_finite,
                "epoch_seconds": train_seconds,
                "throughput_frames_per_second": train_throughput,
            },
            "val": {
                **val_metrics,
                "effective_frame_count": val_frames,
                "finite": val_finite,
                "epoch_seconds": val_seconds,
                "throughput_frames_per_second": val_frames / val_seconds,
            },
        }
        if best is None:
            checkpoint_improved = True
        else:
            checkpoint_miou_delta = val_metrics["miou"] - best["val"]["miou"]
            checkpoint_improved = checkpoint_miou_delta > MIOU_TIE_TOLERANCE or (
                abs(checkpoint_miou_delta) <= MIOU_TIE_TOLERANCE
                and val_metrics["total_loss"] < best["val"]["total_loss"]
            )

        if checkpoint_improved:
            best = row
            output.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "corrections": corrections.state_dict(),
                    "epoch": epoch,
                    "selection_metric": "val_miou",
                    "val_metrics": val_metrics,
                },
                output / "best_semantic_temporal_error_correction.pt",
            )

        meaningful_miou_improvement = (
            early_stopping_best_miou is None
            or val_metrics["miou"]
            > early_stopping_best_miou + EARLY_STOPPING_MIN_MIOU_IMPROVEMENT
        )
        if meaningful_miou_improvement:
            early_stopping_best_miou = val_metrics["miou"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row["early_stopping"] = {
            "meaningful_miou_improvement": meaningful_miou_improvement,
            "best_miou_reference": early_stopping_best_miou,
            "min_miou_improvement": EARLY_STOPPING_MIN_MIOU_IMPROVEMENT,
            "epochs_without_improvement": epochs_without_improvement,
            "patience": EARLY_STOPPING_PATIENCE,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            stopped_early = True
            stop_epoch = epoch
            break
    best_payload = torch.load(
        output / "best_semantic_temporal_error_correction.pt",
        map_location="cpu",
        weights_only=False,
    )
    corrections.load_state_dict(best_payload["corrections"], strict=True)
    corrections.eval()
    torch.cuda.synchronize()
    final_start = time.perf_counter()
    with torch.no_grad():
        final_metrics, final_frames, final_finite = run_epoch(
            model,
            predictor,
            corrections,
            val_loader,
            collect_validation_metrics=True,
            collect_video_metrics=True,
        )
    torch.cuda.synchronize()
    final_seconds = time.perf_counter() - final_start
    summary = {
        "experiment": "kitti_step_semantic_temporal_error_correction_training",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(output / "best_semantic_temporal_error_correction.pt"),
        "parameter_efficiency": parameter_efficiency,
        "config": {"max_epochs": EPOCHS, "early_stopping_patience": EARLY_STOPPING_PATIENCE, "early_stopping_min_miou_improvement": EARLY_STOPPING_MIN_MIOU_IMPROVEMENT, "checkpoint_selection": "highest_val_miou_then_lower_val_loss", "miou_tie_tolerance": MIOU_TIE_TOLERANCE, "truncated_bptt": TRUNCATED_BPTT, "frozen_encode_batch_size": FROZEN_ENCODE_BATCH_SIZE, "loader_workers": LOADER_WORKERS, "loader_prefetch_factor": LOADER_PREFETCH_FACTOR, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "seed": SEED, "temperature": 1.0, "blur_kernel_size": BLUR_KERNEL_SIZE, "blur_sigma_levels": BLUR_SIGMA_LEVELS, "blur_sigma_max": BLUR_SIGMA_MAX, "blur_warmup_fraction": BLUR_WARMUP_FRACTION, "warmup_used_for_state_only": True, "labels_used_for_training": True, "distillation_weight": DISTILL_WEIGHT, "temporal_prediction_loss_weight": TEMPORAL_PREDICTION_LOSS_WEIGHT, "temporal_prediction_enabled": corrections[0].temporal_prediction is not None, "temporal_baselines": ["zero", "persistence"], "prediction_error_definition": "observation_minus_prediction", "dynamic_error_definition": "epsilon_t=epsilon_(t-1)+(Ts/tau_e)*(e_t-K_e*epsilon_(t-1))", "dynamic_error_sample_time": DYNAMIC_ERROR_SAMPLE_TIME, "dynamic_error_time_constant": DYNAMIC_ERROR_TIME_CONSTANT, "dynamic_error_gain": DYNAMIC_ERROR_GAIN, "dynamic_error_effective_q": DYNAMIC_ERROR_SAMPLE_TIME * DYNAMIC_ERROR_GAIN / DYNAMIC_ERROR_TIME_CONSTANT, "dynamic_error_enabled": corrections[0].use_dynamic_error, "dynamic_error_usage": "gate_modulation_and_H_next_error_prediction" if corrections[0].use_dynamic_error else "tracked_only_not_connected_to_correction"},
        "dataset": {"train_sequence_count": len(train_groups), "train_frame_count": sum(len(samples) for samples in train_groups.values()), "val_sequence_count": len(val_groups), "val_frame_count": sum(len(samples) for samples in val_groups.values()), "full_protocol_train_frame_count": len(train.samples), "full_protocol_val_frame_count": len(val.samples), "quick_subset": any((TRAIN_SEQUENCE_LIMIT, VAL_SEQUENCE_LIMIT, FRAMES_PER_SEQUENCE_LIMIT))},
        "base_checkpoints": {key: str(value) for key, value in paths.items()},
        "gates": gate,
        "history": history,
        "training_throughput": {
            "unit": "frames_per_second",
            "per_epoch": [
                row["train"]["throughput_frames_per_second"]
                for row in history
            ],
            "mean": sum(
                row["train"]["throughput_frames_per_second"]
                for row in history
            ) / len(history),
        },
        "best_epoch": best["epoch"],
        "best_val_miou": best["val"]["miou"],
        "best_val_total_loss": best["val"]["total_loss"],
        "final_validation": {
            **final_metrics,
            "effective_frame_count": final_frames,
            "finite": final_finite,
            "seconds": final_seconds,
            "throughput_frames_per_second": final_frames / final_seconds,
        },
        "early_stopping_best_miou": early_stopping_best_miou,
        "epochs_without_miou_improvement": epochs_without_improvement,
        "stopped_early": stopped_early,
        "stop_epoch": stop_epoch,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
