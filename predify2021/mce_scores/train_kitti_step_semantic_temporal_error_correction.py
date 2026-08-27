import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
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
    persistent_gaussian_blur,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    residual_writeback_host_feature,
)
from predify2021.mce_scores.semantic_temporal_error_step import (
    detach_error_state,
    semantic_temporal_error_step,
    zero_error_state,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    HostFeature,
    UnifiedFeatures,
)
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import (
    build_semantic_temporal_corrections,
)


SEED = 0
EPOCHS = 3
TRUNCATED_BPTT = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
DISTILL_WEIGHT = 0.5


def corrected_host_feature(model, raw_features, observation, posterior, output_size):
    delta = UnifiedFeatures(
        posterior.z1 - observation.z1,
        torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3),
        posterior.z4 - observation.z4,
    )
    return residual_writeback_host_feature(model, raw_features, delta, output_size)


def semantic_losses(model, raw_features, observation, posterior, clean_logits, mask, output_size):
    host_feature = corrected_host_feature(model, raw_features, observation, posterior, output_size)
    logits = model.decode_from_host_feature(host_feature)
    semantic = F.cross_entropy(logits, mask.unsqueeze(0).cuda(), ignore_index=255)
    teacher_probability = F.softmax(clean_logits.detach(), dim=1)
    distillation = F.kl_div(
        F.log_softmax(logits, dim=1), teacher_probability, reduction="none"
    ).sum(dim=1).mean()
    return semantic, distillation, logits


def flush_chunk(optimizer, losses):
    if not losses:
        return
    optimizer.zero_grad(set_to_none=True)
    (torch.stack(losses).mean()).backward()
    optimizer.step()
    losses.clear()


def run_epoch(model, predictor, corrections, groups, optimizer=None):
    training = optimizer is not None
    corrections.train(training)
    totals = {"segmentation_cross_entropy": 0.0, "distillation_kl": 0.0, "total_loss": 0.0}
    frame_count = 0
    chunk_losses = []
    finite = True
    for samples in groups.values():
        hidden = None
        pending_dynamics = None
        pending_semantic = None
        for frame_index, sample in enumerate(samples):
            clean_image = load_image(sample)
            corrupted_image = persistent_gaussian_blur(clean_image, frame_index, len(samples))
            with torch.no_grad():
                clean_raw = model.extract_backbone_features(clean_image)
                corrupted_raw = model.extract_backbone_features(corrupted_image)
                clean_state = model.encode_backbone_features(clean_raw)
                observation = model.encode_backbone_features(corrupted_raw)
                clean_logits = model.decode_from_host_feature(
                    HostFeature(clean_raw.c4, clean_raw.c1, tuple(clean_image.shape[-2:]))
                )
                if hidden is None:
                    hidden = zero_error_state(observation)
                if frame_index == 0:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor, observation, zero_state(observation), predictor.initial_state()
                    )
                    continue
                error = error_state(observation, pending_dynamics)
                if frame_index == 1:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor, observation, error, predictor_hidden
                    )
                    continue
            posterior, hidden, values = semantic_temporal_error_step(
                corrections, observation, pending_dynamics, pending_semantic, hidden
            )
            mask = semantic_mask_from_panoptic_png(sample["mask_path"])
            semantic, distillation, logits = semantic_losses(
                model,
                corrupted_raw,
                observation,
                posterior,
                clean_logits,
                mask,
                tuple(clean_image.shape[-2:]),
            )
            total = semantic + DISTILL_WEIGHT * distillation
            if training:
                chunk_losses.append(total)
                if len(chunk_losses) == TRUNCATED_BPTT:
                    flush_chunk(optimizer, chunk_losses)
                    hidden = detach_error_state(hidden)
            measurements = {
                "segmentation_cross_entropy": semantic,
                "distillation_kl": distillation,
                "total_loss": total,
            }
            for key, value in measurements.items():
                totals[key] += value.detach().item()
            finite = finite and all(
                torch.isfinite(value).all().item()
                for value in (
                    error.z1,
                    error.z4,
                    values["aligned_error_z1"],
                    values["aligned_error_z4"],
                    values["task_error_z1"],
                    values["task_error_z4"],
                    hidden[0],
                    hidden[1],
                    posterior.z1,
                    posterior.z4,
                    logits,
                    total,
                )
            )
            frame_count += 1
            with torch.no_grad():
                pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                    predictor, observation, error, predictor_hidden
                )
            if not training:
                hidden = detach_error_state(hidden)
        if training:
            flush_chunk(optimizer, chunk_losses)
            if hidden is not None:
                hidden = detach_error_state(hidden)
    return {key: value / frame_count for key, value in totals.items()}, frame_count, finite


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
        hidden = zero_error_state(observation)
        zero_dynamics = observation
        zero_result1 = corrections[0](observation.z1, zero_dynamics.z1, observation.z1, hidden[0])
        zero_result4 = corrections[1](observation.z4, zero_dynamics.z4, observation.z4, hidden[1])
        zero_values = (zero_result1, zero_result4)
        zero_delta = max(
            float(zero_result1[4].abs().max().item()),
            float(zero_result4[4].abs().max().item()),
        )
        error_identity = max(
            float(zero_result1[0].sub(observation.z1 - zero_dynamics.z1).abs().max().item()),
            float(zero_result4[0].sub(observation.z4 - zero_dynamics.z4).abs().max().item()),
        )
        zero_error_driven = max(
            float(value[index].abs().max().item())
            for value in zero_values
            for index in (2, 3, 7, 8)
        )
        local_correlation_finite = all(
            torch.isfinite(value[index]).all().item()
            for value in zero_values
            for index in (1, 5, 6)
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
    output = Path(os.environ.get("PREDIFY_SEMANTIC_TEMPORAL_ERROR_OUTPUT_DIR", "results/kitti_step_semantic_temporal_error_correction"))
    paths = make_paths()
    model, predictor = load_role_components(paths["static"], paths["adapter"], paths["predictor"], paths["writeback"])
    corrections = build_semantic_temporal_corrections()
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    corrections.requires_grad_(True)
    optimizer = torch.optim.AdamW(corrections.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_groups = sequence_groups(train)
    val_groups = sequence_groups(val)
    if len(train_groups) != 12 or len(train.samples) != 5027 or len(val_groups) != 9 or len(val.samples) != 2981:
        raise RuntimeError("KITTI-STEP protocol mismatch")
    gate = gates(model, predictor, corrections, train.samples[0])
    if not all((value if isinstance(value, bool) else value <= 1e-6) for key, value in gate.items() if key.endswith("passed")):
        raise RuntimeError("semantic temporal error correction gate failed")
    history = []
    best = None
    for epoch in range(1, EPOCHS + 1):
        train_metrics, train_frames, train_finite = run_epoch(model, predictor, corrections, train_groups, optimizer)
        with torch.no_grad():
            val_metrics, val_frames, val_finite = run_epoch(model, predictor, corrections, val_groups)
        row = {"epoch": epoch, "train": {**train_metrics, "effective_frame_count": train_frames, "finite": train_finite}, "val": {**val_metrics, "effective_frame_count": val_frames, "finite": val_finite}}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if best is None or val_metrics["total_loss"] < best["val"]["total_loss"]:
            best = row
            output.mkdir(parents=True, exist_ok=True)
            torch.save({"corrections": corrections.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, output / "best_semantic_temporal_error_correction.pt")
    summary = {
        "experiment": "kitti_step_semantic_temporal_error_correction_training",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(output / "best_semantic_temporal_error_correction.pt"),
        "trainable_parameter_count": sum(parameter.numel() for parameter in corrections.parameters()),
        "config": {"epochs": EPOCHS, "truncated_bptt": TRUNCATED_BPTT, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "seed": SEED, "temperature": 1.0, "blur_kernel_size": BLUR_KERNEL_SIZE, "blur_sigma_levels": BLUR_SIGMA_LEVELS, "blur_sigma_max": BLUR_SIGMA_MAX, "labels_used_for_training": True, "distillation_weight": DISTILL_WEIGHT},
        "dataset": {"train_sequence_count": len(train_groups), "train_frame_count": len(train.samples), "val_sequence_count": len(val_groups), "val_frame_count": len(val.samples)},
        "base_checkpoints": {key: str(value) for key, value in paths.items()},
        "gates": gate,
        "history": history,
        "best_epoch": best["epoch"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
