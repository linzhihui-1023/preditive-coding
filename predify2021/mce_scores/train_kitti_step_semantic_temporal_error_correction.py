import json
import os
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
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
    detach_error_state,
    semantic_temporal_error_step,
    zero_dynamic_transition_state,
)
from predify2021.mce_scores.video_metrics import VideoConsistency, weighted_iou
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    HostFeature,
    UnifiedFeatures,
)
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import (
    build_dpc_semantic_temporal_corrections,
)


SEED = 0
EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_MIOU_IMPROVEMENT = 1e-4
MIOU_TIE_TOLERANCE = 1e-6
TRUNCATED_BPTT = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
DISTILL_WEIGHT = 0.5


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
    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_params / total_params,
        "trainable_ratio_percent": 100.0 * trainable_params / total_params,
    }


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


def run_epoch(model, predictor, corrections, groups, optimizer=None, collect_validation_metrics=False):
    training = optimizer is not None
    corrections.train(training)
    totals = {"segmentation_cross_entropy": 0.0, "distillation_kl": 0.0, "total_loss": 0.0}
    frame_count = 0
    chunk_losses = []
    finite = True
    confusion = torch.zeros((19, 19), dtype=torch.int64) if collect_validation_metrics else None
    video_consistency = VideoConsistency(("current_model",)) if collect_validation_metrics else None
    for samples in groups.values():
        hidden = None
        pending_dynamics = None
        pending_semantic = None
        if video_consistency is not None:
            video_consistency.reset_sequence()
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
                    hidden = zero_dynamic_transition_state(observation)
                if frame_index == 0:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor, observation, zero_state(observation), predictor.initial_state()
                    )
                    continue
                prediction_error = error_state(observation, pending_dynamics)
                if frame_index == 1:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor, observation, prediction_error, predictor_hidden
                    )
                    continue
            if frame_index < warmup_frame_count(len(samples)):
                with torch.no_grad():
                    _, hidden, _ = semantic_temporal_error_step(
                        corrections, observation, pending_dynamics, pending_semantic, hidden
                    )
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor, observation, prediction_error, predictor_hidden
                    )
                hidden = detach_error_state(hidden)
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
                    prediction_error.z1,
                    prediction_error.z4,
                    values["aligned_error_z1"],
                    values["aligned_error_z4"],
                    values["task_error_z1"],
                    values["task_error_z4"],
                    hidden.hidden[0],
                    hidden.hidden[1],
                    hidden.dynamic_error[0],
                    hidden.dynamic_error[1],
                    posterior.z1,
                    posterior.z4,
                    logits,
                    total,
                )
            )
            if collect_validation_metrics:
                prediction = logits.argmax(dim=1).squeeze(0).cpu().to(torch.int64)
                update_confusion_matrix(confusion, prediction, mask)
                video_consistency.append(mask, {"current_model": prediction})
            frame_count += 1
            with torch.no_grad():
                pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                    predictor, observation, prediction_error, predictor_hidden
                )
            if not training:
                hidden = detach_error_state(hidden)
        if training:
            flush_chunk(optimizer, chunk_losses)
            if hidden is not None:
                hidden = detach_error_state(hidden)
    metrics = {key: value / frame_count for key, value in totals.items()}
    if collect_validation_metrics:
        mvc = video_consistency.means()
        metrics.update(
            {
                "miou": float(torch.nanmean(compute_iou(confusion)).item()),
                "wiou": weighted_iou(confusion),
                "mvc8": mvc[8]["current_model"],
                "mvc16": mvc[16]["current_model"],
            }
        )
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
        hidden = zero_dynamic_transition_state(observation)
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
    output = Path(os.environ.get("PREDIFY_DPC_TRANSITION_OUTPUT_DIR", "results/kitti_step_dpc_transition_correction"))
    paths = make_paths()
    model, predictor = load_role_components(paths["static"], paths["adapter"], paths["predictor"], paths["writeback"])
    corrections = build_dpc_semantic_temporal_corrections()
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
    gate = gates(model, predictor, corrections, train.samples[0])
    if not all((value if isinstance(value, bool) else value <= 1e-6) for key, value in gate.items() if key.endswith("passed")):
        raise RuntimeError("semantic temporal error correction gate failed")
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
            model, predictor, corrections, train_groups, optimizer
        )
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_start
        train_throughput = train_frames / train_seconds
        with torch.no_grad():
            val_metrics, val_frames, val_finite = run_epoch(
                model,
                predictor,
                corrections,
                val_groups,
                collect_validation_metrics=True,
            )
        row = {
            "epoch": epoch,
            "train": {
                **train_metrics,
                "effective_frame_count": train_frames,
                "finite": train_finite,
                "epoch_seconds": train_seconds,
                "throughput_frames_per_second": train_throughput,
            },
            "val": {**val_metrics, "effective_frame_count": val_frames, "finite": val_finite},
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
                output / "best_dpc_semantic_temporal_error_correction.pt",
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
    summary = {
        "experiment": "kitti_step_dpc_transition_correction_training",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(output / "best_dpc_semantic_temporal_error_correction.pt"),
        "parameter_efficiency": parameter_efficiency,
        "config": {"max_epochs": EPOCHS, "early_stopping_patience": EARLY_STOPPING_PATIENCE, "early_stopping_min_miou_improvement": EARLY_STOPPING_MIN_MIOU_IMPROVEMENT, "checkpoint_selection": "highest_val_miou_then_lower_val_loss", "miou_tie_tolerance": MIOU_TIE_TOLERANCE, "truncated_bptt": TRUNCATED_BPTT, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "seed": SEED, "temperature": 1.0, "blur_kernel_size": BLUR_KERNEL_SIZE, "blur_sigma_levels": BLUR_SIGMA_LEVELS, "blur_sigma_max": BLUR_SIGMA_MAX, "blur_warmup_fraction": BLUR_WARMUP_FRACTION, "warmup_used_for_state_only": True, "labels_used_for_training": True, "distillation_weight": DISTILL_WEIGHT, "dynamic_error_initial_beta": 0.793, "transition_basis_count": 3, "transition_residual_scale": 0.1},
        "dataset": {"train_sequence_count": len(train_groups), "train_frame_count": len(train.samples), "val_sequence_count": len(val_groups), "val_frame_count": len(val.samples)},
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
