import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.kitti_step_persistent_blur import (
    BLUR_KERNEL_SIZE,
    BLUR_SIGMA_LEVELS,
    BLUR_SIGMA_MAX,
    persistent_gaussian_blur,
)
from predify2021.mce_scores.role_separated_direct_state_correction import (
    ALPHA,
    BETA,
    direct_posterior,
    error_state,
    load_image,
    load_role_components,
    make_paths,
    next_role_prediction,
    raw_state_mse,
    relative_state_loss,
    update_dynamic_error,
    zero_initialize,
    zero_state,
    build_corrections,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import residual_writeback_host_feature
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures


SEED = 0
EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EXPECTED_TRAIN_SEQUENCES = 12
EXPECTED_TRAIN_FRAMES = 5027


def semantic_loss(model, raw_features, observation, posterior, target_mask, output_size):
    delta = UnifiedFeatures(
        posterior.z1 - observation.z1,
        torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3),
        posterior.z4 - observation.z4,
    )
    corrected = residual_writeback_host_feature(model, raw_features, delta, output_size)
    return F.cross_entropy(
        model.decode_from_host_feature(corrected),
        target_mask.unsqueeze(0).cuda(),
        ignore_index=255,
    )


def identity_gate(model, sample):
    image = load_image(sample)
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        observation = model.encode_backbone_features(raw)
        zero = zero_state(observation)
        residual = residual_writeback_host_feature(model, raw, zero, tuple(image.shape[-2:]))
        direct = HostFeature(raw.c4, raw.c1, tuple(image.shape[-2:]))
        difference = (model.decode_from_host_feature(residual) - model.decode_from_host_feature(direct)).abs().max()
    return float(difference.item())


def run_epoch(model, predictor, corrections, groups, optimizer, training):
    corrections.train(training)
    totals = {}
    frame_count = 0
    gradient_nonzero = None
    for samples in groups.values():
        hidden = (None, None, None, None)
        dynamic_error = None
        pending_dynamics = None
        for frame_index, sample in enumerate(samples):
            clean_image = load_image(sample)
            corrupted_image = persistent_gaussian_blur(clean_image, frame_index, len(samples))
            with torch.no_grad():
                clean_raw = model.extract_backbone_features(clean_image)
                corrupted_raw = model.extract_backbone_features(corrupted_image)
                clean_state = model.encode_backbone_features(clean_raw)
                observation = model.encode_backbone_features(corrupted_raw)
                if frame_index == 0:
                    pending_dynamics, _, *hidden = next_role_prediction(
                        predictor, observation, zero_state(observation), hidden
                    )
                    continue
                error = error_state(observation, pending_dynamics)
                if frame_index == 1:
                    pending_dynamics, _, *hidden = next_role_prediction(
                        predictor, observation, error, hidden
                    )
                    continue
                dynamic_error = update_dynamic_error(error, dynamic_error)
                if frame_index < len(samples) // 3:
                    pending_dynamics, _, *hidden = next_role_prediction(
                        predictor, observation, error, hidden
                    )
                    dynamic_error = type(dynamic_error)(
                        *(value.detach() for value in dynamic_error.as_tuple())
                    )
                    continue
            posterior, _ = direct_posterior(observation, error, dynamic_error, corrections)
            z1, z4, _ = raw_state_mse(posterior, clean_state)
            relative = relative_state_loss(posterior, clean_state, observation)
            semantic = semantic_loss(
                model,
                corrupted_raw,
                observation,
                posterior,
                semantic_mask_from_panoptic_png(sample["mask_path"]),
                tuple(clean_image.shape[-2:]),
            )
            total = semantic + 0.5 * relative
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                if gradient_nonzero is None:
                    gradient_nonzero = any(
                        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                        for parameter in corrections.parameters()
                    )
                optimizer.step()
            values = {
                "semantic_cross_entropy": semantic,
                "relative_state_loss": relative,
                "total_loss": total,
                "raw_state_mse_z1": z1,
                "raw_state_mse_z4": z4,
            }
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value.detach().item()
            frame_count += 1
            with torch.no_grad():
                pending_dynamics, _, *hidden = next_role_prediction(
                    predictor, observation, error, hidden
                )
                dynamic_error = type(dynamic_error)(*(value.detach() for value in dynamic_error.as_tuple()))
    return {key: value / frame_count for key, value in totals.items()}, frame_count, gradient_nonzero


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Persistent blur Corrected B training requires CUDA")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(os.environ.get("PREDIFY_PERSISTENT_BLUR_CORRECTED_B_OUTPUT_DIR", "results/kitti_step_persistent_blur_corrected_b"))
    paths = make_paths()
    model, predictor = load_role_components(paths["static"], paths["adapter"], paths["predictor"], paths["writeback"])
    corrections = build_corrections()
    zero_initialize(corrections)
    parameters = list(corrections.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_groups = sequence_groups(train)
    val_groups = sequence_groups(val)
    if len(train_groups) != EXPECTED_TRAIN_SEQUENCES or len(train.samples) != EXPECTED_TRAIN_FRAMES:
        raise RuntimeError("KITTI-STEP train protocol mismatch")
    identity_difference = identity_gate(model, train.samples[0])
    if identity_difference > 1e-6:
        raise RuntimeError("Residual writeback identity gate failed")
    history = []
    best = None
    for epoch in range(1, EPOCHS + 1):
        train_values, train_frames, gradient_nonzero = run_epoch(model, predictor, corrections, train_groups, optimizer, True)
        with torch.no_grad():
            val_values, val_frames, _ = run_epoch(model, predictor, corrections, val_groups, None, False)
        record = {
            "epoch": epoch,
            "train": {**train_values, "effective_frame_count": train_frames},
            "val": {**val_values, "effective_frame_count": val_frames},
            "correction_gradient_nonzero": gradient_nonzero,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if best is None or val_values["total_loss"] < best["val"]["total_loss"]:
            best = record
            output.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "mode": "corrected_b_persistent_blur",
                    "corrections": {str(index): corrections[position].state_dict() for position, index in enumerate((0, 3))},
                    "epoch": epoch,
                    "val": val_values,
                },
                output / "best_corrected_b_persistent_blur.pt",
            )
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "kitti_step_persistent_blur_corrected_b_training",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(output / "best_corrected_b_persistent_blur.pt"),
        "base_checkpoints": {key: str(value) for key, value in paths.items()},
        "config": {
            "epochs": EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "blur_kernel_size": BLUR_KERNEL_SIZE,
            "blur_sigma_max": BLUR_SIGMA_MAX,
            "blur_sigma_levels": BLUR_SIGMA_LEVELS,
            "alpha": ALPHA,
            "beta": BETA,
            "semantic_loss": "CrossEntropy(ignore_index=255)",
            "relative_state_loss_weight": 0.5,
        },
        "dataset": {
            "train_sequence_count": len(train_groups),
            "train_frame_count": len(train.samples),
            "val_sequence_count": len(val_groups),
            "val_frame_count": len(val.samples),
        },
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "gates": {
            "identity_max_abs_logit_difference": identity_difference,
            "identity_passed": identity_difference <= 1e-6,
            "correction_gradient_nonzero": all(record["correction_gradient_nonzero"] for record in history),
            "predictor_history_is_corrupted_observation": True,
            "correction_feedback_to_predictor": False,
            "clean_state_used_only_as_target": True,
        },
        "history": history,
        "best_epoch": best["epoch"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
