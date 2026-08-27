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
    relative_state_loss,
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
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorDecompositionCorrection,
    UnifiedFeatures,
)


SEED = 0
ALPHA = 0.207
BETA = 0.793
EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EXPECTED_TRAIN_SEQUENCES = 12
EXPECTED_TRAIN_FRAMES = 5027


def build_corrections():
    corrections = torch.nn.ModuleList(
        [ErrorDecompositionCorrection(), ErrorDecompositionCorrection()]
    ).cuda()
    for correction in corrections:
        torch.nn.init.zeros_(correction.direct.output_projection.weight)
        torch.nn.init.zeros_(correction.direct.output_projection.bias)
    return corrections


def state_with_z1_z4(z1, z4, state):
    return UnifiedFeatures(z1, state.z2, state.z3, z4)


def corruption_dynamic_error(estimated_corruption, previous, disable_dynamic):
    if disable_dynamic:
        return zero_state(estimated_corruption)
    if previous is None:
        previous = zero_state(estimated_corruption)
    return UnifiedFeatures(
        ALPHA * estimated_corruption.z1 + BETA * previous.z1,
        ALPHA * estimated_corruption.z2 + BETA * previous.z2,
        ALPHA * estimated_corruption.z3 + BETA * previous.z3,
        ALPHA * estimated_corruption.z4 + BETA * previous.z4,
    )


def decomposition_losses(correction, observation, error, dynamic_error, clean_state):
    estimated_corruption_1, estimated_mismatch_1, reliability_1, delta_1 = correction[0](
        observation.z1, error.z1, dynamic_error.z1
    )
    estimated_corruption_4, estimated_mismatch_4, reliability_4, delta_4 = correction[1](
        observation.z4, error.z4, dynamic_error.z4
    )
    c1 = observation.z1 - clean_state.z1
    c4 = observation.z4 - clean_state.z4
    p1 = clean_state.z1 - (observation.z1 - error.z1)
    p4 = clean_state.z4 - (observation.z4 - error.z4)
    decomp_1 = F.mse_loss(estimated_corruption_1, c1) / (
        F.mse_loss(c1, torch.zeros_like(c1)) + 1e-12
    )
    decomp_4 = F.mse_loss(estimated_corruption_4, c4) / (
        F.mse_loss(c4, torch.zeros_like(c4)) + 1e-12
    )
    target_rel_1 = c1.abs() / (c1.abs() + p1.abs() + 1e-12)
    target_rel_4 = c4.abs() / (c4.abs() + p4.abs() + 1e-12)
    rel_loss = 0.5 * (
        F.l1_loss(reliability_1, target_rel_1)
        + F.l1_loss(reliability_4, target_rel_4)
    )
    posterior = UnifiedFeatures(
        observation.z1 + delta_1,
        observation.z2,
        observation.z3,
        observation.z4 + delta_4,
    )
    return posterior, {
        "decomposition_loss": 0.5 * (decomp_1 + decomp_4),
        "reliability_loss": rel_loss,
        "corruption_nmse_z1": decomp_1,
        "corruption_nmse_z4": decomp_4,
        "mean_abs_true_corruption_z1": c1.abs().mean(),
        "mean_abs_true_corruption_z4": c4.abs().mean(),
        "mean_abs_estimated_corruption_z1": estimated_corruption_1.abs().mean(),
        "mean_abs_estimated_corruption_z4": estimated_corruption_4.abs().mean(),
        "mean_abs_true_mismatch_z1": p1.abs().mean(),
        "mean_abs_true_mismatch_z4": p4.abs().mean(),
        "mean_abs_estimated_mismatch_z1": estimated_mismatch_1.abs().mean(),
        "mean_abs_estimated_mismatch_z4": estimated_mismatch_4.abs().mean(),
        "reliability_l1_z1": F.l1_loss(reliability_1, target_rel_1),
        "reliability_l1_z4": F.l1_loss(reliability_4, target_rel_4),
        "mean_reliability_z1": reliability_1.mean(),
        "mean_reliability_z4": reliability_4.mean(),
        "mean_abs_dynamic_corruption_z1": dynamic_error.z1.abs().mean(),
        "mean_abs_dynamic_corruption_z4": dynamic_error.z4.abs().mean(),
        "mean_abs_delta_z1": delta_1.abs().mean(),
        "mean_abs_delta_z4": delta_4.abs().mean(),
    }


def semantic_posterior_loss(model, raw_features, observation, posterior, target_mask, output_size):
    delta = UnifiedFeatures(
        posterior.z1 - observation.z1,
        torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3),
        posterior.z4 - observation.z4,
    )
    corrected = residual_writeback_host_feature(model, raw_features, delta, output_size)
    logits = model.decode_from_host_feature(corrected)
    return F.cross_entropy(logits, target_mask.unsqueeze(0).cuda(), ignore_index=255)


def run_epoch(model, predictor, corrections, groups, optimizer, training, disable_dynamic):
    corrections.train(training)
    totals = {}
    frame_count = 0
    gradient_nonzero = None
    for samples in groups.values():
        hidden = (None, None, None, None)
        previous_dynamic = None
        pending_dynamics = None
        for frame_index, sample in enumerate(samples):
            clean_image = load_image(sample)
            noisy_image = persistent_gaussian_blur(clean_image, frame_index, len(samples))
            with torch.no_grad():
                clean_raw = model.extract_backbone_features(clean_image)
                noisy_raw = model.extract_backbone_features(noisy_image)
                clean_state = model.encode_backbone_features(clean_raw)
                observation = model.encode_backbone_features(noisy_raw)
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
            dynamic_input = previous_dynamic if previous_dynamic is not None else zero_state(error)
            estimated_for_dynamic = corrections[0].interpreter(observation.z1, error.z1)
            estimated_for_dynamic_4 = corrections[1].interpreter(observation.z4, error.z4)
            dynamic_error = corruption_dynamic_error(
                UnifiedFeatures(
                    estimated_for_dynamic,
                    torch.zeros_like(error.z2),
                    torch.zeros_like(error.z3),
                    estimated_for_dynamic_4,
                ),
                dynamic_input,
                disable_dynamic,
            )
            if frame_index < len(samples) // 3:
                with torch.no_grad():
                    pending_dynamics, _, *hidden = next_role_prediction(
                        predictor, observation, error, hidden
                    )
                previous_dynamic = UnifiedFeatures(
                    *(value.detach() for value in dynamic_error.as_tuple())
                )
                continue
            posterior, values = decomposition_losses(
                corrections, observation, error, dynamic_error, clean_state
            )
            semantic_value = semantic_posterior_loss(
                model,
                noisy_raw,
                observation,
                posterior,
                semantic_mask_from_panoptic_png(sample["mask_path"]),
                tuple(clean_image.shape[-2:]),
            )
            relative_value = relative_state_loss(posterior, clean_state, observation)
            total_loss = (
                semantic_value
                + 0.5 * relative_value
                + 0.25 * values["decomposition_loss"]
                + 0.10 * values["reliability_loss"]
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if gradient_nonzero is None:
                    gradient_nonzero = any(
                        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                        for parameter in corrections.parameters()
                    )
                optimizer.step()
            measurements = {
                "semantic_cross_entropy": semantic_value,
                "relative_state_loss": relative_value,
                "total_loss": total_loss,
                **values,
            }
            for key, value in measurements.items():
                totals[key] = totals.get(key, 0.0) + value.detach().item()
            frame_count += 1
            with torch.no_grad():
                pending_dynamics, _, *hidden = next_role_prediction(
                    predictor, observation, error, hidden
                )
            previous_dynamic = UnifiedFeatures(
                *(value.detach() for value in dynamic_error.as_tuple())
            )
    return {key: value / frame_count for key, value in totals.items()}, frame_count, gradient_nonzero


def writeback_gate(model, sample):
    image = load_image(sample)
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        observation = model.encode_backbone_features(raw)
        zero = zero_state(observation)
        host = residual_writeback_host_feature(model, raw, zero, tuple(image.shape[-2:]))
        noisy_host = type(host)(raw.c4, raw.c1, tuple(image.shape[-2:]))
        logits = model.decode_from_host_feature(host)
        noisy_logits = model.decode_from_host_feature(noisy_host)
    return float((logits - noisy_logits).abs().max().item())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Error decomposition training requires CUDA")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    paths = make_paths()
    model, predictor = load_role_components(
        paths["static"], paths["adapter"], paths["predictor"], paths["writeback"]
    )
    corrections = build_corrections()
    parameters = list(corrections.parameters())
    if any(parameter.requires_grad is False for parameter in parameters):
        raise RuntimeError("new correction parameters must require gradients")
    if any(parameter.requires_grad for parameter in model.parameters()) or any(
        parameter.requires_grad for parameter in predictor.parameters()
    ):
        raise RuntimeError("frozen components unexpectedly require gradients")
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_groups = sequence_groups(train)
    val_groups = sequence_groups(val)
    if len(train_groups) != EXPECTED_TRAIN_SEQUENCES or len(train.samples) != EXPECTED_TRAIN_FRAMES:
        raise RuntimeError("KITTI-STEP train protocol mismatch")
    identity_difference = writeback_gate(model, train.samples[0])
    if identity_difference > 1e-6:
        raise RuntimeError("Residual writeback identity gate failed")
    disable_dynamic = os.environ.get("PREDIFY_ERROR_DECOMPOSITION_DISABLE_DYNAMIC", "0") == "1"
    output = Path(
        os.environ.get(
            "PREDIFY_ERROR_DECOMPOSITION_OUTPUT_DIR",
            f"/home/lin/experiments/kitti_step_error_decomposition_correction_{os.environ.get('PREDIFY_GIT_REVISION', 'working_tree')}",
        )
    )
    history = []
    best = None
    for epoch in range(1, EPOCHS + 1):
        train_values, train_frames, gradient_nonzero = run_epoch(
            model, predictor, corrections, train_groups, optimizer, True, disable_dynamic
        )
        with torch.no_grad():
            val_values, val_frames, _ = run_epoch(
                model, predictor, corrections, val_groups, None, False, disable_dynamic
            )
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
                    "mode": "error_decomposition",
                    "disable_dynamic": disable_dynamic,
                    "corrections": corrections.state_dict(),
                    "epoch": epoch,
                    "val": val_values,
                },
                output / "best_error_decomposition_correction.pt",
            )
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "kitti_step_error_decomposition_reliability_correction_training",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(output / "best_error_decomposition_correction.pt"),
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
            "loss": "semantic + 0.5 relative_state + 0.25 decomposition + 0.10 reliability",
            "disable_dynamic": disable_dynamic,
            "labels_used_for_training": True,
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
            "trainable_responsibility_passed": True,
            "predictor_history_is_noisy_observation": True,
            "correction_feedback_to_predictor": False,
            "clean_state_used_only_as_target": True,
            "finite": all(
                all(torch.isfinite(torch.tensor(value)) for value in record["train"].values() if isinstance(value, (float, int)))
                and all(torch.isfinite(torch.tensor(value)) for value in record["val"].values() if isinstance(value, (float, int)))
                for record in history
            ),
        },
        "history": history,
        "best_epoch": best["epoch"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
