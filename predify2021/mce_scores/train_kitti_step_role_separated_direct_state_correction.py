import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.role_separated_direct_state_correction import (
    ALPHA,
    BETA,
    add_frame_noise,
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
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.role_separated_dynamic_error_correction import residual_writeback_host_feature
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature


SEED = 0
SIGMA = 0.10
EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EXPECTED_TRAIN_SEQUENCES = 12
EXPECTED_TRAIN_FRAMES = 5027


def semantic_loss(model, raw_features, observation, posterior, target_mask, output_size):
    delta = type(posterior)(
        posterior.z1 - observation.z1,
        torch.zeros_like(posterior.z2),
        torch.zeros_like(posterior.z3),
        posterior.z4 - observation.z4,
    )
    corrected = residual_writeback_host_feature(model, raw_features, delta, output_size)
    logits = model.decode_from_host_feature(corrected)
    return F.cross_entropy(
        logits,
        target_mask.unsqueeze(0).cuda(),
        ignore_index=255,
    )


def consistency_gate(model, sample):
    clean_image = load_image(sample)
    noisy_image = add_frame_noise(clean_image, SIGMA)
    with torch.no_grad():
        raw_features = model.extract_backbone_features(noisy_image)
        observation = model.encode_backbone_features(raw_features)
        delta = zero_state(observation)
        training_host = residual_writeback_host_feature(model, raw_features, delta, tuple(clean_image.shape[-2:]))
        evaluation_host = residual_writeback_host_feature(model, raw_features, delta, tuple(clean_image.shape[-2:]))
        training_logits = model.decode_from_host_feature(training_host)
        evaluation_logits = model.decode_from_host_feature(evaluation_host)
        noisy_logits = model.decode_from_host_feature(HostFeature(raw_features.c4, raw_features.c1, tuple(clean_image.shape[-2:])))
    path_difference = float((training_logits - evaluation_logits).abs().max().item())
    identity_difference = float((training_logits - noisy_logits).abs().max().item())
    return {
        "training_validation_path_max_abs_logit_difference": path_difference,
        "training_validation_path_consistent": path_difference <= 1e-6,
        "identity_max_abs_logit_difference": identity_difference,
        "identity_passed": identity_difference <= 1e-6,
    }


def run_epoch(model, predictor, corrections, groups, optimizer, mode, training):
    corrections.train(training)
    totals = {"semantic_cross_entropy": 0.0, "raw_state_mse_z1": 0.0, "raw_state_mse_z4": 0.0, "relative_state_loss": 0.0, "total_loss": 0.0}
    frame_count = 0
    gradient_nonzero = None
    for samples in groups.values():
        hidden = (None, None, None, None)
        dynamic_error = None
        pending_dynamics = None
        for frame_index, sample in enumerate(samples):
            clean_image = load_image(sample)
            noisy_image = add_frame_noise(clean_image, SIGMA)
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
                dynamic_error = update_dynamic_error(error, dynamic_error)
            posterior, _ = direct_posterior(observation, error, dynamic_error, corrections)
            z1, z4, state_loss = raw_state_mse(posterior, clean_state)
            relative_loss = relative_state_loss(posterior, clean_state, observation)
            if mode == "state":
                semantic_value = torch.zeros((), device=state_loss.device)
                total_loss = state_loss
            else:
                semantic_value = semantic_loss(
                    model, noisy_raw, observation, posterior,
                    semantic_mask_from_panoptic_png(sample["mask_path"]),
                    tuple(clean_image.shape[-2:]),
                )
                total_loss = semantic_value + 0.5 * relative_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if gradient_nonzero is None:
                    gradient_nonzero = any(
                        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                        for parameter in corrections.parameters()
                    )
                optimizer.step()
            totals["semantic_cross_entropy"] += semantic_value.detach().item()
            totals["raw_state_mse_z1"] += z1.detach().item()
            totals["raw_state_mse_z4"] += z4.detach().item()
            totals["relative_state_loss"] += relative_loss.detach().item()
            totals["total_loss"] += total_loss.detach().item()
            frame_count += 1
            with torch.no_grad():
                pending_dynamics, _, *hidden = next_role_prediction(
                    predictor, observation, error, hidden
                )
                dynamic_error = type(dynamic_error)(*(value.detach() for value in dynamic_error.as_tuple()))
    return {key: value / frame_count for key, value in totals.items()}, frame_count, gradient_nonzero


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Role-separated direct correction training requires CUDA")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    mode = os.environ.get("PREDIFY_ROLE_SEPARATED_DIRECT_CORRECTION_MODE", "state")
    if mode not in ("state", "semantic_state"):
        raise RuntimeError("mode must be state or semantic_state")
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
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
    gates = consistency_gate(model, train.samples[0])
    if not gates["training_validation_path_consistent"] or not gates["identity_passed"]:
        raise RuntimeError("Residual writeback consistency or identity gate failed")
    history = []
    best = None
    for epoch in range(1, EPOCHS + 1):
        train_values, train_frames, gradient_nonzero = run_epoch(model, predictor, corrections, train_groups, optimizer, mode, True)
        with torch.no_grad():
            val_values, val_frames, _ = run_epoch(model, predictor, corrections, val_groups, None, mode, False)
        record = {"epoch": epoch, "train": {**train_values, "effective_frame_count": train_frames}, "val": {**val_values, "effective_frame_count": val_frames}, "correction_gradient_nonzero": gradient_nonzero}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if best is None or val_values["total_loss"] < best["val"]["total_loss"]:
            best = record
            output = Path(os.environ.get("PREDIFY_ROLE_SEPARATED_DIRECT_CORRECTION_OUTPUT_DIR", f"/home/lin/predify/experiments/kitti_step_role_separated_direct_{mode}"))
            output.mkdir(parents=True, exist_ok=True)
            torch.save({"mode": mode, "corrections": {str(index): corrections[position].state_dict() for position, index in enumerate((0, 3))}, "epoch": epoch, "val": val_values}, output / f"best_role_separated_direct_{mode}.pt")
    output = Path(os.environ.get("PREDIFY_ROLE_SEPARATED_DIRECT_CORRECTION_OUTPUT_DIR", f"/home/lin/predify/experiments/kitti_step_role_separated_direct_{mode}"))
    output.mkdir(parents=True, exist_ok=True)
    summary = {"experiment": "kitti_step_role_separated_direct_state_correction_training", "mode": mode, "git_revision": os.environ.get("PREDIFY_GIT_REVISION"), "checkpoint": str(output / f"best_role_separated_direct_{mode}.pt"), "base_checkpoints": {key: str(value) for key, value in paths.items()}, "config": {"epochs": EPOCHS, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "seed": SEED, "gaussian_noise_sigma": SIGMA, "alpha": ALPHA, "beta": BETA, "labels_used_for_training": mode == "semantic_state"}, "dataset": {"train_sequence_count": len(train_groups), "val_sequence_count": len(val_groups), "train_frame_count": len(train.samples), "val_frame_count": len(val.samples)}, "trainable_parameter_count": sum(parameter.numel() for parameter in parameters), "gates": {**gates, "correction_gradient_nonzero": all(record["correction_gradient_nonzero"] for record in history)}, "history": history, "best_epoch": best["epoch"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
