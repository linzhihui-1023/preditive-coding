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
    error_state,
    load_image,
    load_role_components,
    make_paths,
    next_role_prediction,
    zero_state,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import residual_writeback_host_feature
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import build_stcn_memory_corrections
from predify2021.model_factory.deeplabv3plus_resnet50.space_time_memory_reader import detach_memory


SEED = 0
EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
DISTILL_WEIGHT = 0.5
TRUNCATED_BPTT = 4
BASE_CORRECTION_CHECKPOINT = "/home/lin/experiments/kitti_step_semantic_temporal_error_correction_d7daa95/best_semantic_temporal_error_correction.pt"


def load_base_corrections(corrections, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["corrections"]
    for index, correction in enumerate(corrections):
        prefix = f"{index}."
        correction.base.load_state_dict(
            {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)},
            strict=True,
        )
        correction.base.requires_grad_(False)
        correction.base.eval()
    return payload


def posterior_host(model, raw_features, observation, posterior, output_size):
    delta = UnifiedFeatures(
        posterior.z1 - observation.z1,
        torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3),
        posterior.z4 - observation.z4,
    )
    return residual_writeback_host_feature(model, raw_features, delta, output_size)


def loss_values(model, raw_features, observation, posterior, clean_logits, mask, output_size):
    host = posterior_host(model, raw_features, observation, posterior, output_size)
    logits = model.decode_from_host_feature(host)
    target = mask.unsqueeze(0).cuda()
    cross_entropy = F.cross_entropy(logits, target, ignore_index=255)
    teacher = F.softmax(clean_logits.detach(), dim=1)
    distillation = F.kl_div(
        F.log_softmax(logits, dim=1), teacher, reduction="none"
    ).sum(dim=1).mean()
    return cross_entropy, distillation, logits


def train_epoch(model, predictor, corrections, groups, optimizer):
    corrections.train()
    for correction in corrections:
        correction.base.eval()
    totals = {"cross_entropy": 0.0, "distillation_kl": 0.0, "total_loss": 0.0}
    frame_count = 0
    pending_losses = []
    finite = True

    for samples in groups.values():
        correction_hidden = (None, None)
        predictor_hidden = predictor.initial_state()
        memory = ()
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
                if frame_index == 0:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor, observation, zero_state(observation), predictor_hidden
                    )
                    continue
                error = error_state(observation, pending_dynamics)
                if frame_index == 1:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor, observation, error, predictor_hidden
                    )
                    continue

            _, posterior_z1, hidden_z1, _, values_z1 = corrections[0](
                observation.z1, pending_dynamics.z1, pending_semantic.z1, correction_hidden[0]
            )
            _, posterior_z4, hidden_z4, memory, values_z4 = corrections[1](
                observation.z4, pending_dynamics.z4, pending_semantic.z4, correction_hidden[1], memory
            )
            posterior = UnifiedFeatures(posterior_z1, observation.z2, observation.z3, posterior_z4)
            mask = semantic_mask_from_panoptic_png(sample["mask_path"])
            cross_entropy, distillation, logits = loss_values(
                model,
                corrupted_raw,
                observation,
                posterior,
                clean_logits,
                mask,
                tuple(clean_image.shape[-2:]),
            )
            total = cross_entropy + DISTILL_WEIGHT * distillation
            pending_losses.append(total)
            if len(pending_losses) == TRUNCATED_BPTT:
                optimizer.zero_grad(set_to_none=True)
                torch.stack(pending_losses).mean().backward()
                optimizer.step()
                pending_losses.clear()
                correction_hidden = (hidden_z1.detach(), hidden_z4.detach())
                memory = detach_memory(memory)
            else:
                correction_hidden = (hidden_z1, hidden_z4)
            values = (error.z1, error.z4, posterior.z1, posterior.z4, logits, total, clean_state.z1, clean_state.z4, values_z1["gain"], values_z4["gain"])
            finite = finite and all(torch.isfinite(value).all().item() for value in values)
            for key, value in (("cross_entropy", cross_entropy), ("distillation_kl", distillation), ("total_loss", total)):
                totals[key] += value.detach().item()
            frame_count += 1
            with torch.no_grad():
                pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                    predictor, observation, error, predictor_hidden
                )
        if pending_losses:
            optimizer.zero_grad(set_to_none=True)
            torch.stack(pending_losses).mean().backward()
            optimizer.step()
            pending_losses.clear()
        memory = detach_memory(memory)
        correction_hidden = tuple(value.detach() for value in correction_hidden if value is not None) if correction_hidden[0] is not None else (None, None)
    return {key: value / frame_count for key, value in totals.items()}, frame_count, finite


def validate_epoch(model, predictor, corrections, groups):
    corrections.eval()
    totals = {"cross_entropy": 0.0, "distillation_kl": 0.0, "total_loss": 0.0}
    frame_count = 0
    finite = True
    with torch.no_grad():
        for samples in groups.values():
            correction_hidden = (None, None)
            predictor_hidden = predictor.initial_state()
            memory = ()
            pending_dynamics = None
            pending_semantic = None
            for frame_index, sample in enumerate(samples):
                clean_image = load_image(sample)
                corrupted_image = persistent_gaussian_blur(clean_image, frame_index, len(samples))
                clean_raw = model.extract_backbone_features(clean_image)
                corrupted_raw = model.extract_backbone_features(corrupted_image)
                observation = model.encode_backbone_features(corrupted_raw)
                clean_logits = model.decode_from_host_feature(HostFeature(clean_raw.c4, clean_raw.c1, tuple(clean_image.shape[-2:])))
                if frame_index == 0:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, zero_state(observation), predictor_hidden)
                    continue
                error = error_state(observation, pending_dynamics)
                if frame_index == 1:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, error, predictor_hidden)
                    continue
                _, posterior_z1, hidden_z1, _, values_z1 = corrections[0](observation.z1, pending_dynamics.z1, pending_semantic.z1, correction_hidden[0])
                _, posterior_z4, hidden_z4, memory, values_z4 = corrections[1](observation.z4, pending_dynamics.z4, pending_semantic.z4, correction_hidden[1], memory)
                posterior = UnifiedFeatures(posterior_z1, observation.z2, observation.z3, posterior_z4)
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                cross_entropy, distillation, logits = loss_values(model, corrupted_raw, observation, posterior, clean_logits, mask, tuple(clean_image.shape[-2:]))
                total = cross_entropy + DISTILL_WEIGHT * distillation
                for key, value in (("cross_entropy", cross_entropy), ("distillation_kl", distillation), ("total_loss", total)):
                    totals[key] += value.item()
                finite = finite and all(torch.isfinite(value).all().item() for value in (error.z1, error.z4, posterior.z1, posterior.z4, logits, total, values_z1["gain"], values_z4["gain"]))
                frame_count += 1
                correction_hidden = (hidden_z1, hidden_z4)
                pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, error, predictor_hidden)
            memory = detach_memory(memory)
    return {key: value / frame_count for key, value in totals.items()}, frame_count, finite


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("STCN memory correction training requires CUDA")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(os.environ.get("PREDIFY_STCN_MEMORY_OUTPUT_DIR", "results/kitti_step_stcn_memory_correction"))
    base_checkpoint = os.environ.get("PREDIFY_STCN_BASE_CORRECTION_CHECKPOINT", BASE_CORRECTION_CHECKPOINT)
    paths = make_paths()
    model, predictor = load_role_components(paths["static"], paths["adapter"], paths["predictor"], paths["writeback"])
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    corrections = build_stcn_memory_corrections()
    load_base_corrections(corrections, base_checkpoint)
    corrections.requires_grad_(True)
    for correction in corrections:
        correction.base.requires_grad_(False)
        correction.base.eval()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in corrections.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    train = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_groups = sequence_groups(train)
    val_groups = sequence_groups(val)
    if len(train_groups) != 12 or len(train.samples) != 5027 or len(val_groups) != 9 or len(val.samples) != 2981:
        raise RuntimeError("KITTI-STEP protocol mismatch")
    trainable_count = sum(parameter.numel() for parameter in corrections.parameters() if parameter.requires_grad)
    if trainable_count != 8321:
        raise RuntimeError(f"unexpected trainable parameter count: {trainable_count}")
    history = []
    best = None
    for epoch in range(1, EPOCHS + 1):
        train_metrics, train_frames, train_finite = train_epoch(model, predictor, corrections, train_groups, optimizer)
        val_metrics, val_frames, val_finite = validate_epoch(model, predictor, corrections, val_groups)
        row = {"epoch": epoch, "train": {**train_metrics, "effective_frame_count": train_frames, "finite": train_finite}, "val": {**val_metrics, "effective_frame_count": val_frames, "finite": val_finite}}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if best is None or val_metrics["total_loss"] < best["val"]["total_loss"]:
            best = row
            output.mkdir(parents=True, exist_ok=True)
            torch.save({"corrections": corrections.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, output / "best_stcn_memory_correction.pt")
    summary = {
        "experiment": "kitti_step_stcn_memory_correction_training",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(output / "best_stcn_memory_correction.pt"),
        "base_correction_checkpoint": str(base_checkpoint),
        "trainable_parameter_count": trainable_count,
        "config": {"seed": SEED, "epochs": EPOCHS, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "distillation_weight": DISTILL_WEIGHT, "truncated_bptt": TRUNCATED_BPTT, "blur_kernel_size": BLUR_KERNEL_SIZE, "blur_sigma_levels": BLUR_SIGMA_LEVELS, "blur_sigma_max": BLUR_SIGMA_MAX, "memory_size": 4, "key_channels": 64},
        "dataset": {"train_sequence_count": len(train_groups), "train_frame_count": len(train.samples), "val_sequence_count": len(val_groups), "val_frame_count": len(val.samples)},
        "base_checkpoints": {key: str(value) for key, value in paths.items()},
        "history": history,
        "best_epoch": best["epoch"],
        "finite": all(row["train"]["finite"] and row["val"]["finite"] for row in history),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
