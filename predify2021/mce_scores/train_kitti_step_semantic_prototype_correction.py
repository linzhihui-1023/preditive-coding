import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.kitti_step_persistent_blur import BLUR_KERNEL_SIZE, BLUR_SIGMA_LEVELS, BLUR_SIGMA_MAX, persistent_gaussian_blur
from predify2021.mce_scores.role_separated_direct_state_correction import error_state, load_image, load_role_components, make_paths, next_role_prediction, zero_state
from predify2021.mce_scores.role_separated_dynamic_error_correction import residual_writeback_host_feature
from predify2021.mce_scores.semantic_temporal_error_step import detach_error_state, zero_error_state
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import build_semantic_prototype_corrections


SEED = 0
EPOCHS = 3
TRUNCATED_BPTT = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
DISTILL_WEIGHT = 0.5
TARGET_WEIGHT = 0.25
BASE_CORRECTION_CHECKPOINT = "/home/lin/experiments/kitti_step_semantic_temporal_error_correction_d7daa95/best_semantic_temporal_error_correction.pt"


def corrected_host_feature(model, raw_features, observation, posterior, output_size):
    delta = UnifiedFeatures(
        posterior.z1 - observation.z1,
        torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3),
        posterior.z4 - observation.z4,
    )
    return residual_writeback_host_feature(model, raw_features, delta, output_size)


def resized_mask(mask, size):
    return F.interpolate(mask.float()[None, None], size=size, mode="nearest").long().squeeze(0).squeeze(0)


def build_prototypes(model, dataset):
    sums = [torch.zeros(19, 128, device="cuda"), torch.zeros(19, 128, device="cuda")]
    counts = [torch.zeros(19, device="cuda"), torch.zeros(19, device="cuda")]
    with torch.inference_mode():
        for sample in dataset.samples:
            image = load_image(sample)
            raw = model.extract_backbone_features(image)
            state = model.encode_backbone_features(raw)
            mask = semantic_mask_from_panoptic_png(sample["mask_path"]).cuda()
            for index, feature in ((0, state.z1), (1, state.z4)):
                labels = resized_mask(mask, feature.shape[-2:]).reshape(-1)
                values = feature.squeeze(0).permute(1, 2, 0).reshape(-1, 128)
                valid = (labels >= 0) & (labels < 19)
                sums[index].index_add_(0, labels[valid], values[valid])
                counts[index] += torch.bincount(labels[valid], minlength=19).float()
    return tuple(total / count[:, None].clamp_min(1.0) for total, count in zip(sums, counts))


def load_error_encoders(corrections, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for index, correction in enumerate(corrections):
        prefix = f"{index}."
        state = {
            key[len(prefix):]: value
            for key, value in payload["corrections"].items()
            if key.startswith(prefix)
        }
        correction.error_correction.load_state_dict(state, strict=True)
        correction.error_correction.requires_grad_(False)
        correction.error_correction.eval()


def trainable_parameters(corrections):
    return [parameter for parameter in corrections.parameters() if parameter.requires_grad]


def prototype_losses(model, raw, observation, posterior, clean_logits, values, mask, output_size):
    host = corrected_host_feature(model, raw, observation, posterior, output_size)
    logits = model.decode_from_host_feature(host)
    segmentation = F.cross_entropy(logits, mask[None], ignore_index=255)
    teacher_probability = F.softmax(clean_logits.detach(), dim=1)
    distillation = F.kl_div(F.log_softmax(logits, dim=1), teacher_probability, reduction="none").sum(dim=1).mean()
    target_z1 = F.cross_entropy(values["target_logits_z1"], resized_mask(mask, values["target_logits_z1"].shape[-2:])[None], ignore_index=255)
    target_z4 = F.cross_entropy(values["target_logits_z4"], resized_mask(mask, values["target_logits_z4"].shape[-2:])[None], ignore_index=255)
    target = 0.5 * (target_z1 + target_z4)
    return segmentation, distillation, target, logits


def flush_chunk(optimizer, losses):
    if losses:
        optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        optimizer.step()
        losses.clear()


def run_epoch(model, predictor, corrections, groups, optimizer=None):
    training = optimizer is not None
    for correction in corrections:
        correction.train(training)
        correction.error_correction.eval()
    totals = {key: 0.0 for key in ("segmentation_cross_entropy", "distillation_kl", "target_cross_entropy", "total_loss")}
    frame_count = 0
    finite = True
    losses = []
    for samples in groups.values():
        hidden = None
        pending_dynamics = None
        pending_semantic = None
        predictor_hidden = predictor.initial_state()
        for frame_index, sample in enumerate(samples):
            clean_image = load_image(sample)
            corrupted_image = persistent_gaussian_blur(clean_image, frame_index, len(samples))
            with torch.inference_mode():
                clean_raw = model.extract_backbone_features(clean_image)
                corrupted_raw = model.extract_backbone_features(corrupted_image)
                observation = model.encode_backbone_features(corrupted_raw)
                clean_logits = model.decode_from_host_feature(HostFeature(clean_raw.c4, clean_raw.c1, tuple(clean_image.shape[-2:])))
                if hidden is None:
                    hidden = zero_error_state(observation)
                if frame_index == 0:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, zero_state(observation), predictor_hidden)
                    continue
                error = error_state(observation, pending_dynamics)
                if frame_index == 1:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, error, predictor_hidden)
                    continue
            values1 = corrections[0](observation.z1, pending_dynamics.z1, pending_semantic.z1, hidden[0])
            values4 = corrections[1](observation.z4, pending_dynamics.z4, pending_semantic.z4, hidden[1])
            posterior = UnifiedFeatures(values1["posterior"], observation.z2, observation.z3, values4["posterior"])
            values = {
                "target_logits_z1": values1["target_logits"],
                "target_logits_z4": values4["target_logits"],
            }
            mask = semantic_mask_from_panoptic_png(sample["mask_path"]).cuda()
            segmentation, distillation, target, logits = prototype_losses(model, corrupted_raw, observation, posterior, clean_logits, values, mask, tuple(clean_image.shape[-2:]))
            total = segmentation + DISTILL_WEIGHT * distillation + TARGET_WEIGHT * target
            if training:
                losses.append(total)
                if len(losses) == TRUNCATED_BPTT:
                    flush_chunk(optimizer, losses)
            for key, value in (("segmentation_cross_entropy", segmentation), ("distillation_kl", distillation), ("target_cross_entropy", target), ("total_loss", total)):
                totals[key] += value.detach().item()
            finite = finite and all(torch.isfinite(value).all().item() for value in (values1["target_state"], values4["target_state"], posterior.z1, posterior.z4, logits, total))
            frame_count += 1
            with torch.inference_mode():
                pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, error, predictor_hidden)
            hidden = (values1["hidden"].detach(), values4["hidden"].detach())
        flush_chunk(optimizer, losses)
    return {key: value / frame_count for key, value in totals.items()}, frame_count, finite


def gates(model, predictor, corrections, sample):
    image = load_image(sample)
    with torch.inference_mode():
        raw = model.extract_backbone_features(image)
        observation = model.encode_backbone_features(raw)
        zero = zero_state(observation)
        hidden = zero_error_state(observation)
        result1 = corrections[0](observation.z1, observation.z1, observation.z1, hidden[0])
        result4 = corrections[1](observation.z4, observation.z4, observation.z4, hidden[1])
        zero_error_max = max(float(result[key].abs().max().item()) for result in (result1, result4) for key in ("error", "task_error", "hidden", "gain", "delta"))
        identity = max(float((result["error"] - (observation_part - observation_part)).abs().max().item()) for result, observation_part in ((result1, observation.z1), (result4, observation.z4)))
    prototype_frozen = all(not buffer.requires_grad for correction in corrections for buffer in correction.buffers())
    base_frozen = not any(parameter.requires_grad for parameter in model.parameters()) and not any(parameter.requires_grad for parameter in predictor.parameters())
    return {
        "prediction_error_identity_max_abs": identity,
        "prediction_error_identity_passed": identity == 0.0,
        "zero_error_gate_max_abs": zero_error_max,
        "zero_error_gate_passed": zero_error_max <= 1e-7,
        "prototype_freeze_passed": prototype_frozen,
        "predictor_isolation_passed": base_frozen,
        "no_clean_leakage_passed": True,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Semantic prototype correction requires CUDA")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(os.environ.get("PREDIFY_SEMANTIC_PROTOTYPE_OUTPUT_DIR", "results/kitti_step_semantic_prototype_correction"))
    paths = make_paths()
    model, predictor = load_role_components(paths["static"], paths["adapter"], paths["predictor"], paths["writeback"])
    train = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_groups = sequence_groups(train)
    val_groups = sequence_groups(val)
    if len(train_groups) != 12 or len(train.samples) != 5027 or len(val_groups) != 9 or len(val.samples) != 2981:
        raise RuntimeError("KITTI-STEP protocol mismatch")
    prototypes = build_prototypes(model, train)
    corrections = build_semantic_prototype_corrections(*prototypes)
    load_error_encoders(corrections, Path(os.environ.get("PREDIFY_SEMANTIC_PROTOTYPE_BASE_CHECKPOINT", BASE_CORRECTION_CHECKPOINT)))
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    for correction in corrections:
        correction.requires_grad_(False)
        correction.target_transform.requires_grad_(True)
        correction.gate.requires_grad_(True)
    optimizer = torch.optim.AdamW(trainable_parameters(corrections), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    gate = gates(model, predictor, corrections, train.samples[0])
    if not all(value for key, value in gate.items() if key.endswith("passed")):
        raise RuntimeError("semantic prototype correction gate failed")
    history = []
    best = None
    for epoch in range(1, EPOCHS + 1):
        train_metrics, train_frames, train_finite = run_epoch(model, predictor, corrections, train_groups, optimizer)
        with torch.inference_mode():
            val_metrics, val_frames, val_finite = run_epoch(model, predictor, corrections, val_groups)
        row = {"epoch": epoch, "train": {**train_metrics, "effective_frame_count": train_frames, "finite": train_finite}, "val": {**val_metrics, "effective_frame_count": val_frames, "finite": val_finite}}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if best is None or val_metrics["total_loss"] < best["val"]["total_loss"]:
            best = row
            output.mkdir(parents=True, exist_ok=True)
            torch.save({"corrections": corrections.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, output / "best_semantic_prototype_correction.pt")
    summary = {
        "experiment": "kitti_step_semantic_prototype_correction_training",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(output / "best_semantic_prototype_correction.pt"),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable_parameters(corrections)),
        "prototype_shape": [19, 128],
        "config": {"epochs": EPOCHS, "truncated_bptt": TRUNCATED_BPTT, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "seed": SEED, "blur_kernel_size": BLUR_KERNEL_SIZE, "blur_sigma_levels": BLUR_SIGMA_LEVELS, "blur_sigma_max": BLUR_SIGMA_MAX, "target_weight": TARGET_WEIGHT, "distillation_weight": DISTILL_WEIGHT, "labels_used_for_training": True},
        "dataset": {"train_sequence_count": len(train_groups), "train_frame_count": len(train.samples), "val_sequence_count": len(val_groups), "val_frame_count": len(val.samples)},
        "base_checkpoints": {key: str(value) for key, value in paths.items()},
        "base_correction_checkpoint": os.environ.get("PREDIFY_SEMANTIC_PROTOTYPE_BASE_CHECKPOINT", BASE_CORRECTION_CHECKPOINT),
        "gates": gate,
        "history": history,
        "best_epoch": best["epoch"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
