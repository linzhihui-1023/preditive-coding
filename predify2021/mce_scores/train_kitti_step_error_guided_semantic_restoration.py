"""Stage A: train prediction-error-guided same-frame Z4 semantic restoration.

Only Semantic V2 is trainable. Host, adapters, Dynamics Branch, current
writeback, ErrorState ConvGRU and Temporal Gate remain frozen. Validation uses
the same Blur-Mid/Blur-Max pressure conditions as the prior Host/Semantic Z4
diagnostics so results are directly comparable.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
    sequence_groups,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.kitti_step_persistent_blur import (
    BLUR_KERNEL_SIZE,
    warmup_frame_count,
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
from predify2021.mce_scores.train_kitti_step_convgru_decoupling_quick import (
    diagnostic_blur,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorGuidedSemanticRestorationPredictor,
    HostFeature,
    UnifiedFeatures,
)

SEED = 0
MAX_EPOCHS = 15
PATIENCE = 3
TBPTT_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
TRAIN_CONDITIONS = ("Blur-Mid", "Blur-Max")
TRAIN_PATTERNS = (
    "clean_to_blur",
    "blur_to_clean",
    "clean_continuous",
    "blur_continuous",
)
BLUR_SIGMA = {"Blur-Mid": 2.25, "Blur-Max": 3.0}


def slice_state(state, index):
    return UnifiedFeatures(*(value[index:index + 1] for value in state.as_tuple()))


def detach_semantic(hidden):
    return None if hidden is None else hidden.detach()


def apply_gaussian_blur(image, condition):
    import torchvision.transforms.functional as TF

    sigma = BLUR_SIGMA[condition]
    return TF.gaussian_blur(
        image,
        [BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE],
        [sigma, sigma],
    )


def pattern_uses_blur(pattern, frame_index, sequence_length):
    split = max(1, sequence_length // 2)
    if pattern == "clean_to_blur":
        return frame_index >= split
    if pattern == "blur_to_clean":
        return frame_index < split
    if pattern == "clean_continuous":
        return False
    if pattern == "blur_continuous":
        return True
    raise ValueError(pattern)


def training_assignment(epoch, sequence_index):
    combinations = tuple(
        (condition, pattern)
        for pattern in TRAIN_PATTERNS
        for condition in TRAIN_CONDITIONS
    )
    return combinations[(epoch - 1 + sequence_index) % len(combinations)]


def encode_pair(model, sample, condition, use_blur):
    clean_image = load_image(sample)
    observed_image = apply_gaussian_blur(clean_image, condition) if use_blur else clean_image
    images = torch.cat((clean_image, observed_image), dim=0)
    output_size = tuple(clean_image.shape[-2:])
    with torch.no_grad():
        raw = model.extract_backbone_features(images)
        states = model.encode_backbone_features(raw)
    clean_state = slice_state(states, 0)
    observation = slice_state(states, 1)
    observed_raw = type(raw)(*(value[1:2] for value in raw.as_tuple()))
    return observation, clean_state, observed_raw, output_size


def zero_z4_delta(reference, z4_delta):
    zero = zero_state(reference)
    return UnifiedFeatures(zero.z1, zero.z2, zero.z3, z4_delta)


def decode_z4_writeback(model, raw, observation, restored, output_size):
    delta = zero_z4_delta(observation, restored.z4 - observation.z4)
    host_feature = residual_writeback_host_feature(model, raw, delta, output_size)
    return model.decode_from_host_feature(host_feature)


def bootstrap_dynamics(predictor, observation):
    zero = zero_state(observation)
    with torch.no_grad():
        return predictor.predict_next(observation, zero, None, None)


def smoke_structure_checks(predictor, observation):
    predictor.eval()
    current_prediction = observation
    restored, _, diagnostics = predictor.restore_current(
        observation,
        current_prediction,
        None,
    )
    identity_diff = float((restored.z4 - observation.z4).abs().max().item())
    state_change = float(diagnostics["state_innovation"].abs().max().item())
    dynamics_trainable = sum(
        parameter.numel()
        for name in predictor.DYNAMICS_MODULES
        for parameter in getattr(predictor, name).parameters()
        if parameter.requires_grad
    )
    if identity_diff != 0.0:
        raise RuntimeError(f"Initial clean identity failed: max Z4 diff={identity_diff}")
    if state_change != 0.0:
        raise RuntimeError(f"Zero prediction error changed semantic state: {state_change}")
    if dynamics_trainable != 0:
        raise RuntimeError(f"Dynamics Branch is trainable: {dynamics_trainable} parameters")
    return {
        "initial_restoration_identity_max_abs_diff": identity_diff,
        "zero_error_state_innovation_max_abs": state_change,
        "dynamics_trainable_parameter_count": dynamics_trainable,
        "prediction_error_directly_drives_state_innovation": True,
    }


def train_sequence(
    model,
    predictor,
    samples,
    epoch,
    sequence_index,
    optimizer,
    tbptt_steps,
    max_steps=0,
):
    if len(samples) < 2:
        return {"steps": 0, "loss_sum": 0.0}

    condition, pattern = training_assignment(epoch, sequence_index)
    first_observation, _, _, _ = encode_pair(
        model,
        samples[0],
        condition,
        pattern_uses_blur(pattern, 0, len(samples)),
    )
    pending_prediction, h4_dyn, h1_dyn = bootstrap_dynamics(predictor, first_observation)
    h_sem = predictor.initial_semantic_state()

    steps = min(len(samples) - 1, max_steps) if max_steps else len(samples) - 1
    loss_chunk = []
    loss_sum = 0.0
    trained_steps = 0

    for offset in range(steps):
        frame_index = offset + 1
        observation, clean_state, _, _ = encode_pair(
            model,
            samples[frame_index],
            condition,
            pattern_uses_blur(pattern, frame_index, len(samples)),
        )
        prediction_error = error_state(observation, pending_prediction)
        restored, h_sem, _ = predictor.restore_current(
            observation,
            pending_prediction,
            h_sem,
        )
        loss = F.smooth_l1_loss(restored.z4, clean_state.z4.detach())
        loss_chunk.append(loss)
        loss_sum += float(loss.detach().item())
        trained_steps += 1

        with torch.no_grad():
            pending_prediction, h4_dyn, h1_dyn = predictor.predict_next(
                observation,
                prediction_error,
                h4_dyn,
                h1_dyn,
            )

        if len(loss_chunk) == tbptt_steps or offset + 1 == steps:
            optimizer.zero_grad(set_to_none=True)
            torch.stack(loss_chunk).mean().backward()
            optimizer.step()
            h_sem = detach_semantic(h_sem)
            loss_chunk = []

    return {"steps": trained_steps, "loss_sum": loss_sum}


def train_epoch(model, predictor, groups, epoch, optimizer, tbptt_steps, max_steps=0):
    predictor.train()
    total_steps = 0
    total_loss = 0.0
    for sequence_index, samples in enumerate(groups.values()):
        metrics = train_sequence(
            model,
            predictor,
            samples,
            epoch,
            sequence_index,
            optimizer,
            tbptt_steps,
            max_steps=max_steps,
        )
        total_steps += metrics["steps"]
        total_loss += metrics["loss_sum"]
    if total_steps == 0:
        raise RuntimeError("Training produced zero semantic restoration steps")
    return {
        "step_count": total_steps,
        "z4_smooth_l1": total_loss / total_steps,
    }


def evaluate_condition(model, predictor, groups, condition, max_effective_frames=0):
    blur_confusion = torch.zeros((19, 19), dtype=torch.int64)
    restored_confusion = torch.zeros((19, 19), dtype=torch.int64)
    observation_sse = 0.0
    restored_sse = 0.0
    element_count = 0
    effective_frames = 0

    predictor.eval()
    with torch.inference_mode():
        for samples in groups.values():
            if len(samples) < 2:
                continue
            onset = warmup_frame_count(len(samples))

            first_clean = load_image(samples[0])
            first_observed = diagnostic_blur(first_clean, 0, len(samples), condition)
            first_raw = model.extract_backbone_features(first_observed)
            first_observation = model.encode_backbone_features(first_raw)
            pending_prediction, h4_dyn, h1_dyn = bootstrap_dynamics(predictor, first_observation)
            h_sem = predictor.initial_semantic_state()

            for frame_index in range(1, len(samples)):
                if max_effective_frames and effective_frames >= max_effective_frames:
                    break

                sample = samples[frame_index]
                clean_image = load_image(sample)
                observed_image = diagnostic_blur(
                    clean_image,
                    frame_index,
                    len(samples),
                    condition,
                )
                output_size = tuple(clean_image.shape[-2:])
                images = torch.cat((clean_image, observed_image), dim=0)
                raw = model.extract_backbone_features(images)
                states = model.encode_backbone_features(raw)
                clean_state = slice_state(states, 0)
                observation = slice_state(states, 1)
                observed_raw = type(raw)(*(value[1:2] for value in raw.as_tuple()))

                prediction_error = error_state(observation, pending_prediction)
                restored, h_sem, _ = predictor.restore_current(
                    observation,
                    pending_prediction,
                    h_sem,
                )

                if frame_index >= onset:
                    obs_error = observation.z4.float() - clean_state.z4.float()
                    restored_error = restored.z4.float() - clean_state.z4.float()
                    observation_sse += float(obs_error.square().sum().item())
                    restored_sse += float(restored_error.square().sum().item())
                    element_count += obs_error.numel()

                    mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                    blur_logits = model.decode_from_host_feature(
                        HostFeature(observed_raw.c4, observed_raw.c1, output_size)
                    )
                    restored_logits = decode_z4_writeback(
                        model,
                        observed_raw,
                        observation,
                        restored,
                        output_size,
                    )
                    update_confusion_matrix(
                        blur_confusion,
                        blur_logits.argmax(1).squeeze(0).cpu(),
                        mask,
                    )
                    update_confusion_matrix(
                        restored_confusion,
                        restored_logits.argmax(1).squeeze(0).cpu(),
                        mask,
                    )
                    effective_frames += 1

                pending_prediction, h4_dyn, h1_dyn = predictor.predict_next(
                    observation,
                    prediction_error,
                    h4_dyn,
                    h1_dyn,
                )

            if max_effective_frames and effective_frames >= max_effective_frames:
                break

    elements = max(element_count, 1)
    observation_mse = observation_sse / elements
    restored_mse = restored_sse / elements
    feature_recovery = 1.0 - restored_mse / max(observation_mse, 1e-12)
    blur_miou = float(torch.nanmean(compute_iou(blur_confusion)).item())
    restored_miou = float(torch.nanmean(compute_iou(restored_confusion)).item())
    return {
        "effective_frame_count": effective_frames,
        "observation_mse_to_clean_z4": observation_mse,
        "restored_mse_to_clean_z4": restored_mse,
        "feature_recovery_fraction": feature_recovery,
        "blur_host_mIoU": blur_miou,
        "restored_mIoU_fixed_writeback": restored_miou,
        "mIoU_gain_vs_blur": restored_miou - blur_miou,
    }


def evaluate(model, predictor, groups, max_effective_frames=0):
    by_condition = {
        condition: evaluate_condition(
            model,
            predictor,
            groups,
            condition,
            max_effective_frames=max_effective_frames,
        )
        for condition in TRAIN_CONDITIONS
    }
    return {
        "conditions": by_condition,
        "mean_feature_recovery": sum(
            result["feature_recovery_fraction"] for result in by_condition.values()
        ) / len(by_condition),
        "mean_restored_mIoU": sum(
            result["restored_mIoU_fixed_writeback"] for result in by_condition.values()
        ) / len(by_condition),
        "mean_mIoU_gain_vs_blur": sum(
            result["mIoU_gain_vs_blur"] for result in by_condition.values()
        ) / len(by_condition),
    }


def is_better(validation, best):
    if best is None:
        return True
    if validation["mean_feature_recovery"] > best["val"]["mean_feature_recovery"]:
        return True
    return (
        validation["mean_feature_recovery"] == best["val"]["mean_feature_recovery"]
        and validation["mean_restored_mIoU"] > best["val"]["mean_restored_mIoU"]
    )


def load_model_and_predictor(args):
    model, source_predictor = load_components(
        STATIC_CHECKPOINT_DEFAULT,
        ADAPTER_CHECKPOINT_DEFAULT,
        args.dynamics_checkpoint,
        WRITEBACK_CHECKPOINT_DEFAULT,
    )
    predictor = ErrorGuidedSemanticRestorationPredictor().cuda()
    predictor.load_dynamics_from_role_separated_state_dict(source_predictor.state_dict())
    predictor.freeze_dynamics()
    return model, predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument(
        "--output",
        default="/home/lin/predify/experiments/kitti_step_error_guided_semantic_restoration",
    )
    parser.add_argument(
        "--dynamics-checkpoint",
        default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    )
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--tbptt-steps", type=int, default=TBPTT_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Stage A semantic restoration requires CUDA")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, predictor = load_model_and_predictor(args)
    model.requires_grad_(False)
    model.eval()

    semantic_parameters = [
        parameter for parameter in predictor.semantic_parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        semantic_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_groups = sequence_groups(
        KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    )
    val_groups = sequence_groups(
        KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    )

    epochs = args.epochs
    train_max_steps = 0
    val_max_frames = 0
    output = Path(args.output)
    if args.smoke:
        train_groups = dict(list(train_groups.items())[:1])
        val_groups = dict(list(val_groups.items())[:1])
        epochs = 1
        train_max_steps = 16
        val_max_frames = 32
        output = output / "smoke"
    output.mkdir(parents=True, exist_ok=True)

    first_samples = next(iter(train_groups.values()))
    first_observation, _, _, _ = encode_pair(
        model,
        first_samples[0],
        "Blur-Mid",
        False,
    )
    smoke_checks = smoke_structure_checks(predictor, first_observation)

    checkpoint = output / "best_error_guided_semantic_restoration.pt"
    history = []
    best = None
    stale = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model,
            predictor,
            train_groups,
            epoch,
            optimizer,
            args.tbptt_steps,
            max_steps=train_max_steps,
        )
        val_metrics = evaluate(
            model,
            predictor,
            val_groups,
            max_effective_frames=val_max_frames,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        if is_better(val_metrics, best):
            best = row
            stale = 0
            torch.save(
                {
                    "model_state_dict": predictor.state_dict(),
                    "source_dynamics_checkpoint": args.dynamics_checkpoint,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    best_conditions = best["val"]["conditions"]
    go = all(
        best_conditions[condition]["feature_recovery_fraction"] > 0.0
        and best_conditions[condition]["mIoU_gain_vs_blur"] > 0.0
        for condition in TRAIN_CONDITIONS
    )

    summary = {
        "experiment": "kitti_step_error_guided_semantic_restoration_stage_a",
        "checkpoint": str(checkpoint),
        "source_dynamics_checkpoint": args.dynamics_checkpoint,
        "parameters_updated": "Semantic V2 only",
        "trainable_parameter_count": sum(p.numel() for p in semantic_parameters),
        "dynamics_trainable_parameter_count": 0,
        "config": {
            "max_epochs": epochs,
            "patience": args.patience,
            "tbptt_steps": args.tbptt_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss": "SmoothL1(restored Z4_t, clean Z4_t)",
            "time_alignment": "predict t -> observe t -> error t -> restore t -> predict t+1",
            "training_patterns": list(TRAIN_PATTERNS),
            "training_conditions": {
                "Blur-Mid": "sigma=2.25",
                "Blur-Max": "sigma=3.0",
            },
            "validation": "persistent diagnostic Blur-Mid/Blur-Max, post-warmup frames",
            "selection": "highest mean feature recovery; mean fixed-writeback mIoU breaks exact ties",
        },
        "smoke_checks": smoke_checks,
        "history": history,
        "best": best,
        "decision": {
            "STAGE_A": "GO" if go else "NO-GO",
            "criterion": (
                "Blur-Mid and Blur-Max both require feature recovery > 0 and "
                "fixed-writeback mIoU gain vs Blur Host > 0"
            ),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"best": best, "decision": summary["decision"]}, indent=2), flush=True)
    print(f"results={output}", flush=True)


if __name__ == "__main__":
    main()
