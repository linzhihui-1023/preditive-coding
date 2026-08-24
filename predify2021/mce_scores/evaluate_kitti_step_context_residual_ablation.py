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
from predify2021.mce_scores.evaluate_kitti_step_adaptive_dynamic_correction import (
    detach_state,
)
from predify2021.mce_scores.evaluate_kitti_step_closed_loop_dynamic_correction import (
    add_frame_noise,
    posterior_state,
)
from predify2021.mce_scores.evaluate_kitti_step_context_residual_correction import (
    EPOCHS,
    LEARNING_RATE,
    SIGMA,
    WEIGHT_DECAY,
    legacy_gains,
    load_components,
    posterior_mse,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    encode_image,
    load_image,
    predict_current,
    sequence_groups,
    update_dynamic_error,
)
from predify2021.mce_scores.evaluate_kitti_step_error_correction import (
    corrected_host_feature,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.context_residual_correction import (
    ContextResidualCorrection,
)


SEED = 0
STATE_BATCH_SIZE = 4
VARIANTS = (
    ("GAIN_ONLY", (False, False, False, False, True)),
    ("GAIN_DYNAMIC", (False, False, False, True, True)),
    ("GAIN_DYNAMIC_INSTANT", (False, False, True, True, True)),
    ("FULL_CONTEXT", (True, True, True, True, True)),
)


def reset_rng():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def masked_context_inputs(predicted, observation, error, dynamic_error, gain, mask):
    values = (predicted, observation, error, dynamic_error, gain)
    return tuple(value if enabled else torch.zeros_like(value)
                 for value, enabled in zip(values, mask))


def context_step(predicted, observation, error, dynamic_error, gain, context, mask):
    inputs = masked_context_inputs(
        predicted.z1,
        observation.z1,
        error.z1,
        dynamic_error.z1,
        gain[0],
        mask,
    )
    residual_z1 = context[0](*inputs)
    inputs = masked_context_inputs(
        predicted.z4,
        observation.z4,
        error.z4,
        dynamic_error.z4,
        gain[1],
        mask,
    )
    residual_z4 = context[1](*inputs)
    residuals = (residual_z1, residual_z4)
    new_gains = tuple(
        torch.sigmoid(
            torch.logit(old_gain.clamp(1e-6, 1.0 - 1e-6)) + residual
        )
        for old_gain, residual in zip(gain, residuals)
    )
    posterior = UnifiedFeatures(
        predicted.z1 + new_gains[0] * error.z1,
        predicted.z2,
        predicted.z3,
        predicted.z4 + new_gains[1] * error.z4,
    )
    posterior = UnifiedFeatures(
        posterior.z1,
        observation.z2,
        observation.z3,
        posterior.z4,
    )
    return posterior, new_gains, residuals


def split_state(state, index):
    return UnifiedFeatures(*(value[index:index + 1] for value in state.as_tuple()))


def encode_with_raw_features(model, image):
    with torch.no_grad():
        raw_features = model.extract_backbone_features(image)
        state = model.encode_backbone_features(raw_features)
    return state, raw_features


def encoded_sequence_states(model, samples):
    for start in range(0, len(samples), STATE_BATCH_SIZE):
        batch_samples = samples[start:start + STATE_BATCH_SIZE]
        clean_batch = torch.cat([load_image(sample) for sample in batch_samples], dim=0)
        noisy_batch = add_frame_noise(clean_batch, SIGMA)
        with torch.no_grad():
            clean_states = model.encode_backbone_features(
                model.extract_backbone_features(clean_batch)
            )
            noisy_states = model.encode_backbone_features(
                model.extract_backbone_features(noisy_batch)
            )
        yield [
            (split_state(clean_states, index), split_state(noisy_states, index))
            for index in range(len(batch_samples))
        ]


def run_epoch(model, predictor, legacy, context, groups, optimizer, training, mask):
    model.eval()
    predictor.eval()
    legacy.eval()
    context.train(training)
    total_loss = 0.0
    frame_count = 0
    with torch.set_grad_enabled(training):
        for samples in groups.values():
            previous_previous = None
            previous = None
            dynamic_error = None
            for state_batch in encoded_sequence_states(model, samples):
                for clean_state, observation in state_batch:
                    if previous is None:
                        previous = observation
                        continue
                    if previous_previous is None:
                        previous_previous = previous
                        previous = observation
                        continue
                    with torch.no_grad():
                        predicted, error = predict_current(
                            predictor, previous_previous, previous, observation
                        )
                        dynamic_error = update_dynamic_error(error, dynamic_error)
                        gain = legacy_gains(dynamic_error, legacy)
                    posterior, _, _ = context_step(
                        predicted, observation, error, dynamic_error, gain, context, mask
                    )
                    loss = posterior_mse(posterior, clean_state)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    total_loss += loss.detach().item()
                    frame_count += 1
                    dynamic_error = detach_state(dynamic_error)
                    previous_previous = previous
                    previous = detach_state(posterior)
    return total_loss / frame_count


def evaluate_variant(model, predictor, legacy, context, groups, mask, include_reference_paths=True):
    confusion = {
        name: torch.zeros((19, 19), dtype=torch.int64)
        for name in (("clean", "noisy", "prediction", "legacy", "context")
                    if include_reference_paths else ("legacy", "context"))
    }
    gain_change = {"z1": 0.0, "z4": 0.0}
    max_gain_difference = 0.0
    logit_residual = {"z1": 0.0, "z4": 0.0}
    legacy_mse = 0.0
    context_mse = 0.0
    frame_count = 0
    finite = True
    model.eval()
    predictor.eval()
    legacy.eval()
    context.eval()
    with torch.no_grad():
        for sequence_id, samples in groups.items():
            observation_previous_previous = None
            observation_previous = None
            legacy_previous_previous = None
            legacy_previous = None
            context_previous_previous = None
            context_previous = None
            legacy_error = None
            context_error = None
            for frame_index, sample in enumerate(samples):
                clean_image = load_image(sample)
                noisy_image = add_frame_noise(clean_image, SIGMA)
                clean_state, clean_features = (
                    encode_with_raw_features(model, clean_image)
                    if include_reference_paths
                    else (None, None)
                )
                observation, noisy_features = encode_with_raw_features(model, noisy_image)
                if observation_previous is None:
                    observation_previous = observation
                    legacy_previous = observation
                    context_previous = observation
                    continue
                if observation_previous_previous is None:
                    observation_previous_previous = observation_previous
                    observation_previous = observation
                    legacy_previous_previous = legacy_previous
                    legacy_previous = observation
                    context_previous_previous = context_previous
                    context_previous = observation
                    continue
                if include_reference_paths:
                    prediction_prior, _ = predict_current(
                        predictor,
                        observation_previous_previous,
                        observation_previous,
                        observation,
                    )
                legacy_prior, legacy_error_now = predict_current(
                    predictor,
                    legacy_previous_previous,
                    legacy_previous,
                    observation,
                )
                context_prior, context_error_now = predict_current(
                    predictor,
                    context_previous_previous,
                    context_previous,
                    observation,
                )
                legacy_error = update_dynamic_error(legacy_error_now, legacy_error)
                context_error = update_dynamic_error(context_error_now, context_error)
                legacy_posterior = posterior_state(
                    legacy_prior,
                    legacy_error_now,
                    legacy_error,
                    observation,
                    legacy,
                )
                gain = legacy_gains(context_error, legacy)
                context_posterior, new_gains, residuals = context_step(
                    context_prior,
                    observation,
                    context_error_now,
                    context_error,
                    gain,
                    context,
                    mask,
                )
                output_size = tuple(clean_image.shape[-2:])
                noisy_host = corrected_host_feature(
                    model, noisy_features, observation, observation, output_size
                )
                legacy_host = corrected_host_feature(
                    model, noisy_features, observation, legacy_posterior, output_size
                )
                context_host = corrected_host_feature(
                    model, noisy_features, observation, context_posterior, output_size
                )
                logits = {
                    "legacy": model.decode_from_host_feature(legacy_host),
                    "context": model.decode_from_host_feature(context_host),
                }
                if include_reference_paths:
                    clean_host = corrected_host_feature(
                        model, clean_features, clean_state, clean_state, output_size
                    )
                    prediction_host = corrected_host_feature(
                        model, noisy_features, observation, prediction_prior, output_size
                    )
                    logits.update(
                        {
                            "clean": model.decode_from_host_feature(clean_host),
                            "noisy": model.decode_from_host_feature(noisy_host),
                            "prediction": model.decode_from_host_feature(prediction_host),
                        }
                    )
                mask_tensor = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name, value in logits.items():
                    prediction = value.argmax(dim=1).squeeze(0).cpu().to(torch.int64)
                    update_confusion_matrix(confusion[name], prediction, mask_tensor)
                if include_reference_paths:
                    legacy_mse += posterior_mse(legacy_posterior, clean_state).item()
                    context_mse += posterior_mse(context_posterior, clean_state).item()
                for name, old_gain, updated_gain, residual in zip(
                    ("z1", "z4"), gain, new_gains, residuals
                ):
                    difference = (updated_gain - old_gain).abs()
                    gain_change[name] += difference.mean().item()
                    max_gain_difference = max(max_gain_difference, difference.max().item())
                    logit_residual[name] += residual.abs().mean().item()
                finite = finite and all(
                    torch.isfinite(value).all().item()
                    for value in (
                        context_error_now.z1,
                        context_error_now.z4,
                        context_error.z1,
                        context_error.z4,
                        context_posterior.z1,
                        context_posterior.z4,
                        new_gains[0],
                        new_gains[1],
                        residuals[0],
                        residuals[1],
                    )
                )
                frame_count += 1
                legacy_error = detach_state(legacy_error)
                context_error = detach_state(context_error)
                observation_previous_previous = observation_previous
                observation_previous = observation
                legacy_previous_previous = legacy_previous
                legacy_previous = detach_state(legacy_posterior)
                context_previous_previous = context_previous
                context_previous = detach_state(context_posterior)
    metrics = {
        name: float(torch.nanmean(compute_iou(value)).item())
        for name, value in confusion.items()
    }
    return {
        "metrics": metrics,
        "val_legacy_posterior_mse": legacy_mse / frame_count if include_reference_paths else None,
        "val_context_posterior_mse": context_mse / frame_count if include_reference_paths else None,
        "diagnostics": {
            "z1": {
                "mean_abs_gain_change": gain_change["z1"] / frame_count,
                "mean_abs_logit_residual": logit_residual["z1"] / frame_count,
            },
            "z4": {
                "mean_abs_gain_change": gain_change["z4"] / frame_count,
                "mean_abs_logit_residual": logit_residual["z4"] / frame_count,
            },
        },
        "max_abs_gain_difference": max_gain_difference,
        "finite": finite,
        "evaluated_frame_count": frame_count,
    }


def contribution_label(delta):
    if delta >= 0.003:
        return "MEANINGFUL CONTRIBUTION"
    if delta > 0.0:
        return "WEAK CONTRIBUTION"
    return "NO CONTRIBUTION"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Context residual ablation expects GPU 0.")
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_CONTEXT_ABLATION_OUTPUT_DIR",
            "/home/lin/predify/experiments/kitti_step_context_residual_ablation",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model, predictor, legacy, _, checkpoints = load_components(root)
    train_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_groups = sequence_groups(train_dataset)
    val_groups = sequence_groups(val_dataset)
    results = {}

    for variant_name, mask in VARIANTS:
        reset_rng()
        context = torch.nn.ModuleList(
            [ContextResidualCorrection(), ContextResidualCorrection()]
        ).cuda()
        context.requires_grad_(True)
        reset_rng()
        initial = evaluate_variant(
            model,
            predictor,
            legacy,
            context,
            val_groups,
            mask,
            include_reference_paths=False,
        )
        initial_gate = {
            "finite": initial["finite"],
            "mIoU_legacy": initial["metrics"]["legacy"],
            "mIoU_zero_residual": initial["metrics"]["context"],
            "mIoU_absolute_difference": abs(
                initial["metrics"]["context"] - initial["metrics"]["legacy"]
            ),
            "max_abs_gain_difference": initial["max_abs_gain_difference"],
            "pass": (
                initial["finite"]
                and abs(initial["metrics"]["context"] - initial["metrics"]["legacy"]) < 1e-4
                and initial["max_abs_gain_difference"] < 1e-5
            ),
        }
        print(json.dumps({variant_name: {"initial_legacy_equivalence": initial_gate}}, sort_keys=True), flush=True)
        if not initial_gate["pass"]:
            results[variant_name] = {"initial_legacy_equivalence": initial_gate}
            continue
        optimizer = torch.optim.AdamW(
            context.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        best_checkpoint = output_dir / f"best_{variant_name.lower()}.pt"
        best_val_loss = float("inf")
        best_epoch = None
        history = []
        for epoch in range(1, EPOCHS + 1):
            reset_rng()
            train_loss = run_epoch(
                model, predictor, legacy, context, train_groups, optimizer, True, mask
            )
            reset_rng()
            with torch.no_grad():
                val_loss = run_epoch(
                    model, predictor, legacy, context, val_groups, optimizer, False, mask
                )
            record = {
                "epoch": epoch,
                "train_posterior_mse": train_loss,
                "val_posterior_mse": val_loss,
            }
            history.append(record)
            print(json.dumps({variant_name: record}, sort_keys=True), flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(
                    {
                        "context_state_dict": context.state_dict(),
                        "variant": variant_name,
                        "epoch": epoch,
                        "val_posterior_mse": val_loss,
                    },
                    best_checkpoint,
                )
        payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
        context.load_state_dict(payload["context_state_dict"], strict=True)
        reset_rng()
        final = evaluate_variant(model, predictor, legacy, context, val_groups, mask)
        results[variant_name] = {
            "input_mask": {
                "prediction": mask[0],
                "observation": mask[1],
                "instant_error": mask[2],
                "dynamic_error": mask[3],
                "legacy_gain": mask[4],
            },
            "initial_legacy_equivalence": initial_gate,
            "history": history,
            "best_epoch": best_epoch,
            "best_val_posterior_mse": best_val_loss,
            "checkpoint": str(best_checkpoint),
            "metrics": final["metrics"],
            "val_legacy_posterior_mse": final["val_legacy_posterior_mse"],
            "val_context_posterior_mse": final["val_context_posterior_mse"],
            "diagnostics": final["diagnostics"],
            "finite": final["finite"],
            "evaluated_frame_count": final["evaluated_frame_count"],
        }

    legacy_miou = results["FULL_CONTEXT"]["metrics"]["legacy"]
    ordered = ("GAIN_ONLY", "GAIN_DYNAMIC", "GAIN_DYNAMIC_INSTANT", "FULL_CONTEXT")
    for previous, current in zip(ordered[:-1], ordered[1:]):
        results[current]["nested_delta_from_previous"] = (
            results[current]["metrics"]["context"] - results[previous]["metrics"]["context"]
        )
        results[current]["nested_contribution"] = contribution_label(
            results[current]["nested_delta_from_previous"]
        )
    results["GAIN_ONLY"]["delta_from_legacy"] = (
        results["GAIN_ONLY"]["metrics"]["context"] - legacy_miou
    )
    results["FULL_CONTEXT"]["full_context_reproduction"] = {
        "mIoU_at_least_0.3080": results["FULL_CONTEXT"]["metrics"]["context"] >= 0.3080,
        "relative_to_legacy_at_least_0.005": (
            results["FULL_CONTEXT"]["metrics"]["context"] - legacy_miou >= 0.005
        ),
        "pass": (
            results["FULL_CONTEXT"]["metrics"]["context"] >= 0.3080
            and results["FULL_CONTEXT"]["metrics"]["context"] - legacy_miou >= 0.005
        ),
    }
    summary = {
        "experiment": "kitti_step_context_residual_input_ablation",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "config": {
            "seed": SEED,
            "gaussian_noise_sigma": SIGMA,
            "epochs": EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "context_channels": [640, 128, 128, 128],
            "trainable_parameter_count_per_variant": sum(
                parameter.numel() for parameter in ContextResidualCorrection().parameters()
            ) * 2,
            "corrected_layers": [1, 4],
        },
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "dataset": {
            "train_sequence_count": len(train_groups),
            "val_sequence_count": len(val_groups),
            "val_frame_count": len(val_dataset.samples),
            "val_evaluated_frame_count": 2963,
        },
        "legacy_paired_mIoU": legacy_miou,
        "variants": results,
        "full_context_reproduction": results["FULL_CONTEXT"]["full_context_reproduction"],
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
