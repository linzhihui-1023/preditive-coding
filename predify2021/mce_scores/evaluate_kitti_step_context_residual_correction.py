import csv
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
    ADAPTER_CHECKPOINT_DEFAULT,
    PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    add_frame_noise,
    phase_statistics,
    posterior_state,
    posterior_with_observation_passthrough,
    state_metrics,
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
from predify2021.mce_scores.train_kitti_step_state_predictor import (
    load_static_kitti_checkpoint,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    MultiLayerPredictor,
    UnifiedFeatures,
    build_deeplabv3plus_resnet50_host,
)
from predify2021.model_factory.deeplabv3plus_resnet50.context_residual_correction import (
    ContextResidualCorrection,
)
from predify2021.model_factory.deeplabv3plus_resnet50.corrections import (
    ErrorGainCorrection,
)


LEGACY_CORRECTION_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_closed_loop_dynamic_correction_9820f4e/"
    "best_closed_loop_correction.pt"
)
EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
SIGMA = 0.10


def legacy_gains(dynamic_error, legacy):
    return legacy[0](dynamic_error.z1), legacy[1](dynamic_error.z4)


def context_outputs(
    predicted,
    observation,
    error,
    dynamic_error,
    legacy_gain,
    context,
):
    residuals = (
        context[0](
            predicted.z1,
            observation.z1,
            error.z1,
            dynamic_error.z1,
            legacy_gain[0],
        ),
        context[1](
            predicted.z4,
            observation.z4,
            error.z4,
            dynamic_error.z4,
            legacy_gain[1],
        ),
    )
    new_gains = tuple(
        torch.sigmoid(
            torch.logit(gain.clamp(1e-6, 1.0 - 1e-6)) + residual
        )
        for gain, residual in zip(legacy_gain, residuals)
    )
    posterior = UnifiedFeatures(
        predicted.z1 + new_gains[0] * error.z1,
        predicted.z2,
        predicted.z3,
        predicted.z4 + new_gains[1] * error.z4,
    )
    posterior = posterior_with_observation_passthrough(posterior, observation)
    return posterior, new_gains, residuals


def posterior_mse(posterior, clean_state):
    return F.mse_loss(posterior.z1, clean_state.z1) + F.mse_loss(
        posterior.z4, clean_state.z4
    )


def run_epoch(model, predictor, legacy, context, groups, optimizer, training):
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
            for sample in samples:
                clean_image = load_image(sample)
                clean_state = encode_image(model, clean_image)
                noisy_image = add_frame_noise(clean_image, SIGMA)
                observation = encode_image(model, noisy_image)
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
                posterior, _, _ = context_outputs(
                    predicted,
                    observation,
                    error,
                    dynamic_error,
                    gain,
                    context,
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


def evaluate_paths(model, predictor, legacy, context, groups):
    confusion = {
        name: torch.zeros((19, 19), dtype=torch.int64)
        for name in ("clean", "noisy", "prediction", "legacy", "context")
    }
    rows = []
    finite = True
    frame_count = 0
    stats = {
        name: {"gain_change": 0.0, "logit_residual": 0.0, "max_gain_change": 0.0}
        for name in ("z1", "z4")
    }
    legacy_mse = 0.0
    context_mse = 0.0
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
                clean_state = encode_image(model, clean_image)
                observation = encode_image(model, noisy_image)
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
                prediction_prior, _ = predict_current(
                    predictor,
                    observation_previous_previous,
                    observation_previous,
                    observation,
                )
                legacy_prior, legacy_instant = predict_current(
                    predictor,
                    legacy_previous_previous,
                    legacy_previous,
                    observation,
                )
                context_prior, context_instant = predict_current(
                    predictor,
                    context_previous_previous,
                    context_previous,
                    observation,
                )
                legacy_error = update_dynamic_error(legacy_instant, legacy_error)
                context_error = update_dynamic_error(context_instant, context_error)
                legacy_posterior = posterior_state(
                    legacy_prior,
                    legacy_instant,
                    legacy_error,
                    observation,
                    legacy,
                )
                gain = legacy_gains(context_error, legacy)
                context_posterior, new_gain, residuals = context_outputs(
                    context_prior,
                    observation,
                    context_instant,
                    context_error,
                    gain,
                    context,
                )
                noisy_features = model.extract_backbone_features(noisy_image)
                output_size = tuple(clean_image.shape[-2:])
                prediction_host = corrected_host_feature(
                    model, noisy_features, observation, prediction_prior, output_size
                )
                legacy_host = corrected_host_feature(
                    model, noisy_features, observation, legacy_posterior, output_size
                )
                context_host = corrected_host_feature(
                    model, noisy_features, observation, context_posterior, output_size
                )
                logits = {
                    "clean": model(clean_image),
                    "noisy": model(noisy_image),
                    "prediction": model.decode_from_host_feature(prediction_host),
                    "legacy": model.decode_from_host_feature(legacy_host),
                    "context": model.decode_from_host_feature(context_host),
                }
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name, value in logits.items():
                    prediction = value.argmax(dim=1).squeeze(0).cpu().to(torch.int64)
                    update_confusion_matrix(confusion[name], prediction, mask)
                legacy_mse += posterior_mse(legacy_posterior, clean_state).item()
                context_mse += posterior_mse(context_posterior, clean_state).item()
                for name, old_gain, new_gain, residual in zip(
                    ("z1", "z4"), gain, new_gain, residuals
                ):
                    gain_change = (new_gain - old_gain).abs()
                    stats[name]["gain_change"] += gain_change.mean().item()
                    stats[name]["max_gain_change"] = max(
                        stats[name]["max_gain_change"], gain_change.max().item()
                    )
                    stats[name]["logit_residual"] += residual.abs().mean().item()
                finite = finite and all(
                    torch.isfinite(value).all().item()
                    for value in (
                        context_instant.z1,
                        context_instant.z4,
                        context_error.z1,
                        context_error.z4,
                        context_posterior.z1,
                        context_posterior.z4,
                        gain[0],
                        gain[1],
                        new_gain[0],
                        new_gain[1],
                        residuals[0],
                        residuals[1],
                    )
                )
                row = {
                    "sequence_id": sequence_id,
                    "frame_id": sample["frame_id"],
                    "frame_index": frame_index,
                    "gain_change_z1": (new_gain[0] - gain[0]).abs().mean().item(),
                    "logit_residual_z1": residuals[0].abs().mean().item(),
                    "gain_change_z4": (new_gain[1] - gain[1]).abs().mean().item(),
                    "logit_residual_z4": residuals[1].abs().mean().item(),
                }
                row.update(
                    state_metrics(
                        context_instant,
                        context_error,
                        context_posterior,
                        observation,
                    )
                )
                rows.append(row)
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
    diagnostics = {
        name: {
            "mean_abs_gain_change": value["gain_change"] / frame_count,
            "max_abs_gain_difference": value["max_gain_change"],
            "mean_abs_logit_residual": value["logit_residual"] / frame_count,
        }
        for name, value in stats.items()
    }
    return (
        metrics,
        diagnostics,
        rows,
        finite,
        legacy_mse / frame_count,
        context_mse / frame_count,
    )


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def decision_label(delta, posterior_improved):
    if delta >= 0.010 and posterior_improved:
        return "CONTEXT RESIDUAL CORRECTION STRONG GO"
    if delta >= 0.005 and posterior_improved:
        return "CONTEXT RESIDUAL CORRECTION GO"
    if delta >= 0.005 and not posterior_improved:
        return "CONTEXT RESIDUAL MECHANISM INCONCLUSIVE"
    if delta > 0.0:
        return "CONTEXT RESIDUAL CORRECTION INCONCLUSIVE"
    return "CONTEXT RESIDUAL CORRECTION NO-GO"


def load_components(root):
    static_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_STATIC_CHECKPOINT", STATIC_CHECKPOINT_DEFAULT)
    )
    adapter_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_ADAPTER_CHECKPOINT", ADAPTER_CHECKPOINT_DEFAULT)
    )
    predictor_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_PREDICTOR_CHECKPOINT", PREDICTOR_CHECKPOINT_DEFAULT)
    )
    legacy_checkpoint = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_LEGACY_CORRECTION_CHECKPOINT",
            LEGACY_CORRECTION_CHECKPOINT_DEFAULT,
        )
    )
    model = build_deeplabv3plus_resnet50_host(
        checkpoint_path=os.environ.get("PREDIFY_DEEPLABV3PLUS_CITYSCAPES_CHECKPOINT"),
    ).cuda()
    load_static_kitti_checkpoint(model, static_checkpoint)
    adapter_payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.load_state_dict(adapter_payload["adapter_state_dict"], strict=True)
    predictor = MultiLayerPredictor().cuda()
    predictor_payload = torch.load(predictor_checkpoint, map_location="cpu", weights_only=False)
    predictor.load_state_dict(predictor_payload["predictor_state_dict"], strict=True)
    legacy = torch.nn.ModuleList(
        [ErrorGainCorrection(), ErrorGainCorrection()]
    ).cuda()
    legacy_payload = torch.load(legacy_checkpoint, map_location="cpu", weights_only=False)
    legacy.load_state_dict(legacy_payload["correction_state_dict"], strict=True)
    context = torch.nn.ModuleList(
        [ContextResidualCorrection(), ContextResidualCorrection()]
    ).cuda()
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    legacy.requires_grad_(False)
    return model, predictor, legacy, context, {
        "static_host": static_checkpoint,
        "fixed_adapter": adapter_checkpoint,
        "fixed_predictor": predictor_checkpoint,
        "legacy_correction": legacy_checkpoint,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Context residual correction expects GPU 0.")
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_CONTEXT_RESIDUAL_OUTPUT_DIR",
            "/home/lin/predify/experiments/kitti_step_context_residual_correction",
        )
    )
    model, predictor, legacy, context, checkpoints = load_components(root)
    train_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_groups = sequence_groups(train_dataset)
    val_groups = sequence_groups(val_dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initial_metrics, initial_diagnostics, _, initial_finite, _, _ = evaluate_paths(
        model, predictor, legacy, context, val_groups
    )
    initial_gate = {
        "finite": initial_finite,
        "mIoU_legacy": initial_metrics["legacy"],
        "mIoU_zero_residual": initial_metrics["context"],
        "mIoU_absolute_difference": abs(
            initial_metrics["context"] - initial_metrics["legacy"]
        ),
        "max_abs_gain_difference": max(
            value["max_abs_gain_difference"] for value in initial_diagnostics.values()
        ),
        "pass": (
            initial_finite
            and abs(initial_metrics["context"] - initial_metrics["legacy"]) < 1e-4
            and max(
                value["max_abs_gain_difference"]
                for value in initial_diagnostics.values()
            )
            < 1e-5
        ),
    }
    print(json.dumps({"initial_legacy_equivalence": initial_gate}, sort_keys=True), flush=True)
    if not initial_gate["pass"]:
        summary = {
            "experiment": "kitti_step_context_residual_correction",
            "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
            "checkpoints": {name: str(path) for name, path in checkpoints.items()},
            "config": {"seed": seed, "gaussian_noise_sigma": SIGMA},
            "initial_legacy_equivalence": initial_gate,
            "decision": "INITIAL LEGACY EQUIVALENCE FAIL",
        }
        with (output_dir / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return

    optimizer = torch.optim.AdamW(
        context.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_checkpoint = output_dir / "best_context_residual_correction.pt"
    best_val_loss = float("inf")
    best_epoch = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model, predictor, legacy, context, train_groups, optimizer, True
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model, predictor, legacy, context, val_groups, optimizer, False
            )
        record = {
            "epoch": epoch,
            "train_posterior_mse": train_loss,
            "val_posterior_mse": val_loss,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "context_state_dict": context.state_dict(),
                    "epoch": epoch,
                    "val_posterior_mse": val_loss,
                },
                best_checkpoint,
            )
    best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    context.load_state_dict(best_payload["context_state_dict"], strict=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    metrics, diagnostics, rows, finite, legacy_mse, context_mse = evaluate_paths(
        model, predictor, legacy, context, val_groups
    )
    write_rows(output_dir / "per_frame.csv", rows)
    phases, phase_stable = phase_statistics(rows)
    delta = metrics["context"] - metrics["legacy"]
    posterior_improved = context_mse < legacy_mse
    summary = {
        "experiment": "kitti_step_context_residual_correction",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoints": {
            **{name: str(path) for name, path in checkpoints.items()},
            "best_context_residual": str(best_checkpoint),
        },
        "config": {
            "seed": seed,
            "gaussian_noise_sigma": SIGMA,
            "epochs": EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "corrected_layers": [1, 4],
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in context.parameters()
            ),
        },
        "dataset": {
            "train_sequence_count": len(train_groups),
            "val_sequence_count": len(val_groups),
            "val_frame_count": len(val_dataset.samples),
            "val_evaluated_frame_count": len(rows),
        },
        "initial_legacy_equivalence": initial_gate,
        "history": history,
        "best": {"epoch": best_epoch, "val_posterior_mse": best_val_loss},
        "metrics": {
            "mIoU_clean_static": metrics["clean"],
            "mIoU_noisy_static": metrics["noisy"],
            "mIoU_continuous_prediction_only": metrics["prediction"],
            "mIoU_legacy_closed_loop": metrics["legacy"],
            "mIoU_context_residual_closed_loop": metrics["context"],
            "context_residual_minus_legacy": delta,
            "val_legacy_posterior_mse": legacy_mse,
            "val_context_posterior_mse": context_mse,
        },
        "context_diagnostics": diagnostics,
        "stability": {
            "dynamic_error_finite": finite,
            "posterior_finite": finite,
            "gain_finite": finite,
            "logit_residual_finite": finite,
            "phase_stable": phase_stable,
            "phase_statistics": phases,
        },
        "posterior_mse_improved": posterior_improved,
        "decision": decision_label(delta, posterior_improved),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
