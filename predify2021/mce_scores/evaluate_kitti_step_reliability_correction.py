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
    load_image,
    encode_image,
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
from predify2021.model_factory.deeplabv3plus_resnet50.corrections import (
    ErrorGainCorrection,
)
from predify2021.model_factory.deeplabv3plus_resnet50.reliability_correction import (
    ReliabilityAwareCorrection,
)


LEGACY_CORRECTION_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_closed_loop_dynamic_correction_9820f4e/"
    "best_closed_loop_correction.pt"
)
EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
SIGMA = 0.10
GAMMA = 0.207
HISTORY_WEIGHT = 0.793


def reliability_outputs(predicted, observation, error, dynamic_error, corrections):
    values = []
    gains = []
    uncertainties = []
    for index, correction in zip((0, 3), corrections):
        prediction, observation_uncertainty, gain = correction(
            predicted.as_tuple()[index],
            observation.as_tuple()[index],
            error.as_tuple()[index],
            dynamic_error.as_tuple()[index],
        )
        values.append(gain)
        uncertainties.append((prediction, observation_uncertainty))
        gains.append(gain)
    posterior = UnifiedFeatures(
        predicted.z1 + gains[0] * error.z1,
        predicted.z2,
        predicted.z3,
        predicted.z4 + gains[1] * error.z4,
    )
    posterior = posterior_with_observation_passthrough(posterior, observation)
    return posterior, uncertainties, gains


def reliability_loss(predicted, observation, clean_state, uncertainties):
    total = 0.0
    for prediction, observation_uncertainty, prediction_state, observation_state, clean in zip(
        (uncertainties[0][0], uncertainties[1][0]),
        (uncertainties[0][1], uncertainties[1][1]),
        (predicted.z1, predicted.z4),
        (observation.z1, observation.z4),
        (clean_state.z1, clean_state.z4),
    ):
        prediction_residual = (prediction_state - clean).square()
        observation_residual = (observation_state - clean).square()
        total = total + 0.5 * (
            prediction_residual / prediction + prediction.log()
        ).mean()
        total = total + 0.5 * (
            observation_residual / observation_uncertainty
            + observation_uncertainty.log()
        ).mean()
    return total


def run_epoch(model, predictor, corrections, groups, optimizer, training):
    model.eval()
    predictor.eval()
    corrections.train(training)
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
                posterior, uncertainties, _ = reliability_outputs(
                    predicted, observation, error, dynamic_error, corrections
                )
                loss = reliability_loss(
                    predicted, observation, clean_state, uncertainties
                )
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


def evaluate_paths(model, predictor, legacy, reliability, groups):
    confusion = {
        name: torch.zeros((19, 19), dtype=torch.int64)
        for name in ("clean", "noisy", "prediction", "legacy", "reliability")
    }
    rows = []
    finite = True
    stats = {
        name: {
            "p": 0.0,
            "r": 0.0,
            "gain": 0.0,
            "obs_gain_sum": 0.0,
            "obs_gain_count": 0,
            "pred_gain_sum": 0.0,
            "pred_gain_count": 0,
        }
        for name in ("z1", "z4")
    }
    frame_count = 0
    model.eval()
    predictor.eval()
    legacy.eval()
    reliability.eval()
    with torch.no_grad():
        for sequence_id, samples in groups.items():
            observation_previous_previous = None
            observation_previous = None
            legacy_previous_previous = None
            legacy_previous = None
            reliability_previous_previous = None
            reliability_previous = None
            legacy_error = None
            reliability_error = None
            for frame_index, sample in enumerate(samples):
                clean_image = load_image(sample)
                noisy_image = add_frame_noise(clean_image, SIGMA)
                clean_state = encode_image(model, clean_image)
                observation = encode_image(model, noisy_image)
                if observation_previous is None:
                    observation_previous = observation
                    legacy_previous = observation
                    reliability_previous = observation
                    continue
                if observation_previous_previous is None:
                    observation_previous_previous = observation_previous
                    observation_previous = observation
                    legacy_previous_previous = legacy_previous
                    legacy_previous = observation
                    reliability_previous_previous = reliability_previous
                    reliability_previous = observation
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
                reliability_prior, reliability_instant = predict_current(
                    predictor,
                    reliability_previous_previous,
                    reliability_previous,
                    observation,
                )
                legacy_error = update_dynamic_error(legacy_instant, legacy_error)
                reliability_error = update_dynamic_error(
                    reliability_instant, reliability_error
                )
                legacy_posterior = posterior_state(
                    legacy_prior,
                    legacy_instant,
                    legacy_error,
                    observation,
                    legacy,
                )
                reliability_posterior, uncertainties, gains = reliability_outputs(
                    reliability_prior,
                    observation,
                    reliability_instant,
                    reliability_error,
                    reliability,
                )
                noisy_features = model.extract_backbone_features(noisy_image)
                output_size = tuple(clean_image.shape[-2:])
                prediction_host = corrected_host_feature(
                    model, noisy_features, observation, prediction_prior, output_size
                )
                legacy_host = corrected_host_feature(
                    model, noisy_features, observation, legacy_posterior, output_size
                )
                reliability_host = corrected_host_feature(
                    model,
                    noisy_features,
                    observation,
                    reliability_posterior,
                    output_size,
                )
                logits = {
                    "clean": model(clean_image),
                    "noisy": model(noisy_image),
                    "prediction": model.decode_from_host_feature(prediction_host),
                    "legacy": model.decode_from_host_feature(legacy_host),
                    "reliability": model.decode_from_host_feature(reliability_host),
                }
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name, value in logits.items():
                    prediction = value.argmax(dim=1).squeeze(0).cpu().to(torch.int64)
                    update_confusion_matrix(confusion[name], prediction, mask)
                for name, index in (("z1", 0), ("z4", 3)):
                    prediction_uncertainty, observation_uncertainty = uncertainties[index // 3]
                    gain = gains[index // 3]
                    prediction_state = reliability_prior.as_tuple()[index]
                    observation_state = observation.as_tuple()[index]
                    clean = clean_state.as_tuple()[index]
                    prediction_residual = (prediction_state - clean).square()
                    observation_residual = (observation_state - clean).square()
                    observation_better = observation_residual < prediction_residual
                    prediction_better = prediction_residual < observation_residual
                    stats[name]["p"] += prediction_uncertainty.mean().item()
                    stats[name]["r"] += observation_uncertainty.mean().item()
                    stats[name]["gain"] += gain.mean().item()
                    stats[name]["obs_gain_sum"] += gain[observation_better].sum().item()
                    stats[name]["obs_gain_count"] += observation_better.sum().item()
                    stats[name]["pred_gain_sum"] += gain[prediction_better].sum().item()
                    stats[name]["pred_gain_count"] += prediction_better.sum().item()
                finite = finite and all(
                    torch.isfinite(value).all().item()
                    for value in (
                        reliability_instant.z1,
                        reliability_instant.z4,
                        reliability_error.z1,
                        reliability_error.z4,
                        reliability_posterior.z1,
                        reliability_posterior.z4,
                        uncertainties[0][0],
                        uncertainties[0][1],
                        gains[0],
                        uncertainties[1][0],
                        uncertainties[1][1],
                        gains[1],
                    )
                )
                row = {
                    "sequence_id": sequence_id,
                    "frame_id": sample["frame_id"],
                    "frame_index": frame_index,
                    "p_z1": uncertainties[0][0].mean().item(),
                    "r_z1": uncertainties[0][1].mean().item(),
                    "gain_z1": gains[0].mean().item(),
                    "p_z4": uncertainties[1][0].mean().item(),
                    "r_z4": uncertainties[1][1].mean().item(),
                    "gain_z4": gains[1].mean().item(),
                }
                row.update(
                    state_metrics(
                        reliability_instant,
                        reliability_error,
                        reliability_posterior,
                        observation,
                    )
                )
                rows.append(row)
                frame_count += 1
                legacy_error = detach_state(legacy_error)
                reliability_error = detach_state(reliability_error)
                observation_previous_previous = observation_previous
                observation_previous = observation
                legacy_previous_previous = legacy_previous
                legacy_previous = detach_state(legacy_posterior)
                reliability_previous_previous = reliability_previous
                reliability_previous = detach_state(reliability_posterior)
    metrics = {
        name: float(torch.nanmean(compute_iou(value)).item())
        for name, value in confusion.items()
    }
    diagnostics = {}
    for name, value in stats.items():
        diagnostics[name] = {
            "mean_P": value["p"] / frame_count,
            "mean_R": value["r"] / frame_count,
            "mean_gain": value["gain"] / frame_count,
            "mean_gain_when_observation_better": value["obs_gain_sum"]
            / value["obs_gain_count"],
            "mean_gain_when_prediction_better": value["pred_gain_sum"]
            / value["pred_gain_count"],
        }
    return metrics, diagnostics, rows, finite


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def decision_label(delta, direction):
    if delta >= 0.010 and direction:
        return "RELIABILITY CORRECTION STRONG GO"
    if delta >= 0.005 and direction:
        return "RELIABILITY CORRECTION GO"
    if delta >= 0.005 and not direction:
        return "RELIABILITY MECHANISM INCONCLUSIVE"
    if delta > 0.0:
        return "RELIABILITY CORRECTION INCONCLUSIVE"
    return "RELIABILITY CORRECTION NO-GO"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Reliability correction expects GPU 0.")
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
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
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_RELIABILITY_OUTPUT_DIR",
            "/home/lin/predify/experiments/kitti_step_reliability_correction",
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
    reliability = torch.nn.ModuleList(
        [ReliabilityAwareCorrection(), ReliabilityAwareCorrection()]
    ).cuda()
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    legacy.requires_grad_(False)
    train_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_groups = sequence_groups(train_dataset)
    val_groups = sequence_groups(val_dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        reliability.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_checkpoint = output_dir / "best_reliability_correction.pt"
    best_val_loss = float("inf")
    best_epoch = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model, predictor, reliability, train_groups, optimizer, True
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model, predictor, reliability, val_groups, optimizer, False
            )
        record = {
            "epoch": epoch,
            "train_reliability_loss": train_loss,
            "val_reliability_loss": val_loss,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "reliability_state_dict": reliability.state_dict(),
                    "epoch": epoch,
                    "val_reliability_loss": val_loss,
                },
                best_checkpoint,
            )
    best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    reliability.load_state_dict(best_payload["reliability_state_dict"], strict=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    metrics, diagnostics, rows, finite = evaluate_paths(
        model, predictor, legacy, reliability, val_groups
    )
    write_rows(output_dir / "per_frame.csv", rows)
    phases, phase_stable = phase_statistics(rows)
    delta = metrics["reliability"] - metrics["legacy"]
    direction = all(
        diagnostics[name]["mean_gain_when_observation_better"]
        > diagnostics[name]["mean_gain_when_prediction_better"]
        for name in ("z1", "z4")
    )
    summary = {
        "experiment": "kitti_step_reliability_correction",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoints": {
            "static_host": str(static_checkpoint),
            "fixed_adapter": str(adapter_checkpoint),
            "fixed_predictor": str(predictor_checkpoint),
            "legacy_correction": str(legacy_checkpoint),
            "best_reliability_correction": str(best_checkpoint),
        },
        "config": {
            "seed": seed,
            "gaussian_noise_sigma": SIGMA,
            "epochs": EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gamma": GAMMA,
            "history_weight": HISTORY_WEIGHT,
            "corrected_layers": [1, 4],
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in reliability.parameters()
            ),
        },
        "dataset": {
            "split_train": "train",
            "split_val": "val",
            "train_sequence_count": len(train_groups),
            "val_sequence_count": len(val_groups),
            "val_frame_count": len(val_dataset.samples),
            "val_evaluated_frame_count": len(rows),
        },
        "history": history,
        "best": {"epoch": best_epoch, "val_reliability_loss": best_val_loss},
        "metrics": {
            "mIoU_clean_static": metrics["clean"],
            "mIoU_noisy_static": metrics["noisy"],
            "mIoU_continuous_prediction_only": metrics["prediction"],
            "mIoU_legacy_closed_loop": metrics["legacy"],
            "mIoU_reliability_closed_loop": metrics["reliability"],
            "reliability_minus_legacy": delta,
        },
        "reliability_diagnostics": diagnostics,
        "stability": {
            "dynamic_error_finite": finite,
            "posterior_finite": finite,
            "reliability_outputs_finite": finite,
            "phase_stable": phase_stable,
            "phase_statistics": phases,
        },
        "reliability_direction_correct": direction,
        "decision": decision_label(delta, direction),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
