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
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    DYNAMIC_ERROR_GAIN,
    DYNAMIC_ERROR_SAMPLE_TIME,
    DYNAMIC_ERROR_TIME_CONSTANT,
    PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    encode_image,
    load_image,
    predict_current,
    sequence_groups,
    update_dynamic_error,
)
from predify2021.mce_scores.evaluate_kitti_step_error_correction import (
    corrected_host_feature,
    correction_states,
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


FIXED_DYNAMIC_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_dynamic_error_correction_9fc8d81/"
    "best_dynamic_correction.pt"
)


def add_frame_noise(clean_image, sigma):
    return torch.clamp(
        clean_image + sigma * torch.randn_like(clean_image), 0.0, 1.0
    )


def posterior_with_observation_passthrough(corrected, observation):
    return UnifiedFeatures(
        corrected.z1,
        observation.z2,
        observation.z3,
        corrected.z4,
    )


def posterior_state(predicted, error, dynamic_error, observation, corrections):
    corrected = correction_states(
        predicted, dynamic_error, error, corrections
    )
    return posterior_with_observation_passthrough(corrected, observation)


def state_metrics(error, dynamic_error, posterior, observation):
    return {
        "error_z1": error.z1.abs().mean().item(),
        "dynamic_error_z1": dynamic_error.z1.abs().mean().item(),
        "correction_z1": (posterior.z1 - observation.z1).abs().mean().item(),
        "state_z1": posterior.z1.abs().mean().item(),
        "error_z4": error.z4.abs().mean().item(),
        "dynamic_error_z4": dynamic_error.z4.abs().mean().item(),
        "correction_z4": (posterior.z4 - observation.z4).abs().mean().item(),
        "state_z4": posterior.z4.abs().mean().item(),
    }


def intervention_test(model, predictor, corrections, samples, sigma):
    observations = []
    for sample in samples[:4]:
        noisy_image = add_frame_noise(load_image(sample), sigma)
        observations.append(encode_image(model, noisy_image))
    predicted2, error2 = predict_current(
        predictor, observations[0], observations[1], observations[2]
    )
    dynamic_error2 = update_dynamic_error(error2, None)
    posterior2 = posterior_state(
        predicted2, error2, dynamic_error2, observations[2], corrections
    )
    intervened2 = UnifiedFeatures(
        posterior2.z1 + 0.1,
        posterior2.z2,
        posterior2.z3,
        posterior2.z4 + 0.1,
    )
    closed_prior, _ = predict_current(
        predictor, observations[1], posterior2, observations[3]
    )
    intervened_prior, _ = predict_current(
        predictor, observations[1], intervened2, observations[3]
    )
    open_prior, _ = predict_current(
        predictor, observations[1], observations[2], observations[3]
    )
    open_prior_after_intervention, _ = predict_current(
        predictor, observations[1], observations[2], observations[3]
    )
    closed_difference = max(
        (changed - original).abs().max().item()
        for changed, original in zip(
            intervened_prior.as_tuple(), closed_prior.as_tuple()
        )
    )
    open_difference = max(
        (changed - original).abs().max().item()
        for changed, original in zip(
            open_prior_after_intervention.as_tuple(), open_prior.as_tuple()
        )
    )
    return {
        "closed_loop_prior_max_abs_change": closed_difference,
        "open_loop_prior_max_abs_change": open_difference,
        "closed_loop_routing_confirmed": closed_difference > 0.0,
        "open_loop_isolation_confirmed": open_difference == 0.0,
    }


def collect_closed_loop_rows(model, predictor, corrections, groups, sigma):
    rows = []
    finite = True
    model.eval()
    predictor.eval()
    corrections.eval()
    with torch.no_grad():
        for sequence_id, samples in groups.items():
            previous_previous = None
            previous = None
            dynamic_error = None
            for frame_index, sample in enumerate(samples):
                noisy_image = add_frame_noise(load_image(sample), sigma)
                observation = encode_image(model, noisy_image)
                if previous is None:
                    previous = observation
                    continue
                if previous_previous is None:
                    previous_previous = previous
                    previous = observation
                    continue
                predicted, error = predict_current(
                    predictor, previous_previous, previous, observation
                )
                dynamic_error = update_dynamic_error(error, dynamic_error)
                posterior = posterior_state(
                    predicted, error, dynamic_error, observation, corrections
                )
                tensors = (
                    error.z1,
                    error.z4,
                    dynamic_error.z1,
                    dynamic_error.z4,
                    posterior.z1,
                    posterior.z4,
                )
                finite = finite and all(torch.isfinite(value).all().item() for value in tensors)
                row = {
                    "sequence_id": sequence_id,
                    "frame_id": sample["frame_id"],
                    "frame_index": frame_index,
                }
                row.update(state_metrics(error, dynamic_error, posterior, observation))
                rows.append(row)
                dynamic_error = detach_state(dynamic_error)
                previous_previous = previous
                previous = detach_state(posterior)
    return rows, finite


def phase_statistics(rows):
    by_sequence = {}
    for row in rows:
        by_sequence.setdefault(row["sequence_id"], []).append(row)
    metrics = (
        "error_z1",
        "dynamic_error_z1",
        "correction_z1",
        "state_z1",
        "error_z4",
        "dynamic_error_z4",
        "correction_z4",
        "state_z4",
    )
    results = {}
    stable = True
    for sequence_id, sequence_rows in by_sequence.items():
        count = max(1, int(len(sequence_rows) * 0.2))
        first = sequence_rows[:count]
        last = sequence_rows[-count:]
        sequence_result = {}
        for metric in metrics:
            first_mean = sum(row[metric] for row in first) / len(first)
            last_mean = sum(row[metric] for row in last) / len(last)
            ratio = last_mean / first_mean
            sequence_result[metric] = {
                "first_20_percent_mean": first_mean,
                "last_20_percent_mean": last_mean,
                "last_over_first": ratio,
            }
            if metric.startswith("error_") or metric.startswith("correction_"):
                stable = stable and ratio <= 2.0
            stable = stable and ratio <= 10.0
        for previous_row, current_row in zip(sequence_rows, sequence_rows[1:]):
            for metric in metrics:
                stable = stable and current_row[metric] <= 10.0 * previous_row[metric]
        results[sequence_id] = sequence_result
    return results, stable


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_training_epoch(
    model,
    predictor,
    corrections,
    groups,
    optimizer,
    training,
    sigma,
    gradient_report=None,
):
    model.eval()
    predictor.eval()
    corrections.train(training)
    total_loss = 0.0
    frame_count = 0
    for samples in groups.values():
        previous_previous = None
        previous = None
        dynamic_error = None
        for sample in samples:
            clean_image = load_image(sample)
            clean_state = encode_image(model, clean_image)
            noisy_image = add_frame_noise(clean_image, sigma)
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
            posterior = posterior_state(
                predicted, error, dynamic_error, observation, corrections
            )
            loss = F.mse_loss(posterior.z1, clean_state.z1) + F.mse_loss(
                posterior.z4, clean_state.z4
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_report is not None and not gradient_report:
                    gradient_report.update(
                        {
                            "host_has_gradient": any(
                                parameter.grad is not None for parameter in model.parameters()
                            ),
                            "predictor_has_gradient": any(
                                parameter.grad is not None
                                for parameter in predictor.parameters()
                            ),
                            "correction_gradient_tensor_count": sum(
                                parameter.grad is not None
                                for parameter in corrections.parameters()
                            ),
                        }
                    )
                optimizer.step()
            total_loss += loss.detach().item()
            frame_count += 1
            dynamic_error = detach_state(dynamic_error)
            previous_previous = previous
            previous = detach_state(posterior)
    return total_loss / frame_count


def evaluate_paths(model, predictor, corrections, groups, sigma):
    confusion = {
        name: torch.zeros((19, 19), dtype=torch.int64)
        for name in ("clean", "noisy", "prediction", "open", "closed")
    }
    closed_rows = []
    model.eval()
    predictor.eval()
    corrections.eval()
    with torch.no_grad():
        for sequence_id, samples in groups.items():
            observation_previous_previous = None
            observation_previous = None
            closed_previous_previous = None
            closed_previous = None
            open_dynamic_error = None
            closed_dynamic_error = None
            for frame_index, sample in enumerate(samples):
                clean_image = load_image(sample)
                noisy_image = add_frame_noise(clean_image, sigma)
                observation = encode_image(model, noisy_image)
                if observation_previous is None:
                    observation_previous = observation
                    closed_previous = observation
                    continue
                if observation_previous_previous is None:
                    observation_previous_previous = observation_previous
                    observation_previous = observation
                    closed_previous_previous = closed_previous
                    closed_previous = observation
                    continue
                observation_prior, observation_error = predict_current(
                    predictor,
                    observation_previous_previous,
                    observation_previous,
                    observation,
                )
                closed_prior, closed_error = predict_current(
                    predictor,
                    closed_previous_previous,
                    closed_previous,
                    observation,
                )
                open_dynamic_error = update_dynamic_error(
                    observation_error, open_dynamic_error
                )
                closed_dynamic_error = update_dynamic_error(
                    closed_error, closed_dynamic_error
                )
                open_posterior = posterior_state(
                    observation_prior,
                    observation_error,
                    open_dynamic_error,
                    observation,
                    corrections,
                )
                closed_posterior = posterior_state(
                    closed_prior,
                    closed_error,
                    closed_dynamic_error,
                    observation,
                    corrections,
                )
                noisy_features = model.extract_backbone_features(noisy_image)
                output_size = tuple(clean_image.shape[-2:])
                prediction_host = corrected_host_feature(
                    model,
                    noisy_features,
                    observation,
                    observation_prior,
                    output_size,
                )
                open_host = corrected_host_feature(
                    model,
                    noisy_features,
                    observation,
                    open_posterior,
                    output_size,
                )
                closed_host = corrected_host_feature(
                    model,
                    noisy_features,
                    observation,
                    closed_posterior,
                    output_size,
                )
                logits = {
                    "clean": model(clean_image),
                    "noisy": model(noisy_image),
                    "prediction": model.decode_from_host_feature(prediction_host),
                    "open": model.decode_from_host_feature(open_host),
                    "closed": model.decode_from_host_feature(closed_host),
                }
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name, value in logits.items():
                    prediction = value.argmax(dim=1).squeeze(0).cpu().to(torch.int64)
                    update_confusion_matrix(confusion[name], prediction, mask)
                row = {
                    "sequence_id": sequence_id,
                    "frame_id": sample["frame_id"],
                    "frame_index": frame_index,
                }
                row.update(
                    state_metrics(
                        closed_error,
                        closed_dynamic_error,
                        closed_posterior,
                        observation,
                    )
                )
                closed_rows.append(row)
                open_dynamic_error = detach_state(open_dynamic_error)
                closed_dynamic_error = detach_state(closed_dynamic_error)
                observation_previous_previous = observation_previous
                observation_previous = observation
                closed_previous_previous = closed_previous
                closed_previous = detach_state(closed_posterior)
    metrics = {
        name: float(torch.nanmean(compute_iou(value)).item())
        for name, value in confusion.items()
    }
    return metrics, closed_rows


def decision_label(closed_miou, open_miou):
    improvement = closed_miou - open_miou
    if improvement <= 0.0:
        return "NO-GO"
    if improvement < 0.001759:
        return "WEAK GO"
    return "GO"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Closed-loop dynamic correction expects GPU 0.")
    seed = int(os.environ.get("PREDIFY_SEED", "0"))
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
    correction_checkpoint = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_FIXED_DYNAMIC_CHECKPOINT",
            FIXED_DYNAMIC_CHECKPOINT_DEFAULT,
        )
    )
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_CLOSED_LOOP_OUTPUT_DIR",
            "/home/lin/predify/experiments/kitti_step_closed_loop_dynamic_correction",
        )
    )
    epochs = int(os.environ.get("PREDIFY_KITTI_STEP_CLOSED_LOOP_EPOCHS", "3"))
    learning_rate = float(
        os.environ.get("PREDIFY_KITTI_STEP_CLOSED_LOOP_LR", "0.0001")
    )
    weight_decay = float(
        os.environ.get("PREDIFY_KITTI_STEP_CLOSED_LOOP_WEIGHT_DECAY", "0.01")
    )
    sigma = 0.10
    model = build_deeplabv3plus_resnet50_host(
        checkpoint_path=os.environ.get("PREDIFY_DEEPLABV3PLUS_CITYSCAPES_CHECKPOINT"),
    ).cuda()
    load_static_kitti_checkpoint(model, static_checkpoint)
    adapter_payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.load_state_dict(adapter_payload["adapter_state_dict"], strict=True)
    predictor = MultiLayerPredictor().cuda()
    predictor_payload = torch.load(predictor_checkpoint, map_location="cpu", weights_only=False)
    predictor.load_state_dict(predictor_payload["predictor_state_dict"], strict=True)
    corrections = torch.nn.ModuleList(
        [ErrorGainCorrection(), ErrorGainCorrection()]
    ).cuda()
    correction_payload = torch.load(
        correction_checkpoint, map_location="cpu", weights_only=False
    )
    corrections.load_state_dict(
        correction_payload["correction_state_dict"], strict=True
    )
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    corrections.requires_grad_(False)
    train_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_groups = sequence_groups(train_dataset)
    val_groups = sequence_groups(val_dataset)
    selected_ids = sorted(val_groups)[:2]
    stability_groups = {sequence_id: val_groups[sequence_id] for sequence_id in selected_ids}
    output_dir.mkdir(parents=True, exist_ok=True)
    intervention = intervention_test(
        model, predictor, corrections, stability_groups[selected_ids[0]], sigma
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    stability_rows, finite = collect_closed_loop_rows(
        model, predictor, corrections, stability_groups, sigma
    )
    stability_phases, ratio_stable = phase_statistics(stability_rows)
    stable = finite and ratio_stable
    write_rows(output_dir / "stability_per_frame.csv", stability_rows)
    phase_one = {
        "selected_sequences": selected_ids,
        "finite": finite,
        "stable": stable,
        "intervention": intervention,
        "phase_statistics": stability_phases,
    }
    print(json.dumps({"phase_one_stable": stable}, sort_keys=True), flush=True)
    if not stable:
        summary = {
            "experiment": "kitti_step_closed_loop_dynamic_correction",
            "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
            "phase_one": phase_one,
            "decision": "CLOSED-LOOP UNSTABLE",
        }
        with (output_dir / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    corrections.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        corrections.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_checkpoint = output_dir / "best_closed_loop_correction.pt"
    best_val_loss = float("inf")
    best_epoch = None
    history = []
    gradient_report = {}
    for epoch in range(1, epochs + 1):
        train_loss = run_training_epoch(
            model,
            predictor,
            corrections,
            train_groups,
            optimizer,
            True,
            sigma,
            gradient_report,
        )
        with torch.no_grad():
            val_loss = run_training_epoch(
                model,
                predictor,
                corrections,
                val_groups,
                optimizer,
                False,
                sigma,
            )
        record = {
            "epoch": epoch,
            "train_closed_loop_mse": train_loss,
            "val_closed_loop_mse": val_loss,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "correction_state_dict": corrections.state_dict(),
                    "epoch": epoch,
                    "val_closed_loop_mse": val_loss,
                },
                best_checkpoint,
            )
    best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    corrections.load_state_dict(
        best_payload["correction_state_dict"], strict=True
    )
    metrics, closed_rows = evaluate_paths(
        model, predictor, corrections, val_groups, sigma
    )
    write_rows(output_dir / "formal_closed_loop_per_frame.csv", closed_rows)
    formal_phases, _ = phase_statistics(closed_rows)
    recovery = (metrics["closed"] - metrics["noisy"]) / (
        metrics["clean"] - metrics["noisy"]
    )
    summary = {
        "experiment": "kitti_step_closed_loop_dynamic_correction",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoints": {
            "static_host": str(static_checkpoint),
            "fixed_adapter": str(adapter_checkpoint),
            "fixed_predictor": str(predictor_checkpoint),
            "initial_fixed_dynamic_correction": str(correction_checkpoint),
            "best_closed_loop_correction": str(best_checkpoint),
        },
        "config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gaussian_noise_sigma": sigma,
            "dynamic_error_sample_time": DYNAMIC_ERROR_SAMPLE_TIME,
            "dynamic_error_time_constant": DYNAMIC_ERROR_TIME_CONSTANT,
            "dynamic_error_gain": DYNAMIC_ERROR_GAIN,
            "corrected_layers": [1, 4],
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in corrections.parameters()
            ),
        },
        "dataset": {
            "train_sequence_count": len(train_groups),
            "val_sequence_count": len(val_groups),
        },
        "phase_one": phase_one,
        "gradient_boundary": gradient_report,
        "history": history,
        "best": {"epoch": best_epoch, "val_mse": best_val_loss},
        "metrics": {
            "mIoU_clean_static": metrics["clean"],
            "mIoU_noisy_static": metrics["noisy"],
            "mIoU_continuous_prediction_only": metrics["prediction"],
            "mIoU_continuous_open_loop": metrics["open"],
            "mIoU_continuous_closed_loop": metrics["closed"],
            "closed_minus_open": metrics["closed"] - metrics["open"],
            "open_minus_prediction_only": metrics["open"] - metrics["prediction"],
            "closed_minus_prediction_only": metrics["closed"] - metrics["prediction"],
            "recovery": recovery,
        },
        "closed_loop_phase_statistics": formal_phases,
        "decision": decision_label(metrics["closed"], metrics["open"]),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
