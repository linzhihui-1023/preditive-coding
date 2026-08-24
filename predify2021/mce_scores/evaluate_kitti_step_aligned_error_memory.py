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
    posterior_with_observation_passthrough,
    state_metrics,
    write_rows,
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
from predify2021.model_factory.deeplabv3plus_resnet50.local_error_alignment import (
    LocalErrorMemoryAlignment,
)


CLOSED_LOOP_CORRECTION_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_closed_loop_dynamic_correction_9820f4e/"
    "best_closed_loop_correction.pt"
)
ALPHA = 0.207
BETA = 0.793


def aligned_dynamic_error(
    error,
    previous_error,
    observation,
    previous_posterior,
    alignments=None,
    return_attention=False,
):
    if alignments is None:
        return update_dynamic_error(error, previous_error)
    if previous_error is None:
        previous_error = UnifiedFeatures(
            *(torch.zeros_like(value) for value in error.as_tuple())
        )
    aligned1 = alignments[0](
        observation.z1,
        previous_posterior.z1,
        previous_error.z1,
        return_attention=return_attention,
    )
    aligned4 = alignments[1](
        observation.z4,
        previous_posterior.z4,
        previous_error.z4,
        return_attention=return_attention,
    )
    if return_attention:
        aligned1, weights1, valid1 = aligned1
        aligned4, weights4, valid4 = aligned4
    state = UnifiedFeatures(
        ALPHA * error.z1 + BETA * aligned1,
        ALPHA * error.z2 + BETA * previous_error.z2,
        ALPHA * error.z3 + BETA * previous_error.z3,
        ALPHA * error.z4 + BETA * aligned4,
    )
    if return_attention:
        return state, ((weights1, valid1), (weights4, valid4))
    return state


def posterior_state(predicted, error, dynamic_error, observation, corrections):
    corrected = correction_states(predicted, dynamic_error, error, corrections)
    return posterior_with_observation_passthrough(corrected, observation)


def attention_diagnostics(weights, valid, window_size):
    maximum = weights.max(dim=1).values.mean().item()
    entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(
        dim=1
    )
    valid_count = valid.sum(dim=1)
    normalized_entropy = (entropy / valid_count.float().log()).mean().item()
    radius = window_size // 2
    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(-radius, radius + 1, device=weights.device),
            torch.arange(-radius, radius + 1, device=weights.device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    offset_magnitude = offsets.float().square().sum(dim=1).sqrt()
    selected = weights.argmax(dim=1)
    mean_offset = offset_magnitude[selected].mean().item()
    return maximum, normalized_entropy, mean_offset


def collect_aligned_rows(model, predictor, corrections, alignments, groups, sigma):
    rows = []
    finite = True
    diagnostic_sums = {
        "z1": {"max_weight": 0.0, "entropy": 0.0, "offset": 0.0},
        "z4": {"max_weight": 0.0, "entropy": 0.0, "offset": 0.0},
    }
    frame_count = 0
    model.eval()
    predictor.eval()
    corrections.eval()
    alignments.eval()
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
                dynamic_error, attention = aligned_dynamic_error(
                    error,
                    dynamic_error,
                    observation,
                    previous,
                    alignments,
                    return_attention=True,
                )
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
                finite = finite and all(
                    torch.isfinite(value).all().item() for value in tensors
                )
                row = {
                    "sequence_id": sequence_id,
                    "frame_id": sample["frame_id"],
                    "frame_index": frame_index,
                }
                row.update(state_metrics(error, dynamic_error, posterior, observation))
                rows.append(row)
                for name, (weights, valid) in zip(("z1", "z4"), attention):
                    maximum, entropy, offset = attention_diagnostics(
                        weights, valid, alignments[0].window_size
                    )
                    diagnostic_sums[name]["max_weight"] += maximum
                    diagnostic_sums[name]["entropy"] += entropy
                    diagnostic_sums[name]["offset"] += offset
                frame_count += 1
                dynamic_error = detach_state(dynamic_error)
                previous_previous = previous
                previous = detach_state(posterior)
    diagnostics = {
        name: {
            "mean_max_attention_weight": values["max_weight"] / frame_count,
            "normalized_attention_entropy": values["entropy"] / frame_count,
            "mean_argmax_offset_magnitude": values["offset"] / frame_count,
        }
        for name, values in diagnostic_sums.items()
    }
    return rows, finite, diagnostics


def run_training_epoch(
    model,
    predictor,
    corrections,
    alignments,
    groups,
    optimizer,
    training,
    sigma,
    gradient_report=None,
    instability_report=None,
):
    model.eval()
    predictor.eval()
    corrections.eval()
    alignments.train(training)
    total_loss = 0.0
    frame_count = 0
    for samples in groups.values():
        previous_previous = None
        previous = None
        dynamic_error = None
        initial_metrics = None
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
            dynamic_error = aligned_dynamic_error(
                error,
                dynamic_error,
                observation,
                previous,
                alignments,
            )
            posterior = posterior_state(
                predicted, error, dynamic_error, observation, corrections
            )
            current_metrics = state_metrics(
                error, dynamic_error, posterior, observation
            )
            values_finite = all(
                torch.isfinite(value).all().item()
                for value in (
                    error.z1,
                    error.z4,
                    dynamic_error.z1,
                    dynamic_error.z4,
                    posterior.z1,
                    posterior.z4,
                )
            )
            if initial_metrics is None:
                initial_metrics = current_metrics
            unstable_metrics = {
                name: {
                    "initial": initial_metrics[name],
                    "current": value,
                    "ratio": value / initial_metrics[name],
                }
                for name, value in current_metrics.items()
                if value > 10.0 * initial_metrics[name]
            }
            if instability_report is not None and (
                not values_finite or unstable_metrics
            ):
                instability_report.update(
                    {
                        "sequence_id": sample["sequence_id"],
                        "frame_id": sample["frame_id"],
                        "finite": values_finite,
                        "unstable_metrics": unstable_metrics,
                    }
                )
                return total_loss / max(frame_count, 1)
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
                            "correction_has_gradient": any(
                                parameter.grad is not None
                                for parameter in corrections.parameters()
                            ),
                            "alignment_gradient_tensor_count": sum(
                                parameter.grad is not None
                                for parameter in alignments.parameters()
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


def evaluate_paths(model, predictor, corrections, alignments, groups, sigma):
    confusion = {
        name: torch.zeros((19, 19), dtype=torch.int64)
        for name in ("clean", "noisy", "prediction", "baseline", "aligned")
    }
    diagnostic_sums = {
        "z1": {"max_weight": 0.0, "entropy": 0.0, "offset": 0.0},
        "z4": {"max_weight": 0.0, "entropy": 0.0, "offset": 0.0},
    }
    aligned_rows = []
    frame_count = 0
    finite = True
    model.eval()
    predictor.eval()
    corrections.eval()
    alignments.eval()
    with torch.no_grad():
        for sequence_id, samples in groups.items():
            observation_previous_previous = None
            observation_previous = None
            baseline_previous_previous = None
            baseline_previous = None
            aligned_previous_previous = None
            aligned_previous = None
            baseline_error = None
            aligned_error = None
            for frame_index, sample in enumerate(samples):
                clean_image = load_image(sample)
                noisy_image = add_frame_noise(clean_image, sigma)
                observation = encode_image(model, noisy_image)
                if observation_previous is None:
                    observation_previous = observation
                    baseline_previous = observation
                    aligned_previous = observation
                    continue
                if observation_previous_previous is None:
                    observation_previous_previous = observation_previous
                    observation_previous = observation
                    baseline_previous_previous = baseline_previous
                    baseline_previous = observation
                    aligned_previous_previous = aligned_previous
                    aligned_previous = observation
                    continue
                prediction_prior, _ = predict_current(
                    predictor,
                    observation_previous_previous,
                    observation_previous,
                    observation,
                )
                baseline_prior, baseline_instant = predict_current(
                    predictor,
                    baseline_previous_previous,
                    baseline_previous,
                    observation,
                )
                aligned_prior, aligned_instant = predict_current(
                    predictor,
                    aligned_previous_previous,
                    aligned_previous,
                    observation,
                )
                baseline_error = aligned_dynamic_error(
                    baseline_instant,
                    baseline_error,
                    observation,
                    baseline_previous,
                    alignments=None,
                )
                aligned_error, attention = aligned_dynamic_error(
                    aligned_instant,
                    aligned_error,
                    observation,
                    aligned_previous,
                    alignments,
                    return_attention=True,
                )
                baseline_posterior = posterior_state(
                    baseline_prior,
                    baseline_instant,
                    baseline_error,
                    observation,
                    corrections,
                )
                aligned_posterior = posterior_state(
                    aligned_prior,
                    aligned_instant,
                    aligned_error,
                    observation,
                    corrections,
                )
                finite = finite and all(
                    torch.isfinite(value).all().item()
                    for value in (
                        aligned_instant.z1,
                        aligned_instant.z4,
                        aligned_error.z1,
                        aligned_error.z4,
                        aligned_posterior.z1,
                        aligned_posterior.z4,
                    )
                )
                noisy_features = model.extract_backbone_features(noisy_image)
                output_size = tuple(clean_image.shape[-2:])
                prediction_host = corrected_host_feature(
                    model,
                    noisy_features,
                    observation,
                    prediction_prior,
                    output_size,
                )
                baseline_host = corrected_host_feature(
                    model,
                    noisy_features,
                    observation,
                    baseline_posterior,
                    output_size,
                )
                aligned_host = corrected_host_feature(
                    model,
                    noisy_features,
                    observation,
                    aligned_posterior,
                    output_size,
                )
                logits = {
                    "clean": model(clean_image),
                    "noisy": model(noisy_image),
                    "prediction": model.decode_from_host_feature(prediction_host),
                    "baseline": model.decode_from_host_feature(baseline_host),
                    "aligned": model.decode_from_host_feature(aligned_host),
                }
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name, value in logits.items():
                    prediction = value.argmax(dim=1).squeeze(0).cpu().to(torch.int64)
                    update_confusion_matrix(confusion[name], prediction, mask)
                for name, (weights, valid) in zip(("z1", "z4"), attention):
                    maximum, entropy, offset = attention_diagnostics(
                        weights, valid, alignments[0].window_size
                    )
                    diagnostic_sums[name]["max_weight"] += maximum
                    diagnostic_sums[name]["entropy"] += entropy
                    diagnostic_sums[name]["offset"] += offset
                row = {
                    "sequence_id": sequence_id,
                    "frame_id": sample["frame_id"],
                    "frame_index": frame_index,
                }
                row.update(
                    state_metrics(
                        aligned_instant,
                        aligned_error,
                        aligned_posterior,
                        observation,
                    )
                )
                aligned_rows.append(row)
                frame_count += 1
                baseline_error = detach_state(baseline_error)
                aligned_error = detach_state(aligned_error)
                observation_previous_previous = observation_previous
                observation_previous = observation
                baseline_previous_previous = baseline_previous
                baseline_previous = detach_state(baseline_posterior)
                aligned_previous_previous = aligned_previous
                aligned_previous = detach_state(aligned_posterior)
    metrics = {
        name: float(torch.nanmean(compute_iou(value)).item())
        for name, value in confusion.items()
    }
    diagnostics = {
        name: {
            "mean_max_attention_weight": values["max_weight"] / frame_count,
            "normalized_attention_entropy": values["entropy"] / frame_count,
            "mean_argmax_offset_magnitude": values["offset"] / frame_count,
        }
        for name, values in diagnostic_sums.items()
    }
    return metrics, diagnostics, aligned_rows, finite


def decision_label(aligned_miou, baseline_miou):
    improvement = aligned_miou - baseline_miou
    if improvement <= 0.0:
        return "NO-GO"
    if improvement < 0.003:
        return "WEAK GO"
    if improvement < 0.005741:
        return "GO"
    return "STRONG GO"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Local error-memory alignment expects GPU 0.")
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
            "PREDIFY_KITTI_STEP_CLOSED_LOOP_CORRECTION_CHECKPOINT",
            CLOSED_LOOP_CORRECTION_CHECKPOINT_DEFAULT,
        )
    )
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_ERROR_ALIGNMENT_OUTPUT_DIR",
            "/home/lin/predify/experiments/kitti_step_error_memory_alignment",
        )
    )
    epochs = int(os.environ.get("PREDIFY_KITTI_STEP_ERROR_ALIGNMENT_EPOCHS", "3"))
    learning_rate = float(
        os.environ.get("PREDIFY_KITTI_STEP_ERROR_ALIGNMENT_LR", "0.0001")
    )
    weight_decay = float(
        os.environ.get("PREDIFY_KITTI_STEP_ERROR_ALIGNMENT_WEIGHT_DECAY", "0.01")
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
    alignments = torch.nn.ModuleList(
        [LocalErrorMemoryAlignment(), LocalErrorMemoryAlignment()]
    ).cuda()
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
    stability_rows, finite, stability_attention = collect_aligned_rows(
        model, predictor, corrections, alignments, stability_groups, sigma
    )
    stability_phases, ratio_stable = phase_statistics(stability_rows)
    stable = finite and ratio_stable
    write_rows(output_dir / "stability_per_frame.csv", stability_rows)
    phase_one = {
        "selected_sequences": selected_ids,
        "finite": finite,
        "stable": stable,
        "attention_diagnostics": stability_attention,
        "phase_statistics": stability_phases,
    }
    print(json.dumps({"phase_one_stable": stable}, sort_keys=True), flush=True)
    if not stable:
        summary = {
            "experiment": "kitti_step_local_error_memory_alignment",
            "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
            "phase_one": phase_one,
            "decision": "ALIGNED CLOSED-LOOP UNSTABLE",
        }
        with (output_dir / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(
        alignments.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_checkpoint = output_dir / "best_local_error_alignment.pt"
    best_val_loss = float("inf")
    best_epoch = None
    history = []
    gradient_report = {}
    training_instability = None
    for epoch in range(1, epochs + 1):
        epoch_instability = {}
        train_loss = run_training_epoch(
            model,
            predictor,
            corrections,
            alignments,
            train_groups,
            optimizer,
            True,
            sigma,
            gradient_report,
            epoch_instability,
        )
        if epoch_instability:
            epoch_instability["epoch"] = epoch
            training_instability = epoch_instability
            print(json.dumps({"training_instability": epoch_instability}, sort_keys=True), flush=True)
            break
        with torch.no_grad():
            val_loss = run_training_epoch(
                model,
                predictor,
                corrections,
                alignments,
                val_groups,
                optimizer,
                False,
                sigma,
            )
        record = {
            "epoch": epoch,
            "train_alignment_mse": train_loss,
            "val_alignment_mse": val_loss,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "alignment_state_dict": alignments.state_dict(),
                    "epoch": epoch,
                    "val_alignment_mse": val_loss,
                },
                best_checkpoint,
            )
    if training_instability is not None:
        summary = {
            "experiment": "kitti_step_local_error_memory_alignment",
            "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
            "config": {
                "gaussian_noise_sigma": sigma,
                "alpha": ALPHA,
                "beta": BETA,
                "window_size": 7,
                "query_key_channels": 32,
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in alignments.parameters()
                ),
            },
            "phase_one": phase_one,
            "gradient_boundary": gradient_report,
            "training_instability": training_instability,
            "decision": "ALIGNED CLOSED-LOOP UNSTABLE",
        }
        with (output_dir / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return
    best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    alignments.load_state_dict(best_payload["alignment_state_dict"], strict=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    metrics, diagnostics, aligned_rows, formal_finite = evaluate_paths(
        model, predictor, corrections, alignments, val_groups, sigma
    )
    write_rows(output_dir / "formal_aligned_per_frame.csv", aligned_rows)
    aligned_phases, formal_ratio_stable = phase_statistics(aligned_rows)
    formal_stable = formal_finite and formal_ratio_stable
    recovery = (metrics["aligned"] - metrics["noisy"]) / (
        metrics["clean"] - metrics["noisy"]
    )
    summary = {
        "experiment": "kitti_step_local_error_memory_alignment",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoints": {
            "static_host": str(static_checkpoint),
            "fixed_adapter": str(adapter_checkpoint),
            "fixed_predictor": str(predictor_checkpoint),
            "fixed_closed_loop_correction": str(correction_checkpoint),
            "best_alignment": str(best_checkpoint),
        },
        "config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gaussian_noise_sigma": sigma,
            "alpha": ALPHA,
            "beta": BETA,
            "window_size": 7,
            "query_key_channels": 32,
            "aligned_layers": [1, 4],
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in alignments.parameters()
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
            "mIoU_closed_loop_no_alignment": metrics["baseline"],
            "mIoU_closed_loop_local_alignment": metrics["aligned"],
            "alignment_gain": metrics["aligned"] - metrics["baseline"],
            "aligned_minus_prediction": metrics["aligned"] - metrics["prediction"],
            "aligned_minus_noisy": metrics["aligned"] - metrics["noisy"],
            "recovery": recovery,
        },
        "attention_diagnostics": diagnostics,
        "aligned_phase_statistics": aligned_phases,
        "formal_stability": {
            "finite": formal_finite,
            "stable": formal_stable,
        },
        "decision": (
            decision_label(metrics["aligned"], metrics["baseline"])
            if formal_stable
            else "ALIGNED CLOSED-LOOP UNSTABLE"
        ),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
