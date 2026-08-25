import json
import os
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
)
from predify2021.mce_scores.evaluate_kitti_step_context_residual_ablation import (
    SEED,
    STATE_BATCH_SIZE,
    context_step,
    reset_rng,
    split_state,
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
from predify2021.model_factory.deeplabv3plus_resnet50 import BackboneFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.context_residual_correction import (
    ContextResidualCorrection,
)


CONTEXT_MASK = (False, False, True, True, True)
IGNORE_INDEX = 255
HISTORICAL_MIOU = {
    "clean": 0.6552125562,
    "noisy": 0.2940764905,
    "prediction_only": 0.2897704727,
    "legacy": 0.3011305694,
    "feature_mse_gain_dynamic_instant": 0.3157625378,
}


def split_backbone_features(features, index):
    return BackboneFeatures(
        *(value[index:index + 1] for value in features.as_tuple())
    )


def encoded_sequence_batches(model, samples, include_clean_state):
    for start in range(0, len(samples), STATE_BATCH_SIZE):
        batch_samples = samples[start:start + STATE_BATCH_SIZE]
        clean_batch = torch.cat([load_image(sample) for sample in batch_samples], dim=0)
        noisy_batch = add_frame_noise(clean_batch, SIGMA)
        with torch.no_grad():
            noisy_features = model.extract_backbone_features(noisy_batch)
            observations = model.encode_backbone_features(noisy_features)
            clean_states = None
            if include_clean_state:
                clean_states = model.encode_backbone_features(
                    model.extract_backbone_features(clean_batch)
                )
        rows = []
        for index, sample in enumerate(batch_samples):
            clean_state = (
                split_state(clean_states, index) if clean_states is not None else None
            )
            rows.append(
                (
                    sample,
                    split_state(observations, index),
                    split_backbone_features(noisy_features, index),
                    clean_state,
                    tuple(clean_batch.shape[-2:]),
                )
            )
        yield rows


def semantic_posterior(
    model,
    predictor,
    legacy,
    context,
    previous_previous,
    previous,
    dynamic_error,
    observation,
    noisy_features,
    output_size,
):
    with torch.no_grad():
        predicted, instant_error = predict_current(
            predictor, previous_previous, previous, observation
        )
        dynamic_error = update_dynamic_error(instant_error, dynamic_error)
        gain = legacy_gains(dynamic_error, legacy)
    posterior, new_gains, residuals = context_step(
        predicted,
        observation,
        instant_error,
        dynamic_error,
        gain,
        context,
        CONTEXT_MASK,
    )
    host_feature = corrected_host_feature(
        model, noisy_features, observation, posterior, output_size
    )
    logits = model.decode_from_host_feature(host_feature)
    return posterior, dynamic_error, gain, new_gains, residuals, logits


def run_epoch(model, predictor, legacy, context, groups, optimizer, training):
    model.eval()
    predictor.eval()
    legacy.eval()
    context.train(training)
    total_loss = 0.0
    frame_count = 0
    finite = True
    gradient_gate = None
    closed_loop_gate = False
    with torch.set_grad_enabled(training):
        for samples in groups.values():
            previous_previous = None
            previous = None
            previous_is_posterior = False
            dynamic_error = None
            for state_batch in encoded_sequence_batches(
                model, samples, include_clean_state=False
            ):
                for sample, observation, noisy_features, _, output_size in state_batch:
                    if previous is None:
                        previous = observation
                        continue
                    if previous_previous is None:
                        previous_previous = previous
                        previous = observation
                        continue
                    closed_loop_gate = closed_loop_gate or previous_is_posterior
                    posterior, dynamic_error, gain, new_gains, residuals, logits = (
                        semantic_posterior(
                            model,
                            predictor,
                            legacy,
                            context,
                            previous_previous,
                            previous,
                            dynamic_error,
                            observation,
                            noisy_features,
                            output_size,
                        )
                    )
                    target = semantic_mask_from_panoptic_png(sample["mask_path"])
                    target = target.unsqueeze(0).cuda(non_blocking=True)
                    loss = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
                    finite = finite and all(
                        torch.isfinite(value).all().item()
                        for value in (
                            loss,
                            posterior.z1,
                            posterior.z4,
                            new_gains[0],
                            new_gains[1],
                            residuals[0],
                            residuals[1],
                            logits,
                        )
                    )
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if gradient_gate is None:
                            gradient_gate = {
                                "context_has_nonzero_gradient": any(
                                    parameter.grad is not None
                                    and parameter.grad.abs().max().item() > 0.0
                                    for parameter in context.parameters()
                                ),
                                "host_trainable": any(
                                    parameter.requires_grad for parameter in model.parameters()
                                ),
                                "predictor_trainable": any(
                                    parameter.requires_grad for parameter in predictor.parameters()
                                ),
                                "legacy_trainable": any(
                                    parameter.requires_grad for parameter in legacy.parameters()
                                ),
                            }
                        optimizer.step()
                    total_loss += loss.detach().item()
                    frame_count += 1
                    dynamic_error = detach_state(dynamic_error)
                    previous_previous = previous
                    previous = detach_state(posterior)
                    previous_is_posterior = True
    return {
        "semantic_cross_entropy": total_loss / frame_count,
        "evaluated_frame_count": frame_count,
        "finite": finite,
        "gradient_gate": gradient_gate,
        "closed_loop_gate": closed_loop_gate,
    }


def evaluate_semantic(model, predictor, legacy, context, groups):
    confusion = torch.zeros((19, 19), dtype=torch.int64)
    total_semantic_loss = 0.0
    total_posterior_mse = 0.0
    gain_change = {"z1": 0.0, "z4": 0.0}
    logit_residual = {"z1": 0.0, "z4": 0.0}
    frame_count = 0
    finite = True
    model.eval()
    predictor.eval()
    legacy.eval()
    context.eval()
    with torch.no_grad():
        for samples in groups.values():
            previous_previous = None
            previous = None
            dynamic_error = None
            for state_batch in encoded_sequence_batches(
                model, samples, include_clean_state=True
            ):
                for sample, observation, noisy_features, clean_state, output_size in state_batch:
                    if previous is None:
                        previous = observation
                        continue
                    if previous_previous is None:
                        previous_previous = previous
                        previous = observation
                        continue
                    posterior, dynamic_error, gain, new_gains, residuals, logits = (
                        semantic_posterior(
                            model,
                            predictor,
                            legacy,
                            context,
                            previous_previous,
                            previous,
                            dynamic_error,
                            observation,
                            noisy_features,
                            output_size,
                        )
                    )
                    target = semantic_mask_from_panoptic_png(sample["mask_path"])
                    target_gpu = target.unsqueeze(0).cuda(non_blocking=True)
                    loss = F.cross_entropy(logits, target_gpu, ignore_index=IGNORE_INDEX)
                    prediction = logits.argmax(dim=1).squeeze(0).cpu().to(torch.int64)
                    update_confusion_matrix(confusion, prediction, target)
                    total_semantic_loss += loss.item()
                    total_posterior_mse += posterior_mse(posterior, clean_state).item()
                    for name, old_gain, new_gain, residual in zip(
                        ("z1", "z4"), gain, new_gains, residuals
                    ):
                        gain_change[name] += (new_gain - old_gain).abs().mean().item()
                        logit_residual[name] += residual.abs().mean().item()
                    finite = finite and all(
                        torch.isfinite(value).all().item()
                        for value in (
                            loss,
                            posterior.z1,
                            posterior.z4,
                            dynamic_error.z1,
                            dynamic_error.z4,
                            new_gains[0],
                            new_gains[1],
                            residuals[0],
                            residuals[1],
                            logits,
                        )
                    )
                    frame_count += 1
                    dynamic_error = detach_state(dynamic_error)
                    previous_previous = previous
                    previous = detach_state(posterior)
    return {
        "mIoU": float(torch.nanmean(compute_iou(confusion)).item()),
        "semantic_cross_entropy": total_semantic_loss / frame_count,
        "posterior_mse": total_posterior_mse / frame_count,
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
        "finite": finite,
        "evaluated_frame_count": frame_count,
    }


def decision_label(delta):
    if delta >= 0.010:
        return "SEMANTIC CORRECTION STRONG GO"
    if delta >= 0.005:
        return "SEMANTIC CORRECTION GO"
    if delta > 0.0:
        return "SEMANTIC CORRECTION WEAK GO"
    return "SEMANTIC CORRECTION NO-GO"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Semantic context residual correction expects GPU 0.")
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_SEMANTIC_CONTEXT_OUTPUT_DIR",
            "/home/lin/predify/experiments/kitti_step_semantic_context_residual",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reset_rng()
    model, predictor, legacy, _, checkpoints = load_components(root)
    context = torch.nn.ModuleList(
        [ContextResidualCorrection(), ContextResidualCorrection()]
    ).cuda()
    trainable_parameter_count = sum(
        parameter.numel() for parameter in context.parameters() if parameter.requires_grad
    )
    trainable_gate = {
        "trainable_parameter_count": trainable_parameter_count,
        "expected_parameter_count": 492288,
        "context_only": (
            trainable_parameter_count == 492288
            and not any(parameter.requires_grad for parameter in model.parameters())
            and not any(parameter.requires_grad for parameter in predictor.parameters())
            and not any(parameter.requires_grad for parameter in legacy.parameters())
        ),
    }
    if not trainable_gate["context_only"]:
        raise RuntimeError("Trainable parameter gate failed.")
    train_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_groups = sequence_groups(train_dataset)
    val_groups = sequence_groups(val_dataset)
    optimizer = torch.optim.AdamW(
        context.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_checkpoint = output_dir / "best_semantic_context_residual.pt"
    best_val_loss = float("inf")
    best_epoch = None
    history = []
    gradient_gate = None
    closed_loop_gate = False
    finite = True
    for epoch in range(1, EPOCHS + 1):
        reset_rng()
        train_result = run_epoch(
            model, predictor, legacy, context, train_groups, optimizer, training=True
        )
        reset_rng()
        val_result = run_epoch(
            model, predictor, legacy, context, val_groups, optimizer, training=False
        )
        gradient_gate = gradient_gate or train_result["gradient_gate"]
        closed_loop_gate = closed_loop_gate or train_result["closed_loop_gate"]
        finite = finite and train_result["finite"] and val_result["finite"]
        record = {
            "epoch": epoch,
            "train_semantic_cross_entropy": train_result["semantic_cross_entropy"],
            "val_semantic_cross_entropy": val_result["semantic_cross_entropy"],
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if val_result["semantic_cross_entropy"] < best_val_loss:
            best_val_loss = val_result["semantic_cross_entropy"]
            best_epoch = epoch
            torch.save(
                {
                    "context_state_dict": context.state_dict(),
                    "epoch": epoch,
                    "val_semantic_cross_entropy": best_val_loss,
                    "input_variant": "GAIN_DYNAMIC_INSTANT",
                },
                best_checkpoint,
            )
    best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    context.load_state_dict(best_payload["context_state_dict"], strict=True)
    reset_rng()
    final = evaluate_semantic(model, predictor, legacy, context, val_groups)
    delta_feature_mse = final["mIoU"] - HISTORICAL_MIOU[
        "feature_mse_gain_dynamic_instant"
    ]
    summary = {
        "experiment": "kitti_step_semantic_gain_dynamic_instant_correction",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoints": {
            **{name: str(path) for name, path in checkpoints.items()},
            "best_semantic_context_residual": str(best_checkpoint),
        },
        "config": {
            "seed": SEED,
            "gaussian_noise_sigma": SIGMA,
            "epochs": EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "CrossEntropyLoss(ignore_index=255)",
            "input_variant": "GAIN_DYNAMIC_INSTANT",
            "trainable_parameter_count": trainable_parameter_count,
        },
        "dataset": {
            "train_sequence_count": len(train_groups),
            "val_sequence_count": len(val_groups),
            "val_frame_count": len(val_dataset.samples),
            "val_evaluated_frame_count": final["evaluated_frame_count"],
        },
        "gates": {
            "trainable_parameters": trainable_gate,
            "gradient_path": gradient_gate,
            "closed_loop": closed_loop_gate,
            "finite": finite and final["finite"],
        },
        "history": history,
        "best": {
            "epoch": best_epoch,
            "val_semantic_cross_entropy": best_val_loss,
        },
        "historical_mIoU": HISTORICAL_MIOU,
        "semantic_result": {
            "mIoU": final["mIoU"],
            "semantic_minus_feature_mse": delta_feature_mse,
            "semantic_minus_legacy": final["mIoU"] - HISTORICAL_MIOU["legacy"],
            "val_semantic_cross_entropy": final["semantic_cross_entropy"],
            "val_posterior_mse": final["posterior_mse"],
            "diagnostics": final["diagnostics"],
            "finite": final["finite"],
        },
        "decision": decision_label(delta_feature_mse),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
