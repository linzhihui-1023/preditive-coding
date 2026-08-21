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
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    INSTANT_CORRECTION_CHECKPOINT_DEFAULT,
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
from predify2021.model_factory.deeplabv3plus_resnet50.adaptive_dynamics import (
    AdaptiveUpdateRate,
)
from predify2021.model_factory.deeplabv3plus_resnet50.corrections import (
    ErrorGainCorrection,
)


FIXED_DYNAMIC_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_dynamic_error_correction_9fc8d81/"
    "best_dynamic_correction.pt"
)


def update_adaptive_error(error, predicted, previous_state, dynamics):
    if previous_state is None:
        previous_state = UnifiedFeatures(
            *(torch.zeros_like(value) for value in error.as_tuple())
        )
    gamma1 = dynamics[0](error.z1, predicted.z1, previous_state.z1)
    gamma4 = dynamics[1](error.z4, predicted.z4, previous_state.z4)
    state = UnifiedFeatures(
        gamma1 * error.z1 + (1.0 - gamma1) * previous_state.z1,
        torch.zeros_like(error.z2),
        torch.zeros_like(error.z3),
        gamma4 * error.z4 + (1.0 - gamma4) * previous_state.z4,
    )
    return state, (gamma1, gamma4)


def detach_state(state):
    return UnifiedFeatures(*(value.detach() for value in state.as_tuple()))


def run_epoch(model, predictor, dynamics, corrections, groups, optimizer, training, sigma):
    model.eval()
    predictor.eval()
    dynamics.train(training)
    corrections.train(training)
    total_loss = 0.0
    frame_count = 0
    for samples in groups.values():
        previous_previous = None
        previous = None
        adaptive_state = None
        for sample in samples:
            clean_image = load_image(sample)
            clean_state = encode_image(model, clean_image)
            if previous is None:
                previous = clean_state
                continue
            if previous_previous is None:
                previous_previous = previous
                previous = clean_state
                continue
            noisy_image = torch.clamp(
                clean_image + sigma * torch.randn_like(clean_image), 0.0, 1.0
            )
            noisy_state = encode_image(model, noisy_image)
            with torch.no_grad():
                predicted, error = predict_current(
                    predictor, previous_previous, previous, noisy_state
                )
            adaptive_state, _ = update_adaptive_error(
                error, predicted, adaptive_state, dynamics
            )
            corrected = correction_states(
                predicted, adaptive_state, error, corrections
            )
            loss = F.mse_loss(corrected.z1, clean_state.z1) + F.mse_loss(
                corrected.z4, clean_state.z4
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.detach().item()
            frame_count += 1
            adaptive_state = detach_state(adaptive_state)
            previous_previous = previous
            previous = clean_state
    return total_loss / frame_count


def gamma_statistics(model, predictor, dynamics, groups, sigma):
    totals = [{"sum": 0.0, "sum_square": 0.0, "count": 0} for _ in range(2)]
    model.eval()
    predictor.eval()
    dynamics.eval()
    with torch.no_grad():
        for samples in groups.values():
            previous_previous = None
            previous = None
            adaptive_state = None
            for sample in samples:
                clean_image = load_image(sample)
                clean_state = encode_image(model, clean_image)
                if previous is None:
                    previous = clean_state
                    continue
                if previous_previous is None:
                    previous_previous = previous
                    previous = clean_state
                    continue
                noisy_image = torch.clamp(
                    clean_image + sigma * torch.randn_like(clean_image), 0.0, 1.0
                )
                noisy_state = encode_image(model, noisy_image)
                predicted, error = predict_current(
                    predictor, previous_previous, previous, noisy_state
                )
                adaptive_state, gammas = update_adaptive_error(
                    error, predicted, adaptive_state, dynamics
                )
                for total, gamma in zip(totals, gammas):
                    total["sum"] += gamma.double().sum().item()
                    total["sum_square"] += gamma.double().square().sum().item()
                    total["count"] += gamma.numel()
                adaptive_state = detach_state(adaptive_state)
                previous_previous = previous
                previous = clean_state
    statistics = []
    for total in totals:
        mean = total["sum"] / total["count"]
        variance = total["sum_square"] / total["count"] - mean * mean
        statistics.append({"mean": mean, "std": max(variance, 0.0) ** 0.5})
    return {"z1": statistics[0], "z4": statistics[1]}


def evaluate(
    model,
    predictor,
    dynamics,
    adaptive_corrections,
    instant_corrections,
    fixed_corrections,
    groups,
    sigma,
):
    confusion = {
        name: torch.zeros((19, 19), dtype=torch.int64)
        for name in ("clean", "noisy", "pred", "inst", "fixed", "adaptive")
    }
    gamma_totals = [{"sum": 0.0, "sum_square": 0.0, "count": 0} for _ in range(2)]
    error_sum = {"z1": 0.0, "z4": 0.0}
    state_sum = {"z1": 0.0, "z4": 0.0}
    frame_count = 0
    model.eval()
    predictor.eval()
    dynamics.eval()
    adaptive_corrections.eval()
    instant_corrections.eval()
    fixed_corrections.eval()
    with torch.no_grad():
        for samples in groups.values():
            previous_previous = None
            previous = None
            fixed_state = None
            adaptive_state = None
            for sample in samples:
                clean_image = load_image(sample)
                clean_state = encode_image(model, clean_image)
                if previous is None:
                    previous = clean_state
                    continue
                if previous_previous is None:
                    previous_previous = previous
                    previous = clean_state
                    continue
                noisy_image = torch.clamp(
                    clean_image + sigma * torch.randn_like(clean_image), 0.0, 1.0
                )
                noisy_state = encode_image(model, noisy_image)
                predicted, error = predict_current(
                    predictor, previous_previous, previous, noisy_state
                )
                fixed_state = update_dynamic_error(error, fixed_state)
                adaptive_state, gammas = update_adaptive_error(
                    error, predicted, adaptive_state, dynamics
                )
                instant_state = correction_states(
                    predicted, error, error, instant_corrections
                )
                fixed_corrected = correction_states(
                    predicted, fixed_state, error, fixed_corrections
                )
                adaptive_corrected = correction_states(
                    predicted, adaptive_state, error, adaptive_corrections
                )
                noisy_features = model.extract_backbone_features(noisy_image)
                output_size = tuple(clean_image.shape[-2:])
                pred_host = corrected_host_feature(
                    model, noisy_features, noisy_state, predicted, output_size
                )
                instant_host = corrected_host_feature(
                    model, noisy_features, noisy_state, instant_state, output_size
                )
                fixed_host = corrected_host_feature(
                    model, noisy_features, noisy_state, fixed_corrected, output_size
                )
                adaptive_host = corrected_host_feature(
                    model, noisy_features, noisy_state, adaptive_corrected, output_size
                )
                logits = {
                    "clean": model(clean_image),
                    "noisy": model(noisy_image),
                    "pred": model.decode_from_host_feature(pred_host),
                    "inst": model.decode_from_host_feature(instant_host),
                    "fixed": model.decode_from_host_feature(fixed_host),
                    "adaptive": model.decode_from_host_feature(adaptive_host),
                }
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name, value in logits.items():
                    prediction = value.argmax(dim=1).squeeze(0).cpu().to(torch.int64)
                    update_confusion_matrix(confusion[name], prediction, mask)
                for total, gamma in zip(gamma_totals, gammas):
                    total["sum"] += gamma.double().sum().item()
                    total["sum_square"] += gamma.double().square().sum().item()
                    total["count"] += gamma.numel()
                error_sum["z1"] += error.z1.abs().mean().item()
                error_sum["z4"] += error.z4.abs().mean().item()
                state_sum["z1"] += adaptive_state.z1.abs().mean().item()
                state_sum["z4"] += adaptive_state.z4.abs().mean().item()
                frame_count += 1
                fixed_state = detach_state(fixed_state)
                adaptive_state = detach_state(adaptive_state)
                previous_previous = previous
                previous = clean_state
    metrics = {
        name: float(torch.nanmean(compute_iou(value)).item())
        for name, value in confusion.items()
    }
    gamma_statistics_result = {}
    for name, total in zip(("z1", "z4"), gamma_totals):
        mean = total["sum"] / total["count"]
        variance = total["sum_square"] / total["count"] - mean * mean
        gamma_statistics_result[name] = {
            "mean": mean,
            "std": max(variance, 0.0) ** 0.5,
        }
    diagnostics = {
        "gamma": gamma_statistics_result,
        "mean_abs_error": {
            name: value / frame_count for name, value in error_sum.items()
        },
        "mean_abs_adaptive_state": {
            name: value / frame_count for name, value in state_sum.items()
        },
    }
    return metrics, diagnostics


def decision_label(adaptive_miou, clean_miou):
    if adaptive_miou <= 0.388578 or clean_miou < 0.6532:
        return "NO-GO"
    if adaptive_miou < 0.390337:
        return "WEAK GO"
    return "GO"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Adaptive dynamic correction training expects GPU 0.")
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
    instant_checkpoint = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_INSTANT_CORRECTION_CHECKPOINT",
            INSTANT_CORRECTION_CHECKPOINT_DEFAULT,
        )
    )
    fixed_checkpoint = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_FIXED_DYNAMIC_CHECKPOINT",
            FIXED_DYNAMIC_CHECKPOINT_DEFAULT,
        )
    )
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_ADAPTIVE_DYNAMIC_OUTPUT_DIR",
            "/home/lin/predify/experiments/kitti_step_adaptive_dynamic_correction",
        )
    )
    epochs = int(os.environ.get("PREDIFY_KITTI_STEP_ADAPTIVE_DYNAMIC_EPOCHS", "3"))
    learning_rate = float(
        os.environ.get("PREDIFY_KITTI_STEP_ADAPTIVE_DYNAMIC_LR", "0.0001")
    )
    weight_decay = float(
        os.environ.get("PREDIFY_KITTI_STEP_ADAPTIVE_DYNAMIC_WEIGHT_DECAY", "0.01")
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
    dynamics = torch.nn.ModuleList([AdaptiveUpdateRate(), AdaptiveUpdateRate()]).cuda()
    adaptive_corrections = torch.nn.ModuleList(
        [ErrorGainCorrection(), ErrorGainCorrection()]
    ).cuda()
    instant_corrections = torch.nn.ModuleList(
        [ErrorGainCorrection(), ErrorGainCorrection()]
    ).cuda()
    instant_payload = torch.load(instant_checkpoint, map_location="cpu", weights_only=False)
    instant_corrections.load_state_dict(instant_payload["correction_state_dict"], strict=True)
    fixed_corrections = torch.nn.ModuleList(
        [ErrorGainCorrection(), ErrorGainCorrection()]
    ).cuda()
    fixed_payload = torch.load(fixed_checkpoint, map_location="cpu", weights_only=False)
    fixed_corrections.load_state_dict(fixed_payload["correction_state_dict"], strict=True)
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    instant_corrections.requires_grad_(False)
    fixed_corrections.requires_grad_(False)
    trainable_parameters = list(dynamics.parameters()) + list(
        adaptive_corrections.parameters()
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=learning_rate, weight_decay=weight_decay
    )
    train_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_groups = sequence_groups(train_dataset)
    val_groups = sequence_groups(val_dataset)
    initial_gamma = gamma_statistics(model, predictor, dynamics, val_groups, sigma)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "best_adaptive_dynamic_correction.pt"
    best_val_loss = float("inf")
    best_epoch = None
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model,
            predictor,
            dynamics,
            adaptive_corrections,
            train_groups,
            optimizer,
            True,
            sigma,
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model,
                predictor,
                dynamics,
                adaptive_corrections,
                val_groups,
                optimizer,
                False,
                sigma,
            )
        record = {
            "epoch": epoch,
            "train_adaptive_dynamic_mse": train_loss,
            "val_adaptive_dynamic_mse": val_loss,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "dynamics_state_dict": dynamics.state_dict(),
                    "correction_state_dict": adaptive_corrections.state_dict(),
                    "epoch": epoch,
                    "val_adaptive_dynamic_mse": val_loss,
                },
                best_checkpoint,
            )
    best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    dynamics.load_state_dict(best_payload["dynamics_state_dict"], strict=True)
    adaptive_corrections.load_state_dict(
        best_payload["correction_state_dict"], strict=True
    )
    metrics, diagnostics = evaluate(
        model,
        predictor,
        dynamics,
        adaptive_corrections,
        instant_corrections,
        fixed_corrections,
        val_groups,
        sigma,
    )
    recovery = (metrics["adaptive"] - metrics["noisy"]) / (
        metrics["clean"] - metrics["noisy"]
    )
    summary = {
        "experiment": "kitti_step_adaptive_dynamic_error_correction",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoints": {
            "static_host": str(static_checkpoint),
            "fixed_adapter": str(adapter_checkpoint),
            "fixed_predictor": str(predictor_checkpoint),
            "instant_correction": str(instant_checkpoint),
            "fixed_dynamic_correction": str(fixed_checkpoint),
            "best_adaptive_dynamic": str(best_checkpoint),
        },
        "config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gaussian_noise_sigma": sigma,
            "initial_update_rate": 0.207,
            "corrected_layers": [1, 4],
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in trainable_parameters
            ),
        },
        "dataset": {
            "train_sequence_count": len(train_groups),
            "val_sequence_count": len(val_groups),
        },
        "initial_gamma": initial_gamma,
        "history": history,
        "best": {"epoch": best_epoch, "val_mse": best_val_loss},
        "metrics": {
            "mIoU_clean": metrics["clean"],
            "mIoU_noisy": metrics["noisy"],
            "mIoU_prediction_only": metrics["pred"],
            "mIoU_instantaneous": metrics["inst"],
            "mIoU_fixed_dynamic": metrics["fixed"],
            "mIoU_adaptive_dynamic": metrics["adaptive"],
            "adaptive_minus_prediction_only": metrics["adaptive"] - metrics["pred"],
            "adaptive_minus_instantaneous": metrics["adaptive"] - metrics["inst"],
            "adaptive_minus_fixed_dynamic": metrics["adaptive"] - metrics["fixed"],
            "recovery": recovery,
        },
        "diagnostics": diagnostics,
        "decision": decision_label(metrics["adaptive"], metrics["clean"]),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
