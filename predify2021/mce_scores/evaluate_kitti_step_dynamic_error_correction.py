import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    pil_rgb_to_unit_tensor,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_error_correction import (
    correction_states,
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


STATIC_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/checkpoints/kitti_step_static_deeplabv3plus_epoch1/"
    "best_kitti_step_static_deeplabv3plus.pt"
)
ADAPTER_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_state_predictor_d655a0d/"
    "best_state_predictor.pt"
)
PREDICTOR_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_fixed_adapter_predictor_41aa5cc/"
    "best_predictor.pt"
)
INSTANT_CORRECTION_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_error_correction_b98805a/"
    "best_correction.pt"
)

ALPHA = 0.207
BETA = 0.793


def sequence_groups(dataset):
    groups = {}
    for sample in dataset.samples:
        groups.setdefault(sample["sequence_id"], []).append(sample)
    for samples in groups.values():
        samples.sort(key=lambda sample: int(sample["frame_id"]))
    return groups


def load_image(sample):
    with Image.open(Path(sample["image_path"])) as opened_image:
        return pil_rgb_to_unit_tensor(opened_image.convert("RGB")).unsqueeze(0).cuda()


def encode_image(model, image):
    with torch.no_grad():
        features = model.extract_backbone_features(image)
        return model.encode_backbone_features(features)


def predict_current(predictor, previous_previous, previous, current):
    previous_delta = UnifiedFeatures(
        *(new - old for new, old in zip(previous.as_tuple(), previous_previous.as_tuple()))
    )
    predicted_delta = predictor(previous, previous_delta)
    predicted = UnifiedFeatures(
        *(state + delta for state, delta in zip(previous.as_tuple(), predicted_delta.as_tuple()))
    )
    error = UnifiedFeatures(
        *(observed - prediction for observed, prediction in zip(current.as_tuple(), predicted.as_tuple()))
    )
    return predicted, error


def update_dynamic_error(error, previous_error):
    if previous_error is None:
        previous_error = UnifiedFeatures(*(torch.zeros_like(value) for value in error.as_tuple()))
    return UnifiedFeatures(
        *(ALPHA * current + BETA * previous
          for current, previous in zip(error.as_tuple(), previous_error.as_tuple()))
    )


def run_epoch(model, predictor, corrections, groups, optimizer, training, sigma):
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
                dynamic_error = update_dynamic_error(error, dynamic_error)
            corrected = correction_states(predicted, error, corrections)
            loss = F.mse_loss(corrected.z1, clean_state.z1) + F.mse_loss(
                corrected.z4, clean_state.z4
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.detach().item()
            frame_count += 1
            dynamic_error = UnifiedFeatures(
                *(value.detach() for value in dynamic_error.as_tuple())
            )
            previous_previous = previous
            previous = clean_state
    return total_loss / frame_count


def evaluate(model, predictor, dynamic_corrections, instant_corrections, groups, sigma):
    confusion = {
        name: torch.zeros((19, 19), dtype=torch.int64)
        for name in ("clean", "noisy", "pred", "inst", "dyn")
    }
    error_sum = {"z1": 0.0, "z4": 0.0}
    dynamic_sum = {"z1": 0.0, "z4": 0.0}
    frame_count = 0
    model.eval()
    predictor.eval()
    dynamic_corrections.eval()
    instant_corrections.eval()
    with torch.no_grad():
        for samples in groups.values():
            previous_previous = None
            previous = None
            dynamic_error = None
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
                dynamic_error = update_dynamic_error(error, dynamic_error)
                instant_state = correction_states(
                    predicted, error, instant_corrections
                )
                dynamic_state = correction_states(
                    predicted, error, dynamic_corrections
                )
                noisy_features = model.extract_backbone_features(noisy_image)
                output_size = tuple(clean_image.shape[-2:])
                pred_host = corrected_host_feature(
                    model, noisy_features, noisy_state, predicted, output_size
                )
                instant_host = corrected_host_feature(
                    model, noisy_features, noisy_state, instant_state, output_size
                )
                dynamic_host = corrected_host_feature(
                    model, noisy_features, noisy_state, dynamic_state, output_size
                )
                logits = {
                    "clean": model(clean_image),
                    "noisy": model(noisy_image),
                    "pred": model.decode_from_host_feature(pred_host),
                    "inst": model.decode_from_host_feature(instant_host),
                    "dyn": model.decode_from_host_feature(dynamic_host),
                }
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name, value in logits.items():
                    prediction = value.argmax(dim=1).squeeze(0).cpu().to(torch.int64)
                    update_confusion_matrix(confusion[name], prediction, mask)
                error_sum["z1"] += error.z1.abs().mean().item()
                error_sum["z4"] += error.z4.abs().mean().item()
                dynamic_sum["z1"] += dynamic_error.z1.abs().mean().item()
                dynamic_sum["z4"] += dynamic_error.z4.abs().mean().item()
                frame_count += 1
                dynamic_error = UnifiedFeatures(
                    *(value.detach() for value in dynamic_error.as_tuple())
                )
                previous_previous = previous
                previous = clean_state
    metrics = {
        name: float(torch.nanmean(compute_iou(value)).item())
        for name, value in confusion.items()
    }
    diagnostics = {
        "mean_abs_error": {name: value / frame_count for name, value in error_sum.items()},
        "mean_abs_dynamic_error": {
            name: value / frame_count for name, value in dynamic_sum.items()
        },
    }
    return metrics, diagnostics


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Dynamic error correction training expects GPU 0.")
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
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_DYNAMIC_ERROR_OUTPUT_DIR",
            "/home/lin/predify/experiments/kitti_step_dynamic_error_correction",
        )
    )
    epochs = int(os.environ.get("PREDIFY_KITTI_STEP_DYNAMIC_ERROR_EPOCHS", "3"))
    learning_rate = float(os.environ.get("PREDIFY_KITTI_STEP_DYNAMIC_ERROR_LR", "0.0001"))
    weight_decay = float(
        os.environ.get("PREDIFY_KITTI_STEP_DYNAMIC_ERROR_WEIGHT_DECAY", "0.01")
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
    dynamic_corrections = torch.nn.ModuleList(
        [ErrorGainCorrection(), ErrorGainCorrection()]
    ).cuda()
    instant_corrections = torch.nn.ModuleList(
        [ErrorGainCorrection(), ErrorGainCorrection()]
    ).cuda()
    instant_payload = torch.load(instant_checkpoint, map_location="cpu", weights_only=False)
    instant_corrections.load_state_dict(instant_payload["correction_state_dict"], strict=True)
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        dynamic_corrections.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_groups = sequence_groups(train_dataset)
    val_groups = sequence_groups(val_dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "best_dynamic_correction.pt"
    best_val_loss = float("inf")
    best_epoch = None
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model, predictor, dynamic_corrections, train_groups, optimizer, True, sigma
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model, predictor, dynamic_corrections, val_groups, optimizer, False, sigma
            )
        record = {
            "epoch": epoch,
            "train_dynamic_correction_mse": train_loss,
            "val_dynamic_correction_mse": val_loss,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "correction_state_dict": dynamic_corrections.state_dict(),
                    "epoch": epoch,
                    "val_dynamic_correction_mse": val_loss,
                },
                best_checkpoint,
            )
    dynamic_corrections.load_state_dict(
        torch.load(best_checkpoint, map_location="cpu", weights_only=False)[
            "correction_state_dict"
        ],
        strict=True,
    )
    metrics, diagnostics = evaluate(
        model, predictor, dynamic_corrections, instant_corrections, val_groups, sigma
    )
    recovery = (metrics["dyn"] - metrics["noisy"]) / (
        metrics["clean"] - metrics["noisy"]
    )
    summary = {
        "experiment": "kitti_step_dynamic_error_correction",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "static_host_checkpoint": str(static_checkpoint),
        "fixed_adapter_checkpoint": str(adapter_checkpoint),
        "fixed_predictor_checkpoint": str(predictor_checkpoint),
        "instant_correction_checkpoint": str(instant_checkpoint),
        "config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gaussian_noise_sigma": sigma,
            "alpha": ALPHA,
            "beta": BETA,
            "corrected_layers": [1, 4],
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in dynamic_corrections.parameters()
            ),
        },
        "dataset": {
            "train_sequence_count": len(train_groups),
            "val_sequence_count": len(val_groups),
        },
        "history": history,
        "best": {"epoch": best_epoch, "checkpoint_path": str(best_checkpoint)},
        "metrics": {
            "mIoU_clean": metrics["clean"],
            "mIoU_noisy": metrics["noisy"],
            "mIoU_pred": metrics["pred"],
            "mIoU_inst": metrics["inst"],
            "mIoU_dyn": metrics["dyn"],
            "delta_dyn_minus_inst": metrics["dyn"] - metrics["inst"],
            "recovery_dyn": recovery,
        },
        "diagnostics": diagnostics,
        "decision": {
            "go": metrics["dyn"] > metrics["inst"] and metrics["dyn"] > metrics["pred"]
        },
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
