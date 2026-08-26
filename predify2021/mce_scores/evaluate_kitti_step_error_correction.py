import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.datasets.kitti_step_triplets import KITTISTEPTripletDataset
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.train_kitti_step_state_predictor import (
    encode_triplet,
    load_static_kitti_checkpoint,
    select_time,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    HostFeature,
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


def collate_triplet(samples):
    images = torch.stack([sample[0] for sample in samples])
    metadata = samples[0][1] if len(samples) == 1 else [sample[1] for sample in samples]
    return images, metadata


def add_current_frame_noise(images, sigma):
    noisy = images.clone()
    noisy[:, 2] = torch.clamp(
        noisy[:, 2] + sigma * torch.randn_like(noisy[:, 2]), 0.0, 1.0
    )
    return noisy


def build_prediction_and_error(model, predictor, clean_images, sigma):
    noisy_images = add_current_frame_noise(clean_images, sigma)
    clean_states = encode_triplet(model, clean_images)
    noisy_states = encode_triplet(model, noisy_images)
    previous_previous = select_time(clean_states, 0)
    previous = select_time(clean_states, 1)
    clean_current = select_time(clean_states, 2)
    noisy_current = select_time(noisy_states, 2)
    previous_delta = UnifiedFeatures(
        *(new - old for new, old in zip(previous.as_tuple(), previous_previous.as_tuple()))
    )
    predicted_delta = predictor(previous, previous_delta)
    predicted = UnifiedFeatures(
        *(state + delta for state, delta in zip(previous.as_tuple(), predicted_delta.as_tuple()))
    )
    error = UnifiedFeatures(
        *(noisy - prediction for noisy, prediction in zip(noisy_current.as_tuple(), predicted.as_tuple()))
    )
    return noisy_images, clean_current, noisy_current, predicted, error


def correction_states(predicted, gain_error, correction_error, corrections):
    gains = (
        corrections[0](gain_error.z1),
        corrections[1](gain_error.z4),
    )
    return UnifiedFeatures(
        predicted.z1 + gains[0] * correction_error.z1,
        predicted.z2,
        predicted.z3,
        predicted.z4 + gains[1] * correction_error.z4,
    )


def correction_loss(clean_current, corrected):
    return F.mse_loss(corrected.z1, clean_current.z1) + F.mse_loss(
        corrected.z4, clean_current.z4
    )


def run_correction_epoch(model, predictor, corrections, loader, optimizer, sigma, training):
    model.eval()
    predictor.eval()
    corrections.train(training)
    total_loss = 0.0
    sample_count = 0
    for images, _ in loader:
        images = images.cuda(non_blocking=True)
        _, clean_current, _, predicted, error = build_prediction_and_error(
            model, predictor, images, sigma
        )
        corrected = correction_states(predicted, error, error, corrections)
        loss = correction_loss(clean_current, corrected)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += loss.detach().item() * images.shape[0]
        sample_count += images.shape[0]
    return total_loss / sample_count


def corrected_host_feature(model, raw_features, noisy_state, target_state, output_size):
    delta = UnifiedFeatures(
        target_state.z1 - noisy_state.z1,
        torch.zeros_like(noisy_state.z2),
        torch.zeros_like(noisy_state.z3),
        target_state.z4 - noisy_state.z4,
    )
    delta_features = model.decode_conditioned_adapter_deltas(raw_features, delta)
    return HostFeature(
        tensor=raw_features.c4 + delta_features.c4,
        low_level=raw_features.c1 + delta_features.c1,
        output_size=output_size,
    )


def evaluate_paths(model, predictor, corrections, loader, mask_paths, sigma):
    confusion = {
        "clean": torch.zeros((19, 19), dtype=torch.int64),
        "noisy": torch.zeros((19, 19), dtype=torch.int64),
        "pred": torch.zeros((19, 19), dtype=torch.int64),
        "corr": torch.zeros((19, 19), dtype=torch.int64),
    }
    model.eval()
    predictor.eval()
    corrections.eval()
    with torch.no_grad():
        for images, metadata in loader:
            clean_images = images.cuda(non_blocking=True)
            noisy_images, clean_state, noisy_state, predicted, error = build_prediction_and_error(
                model, predictor, clean_images, sigma
            )
            clean_current = clean_images[:, 2]
            noisy_current = noisy_images[:, 2]
            noisy_features = model.extract_backbone_features(noisy_current)
            corrected = correction_states(predicted, error, error, corrections)
            output_size = tuple(clean_current.shape[-2:])
            pred_host = corrected_host_feature(
                model, noisy_features, noisy_state, predicted, output_size
            )
            corr_host = corrected_host_feature(
                model, noisy_features, noisy_state, corrected, output_size
            )
            logits = {
                "clean": model(clean_current),
                "noisy": model(noisy_current),
                "pred": model.decode_from_host_feature(pred_host),
                "corr": model.decode_from_host_feature(corr_host),
            }
            sequence_id = metadata["sequence_id"]
            frame_id = metadata["frame_ids"][2]
            mask = semantic_mask_from_panoptic_png(mask_paths[(sequence_id, frame_id)])
            for name, value in logits.items():
                prediction = value.argmax(dim=1).squeeze(0).cpu().to(torch.int64)
                update_confusion_matrix(confusion[name], prediction, mask)
    return {
        name: float(torch.nanmean(compute_iou(value)).item())
        for name, value in confusion.items()
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Error correction training expects GPU 0.")
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
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_ERROR_CORRECTION_OUTPUT_DIR",
            "/home/lin/predify/experiments/kitti_step_error_correction",
        )
    )
    epochs = int(os.environ.get("PREDIFY_KITTI_STEP_ERROR_CORRECTION_EPOCHS", "3"))
    batch_size = int(os.environ.get("PREDIFY_KITTI_STEP_ERROR_CORRECTION_BATCH_SIZE", "1"))
    learning_rate = float(
        os.environ.get("PREDIFY_KITTI_STEP_ERROR_CORRECTION_LR", "0.0001")
    )
    weight_decay = float(
        os.environ.get("PREDIFY_KITTI_STEP_ERROR_CORRECTION_WEIGHT_DECAY", "0.01")
    )
    sigma = 0.10
    num_workers = int(os.environ.get("PREDIFY_KITTI_STEP_NUM_WORKERS", "4"))
    model = build_deeplabv3plus_resnet50_host(
        checkpoint_path=os.environ.get("PREDIFY_DEEPLABV3PLUS_CITYSCAPES_CHECKPOINT"),
    ).cuda()
    load_static_kitti_checkpoint(model, static_checkpoint)
    adapter_payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.load_state_dict(adapter_payload["adapter_state_dict"], strict=True)
    predictor = MultiLayerPredictor().cuda()
    predictor_payload = torch.load(predictor_checkpoint, map_location="cpu", weights_only=False)
    predictor.load_state_dict(predictor_payload["predictor_state_dict"], strict=True)
    corrections = torch.nn.ModuleList([ErrorGainCorrection(), ErrorGainCorrection()]).cuda()
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        corrections.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_dataset = KITTISTEPTripletDataset.from_kitti_step_root(root, "train")
    val_dataset = KITTISTEPTripletDataset.from_kitti_step_root(root, "val")
    val_base = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    mask_paths = {
        (sample["sequence_id"], sample["frame_id"]): sample["mask_path"]
        for sample in val_base.samples
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_triplet,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_triplet,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "best_correction.pt"
    best_val_loss = float("inf")
    best_epoch = None
    best_val_correction_loss = None
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = run_correction_epoch(
            model, predictor, corrections, train_loader, optimizer, sigma, True
        )
        with torch.no_grad():
            val_loss = run_correction_epoch(
                model, predictor, corrections, val_loader, optimizer, sigma, False
            )
        history.append({"epoch": epoch, "train_correction_mse": train_loss, "val_correction_mse": val_loss})
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_correction_loss = val_loss
            torch.save(
                {
                    "correction_state_dict": corrections.state_dict(),
                    "epoch": epoch,
                    "val_correction_mse": val_loss,
                },
                best_checkpoint,
            )
    corrections.load_state_dict(
        torch.load(best_checkpoint, map_location="cpu", weights_only=False)[
            "correction_state_dict"
        ],
        strict=True,
    )
    metrics = evaluate_paths(model, predictor, corrections, val_loader, mask_paths, sigma)
    recovery = (metrics["clean"] - metrics["noisy"])
    recovery = (metrics["corr"] - metrics["noisy"]) / recovery
    summary = {
        "experiment": "kitti_step_error_correction",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "static_host_checkpoint": str(static_checkpoint),
        "fixed_adapter_checkpoint": str(adapter_checkpoint),
        "fixed_predictor_checkpoint": str(predictor_checkpoint),
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gaussian_noise_sigma": sigma,
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in corrections.parameters()
            ),
            "corrected_layers": [1, 4],
        },
        "dataset": {
            "train_triplet_count": len(train_dataset),
            "val_triplet_count": len(val_dataset),
        },
        "history": history,
        "best": {
            "epoch": best_epoch,
            "val_correction_mse": best_val_correction_loss,
            "checkpoint_path": str(best_checkpoint),
        },
        "metrics": {
            "mIoU_clean": metrics["clean"],
            "mIoU_noisy": metrics["noisy"],
            "mIoU_pred": metrics["pred"],
            "mIoU_corr": metrics["corr"],
            "recovery": recovery,
        },
        "decision": {
            "go": metrics["corr"] > metrics["noisy"] and metrics["corr"] > metrics["pred"],
        },
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
