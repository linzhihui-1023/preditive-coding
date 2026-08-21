import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from predify2021.datasets.kitti_step_triplets import KITTISTEPTripletDataset
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    CITYSCAPES_CHECKPOINT_NAME,
    MultiLayerPredictor,
    UnifiedFeatures,
    build_deeplabv3plus_resnet50_host,
)


STATIC_CHECKPOINT_DEFAULT = (
    "/tmp/predify-storage/experiments/kitti_step_static_finetune_16d0229/"
    "best_kitti_step_static_deeplabv3plus.pt"
)


def load_static_kitti_checkpoint(model, checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict", payload)
    for prefix, module in (
        ("backbone", model.backbone),
        ("decode_head", model.decode_head),
        ("auxiliary_head", model.auxiliary_head),
    ):
        module_state = {
            key[len(prefix) + 1 :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix + ".")
        }
        module.load_state_dict(module_state, strict=True)
    return payload


def configure_trainable_modules(model, predictor):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.multi_layer_adapter.requires_grad_(True)
    predictor.requires_grad_(True)
    return [
        parameter
        for parameter in list(model.multi_layer_adapter.parameters()) + list(predictor.parameters())
        if parameter.requires_grad
    ]


def encode_triplet(model, images):
    batch_size, time_steps, channels, height, width = images.shape
    flat_images = images.reshape(batch_size * time_steps, channels, height, width)
    with torch.no_grad():
        features = model.extract_backbone_features(flat_images)
    states = model.encode_backbone_features(features)
    return UnifiedFeatures(
        *(value.reshape(batch_size, time_steps, *value.shape[1:]) for value in states.as_tuple())
    )


def select_time(states, time_index):
    return UnifiedFeatures(*(value[:, time_index] for value in states.as_tuple()))


def compute_state_prediction(model, predictor, images):
    states = encode_triplet(model, images)
    previous_previous = select_time(states, 0)
    previous = select_time(states, 1)
    current = select_time(states, 2)
    previous_delta = UnifiedFeatures(
        *(new - old for new, old in zip(previous.as_tuple(), previous_previous.as_tuple()))
    )
    predicted_delta = predictor(previous, previous_delta)
    predicted = UnifiedFeatures(
        *(state + delta for state, delta in zip(previous.as_tuple(), predicted_delta.as_tuple()))
    )
    return previous, current, predicted


def state_losses(model, predictor, images, lambda_recon):
    previous, current, predicted = compute_state_prediction(model, predictor, images)
    prediction_losses = [
        F.mse_loss(predicted_value, target_value)
        for predicted_value, target_value in zip(predicted.as_tuple(), current.as_tuple())
    ]
    copy_losses = [
        F.mse_loss(previous_value, target_value)
        for previous_value, target_value in zip(previous.as_tuple(), current.as_tuple())
    ]
    reconstructed = model.decode_adapter_deltas(current)
    input_features = model.extract_backbone_features(images[:, 2].reshape(images.shape[0], *images.shape[2:]))
    reconstruction_losses = [
        F.mse_loss(reconstructed_value, target_value)
        for reconstructed_value, target_value in zip(
            reconstructed.as_tuple(), input_features.as_tuple()
        )
    ]
    prediction_loss = sum(prediction_losses)
    reconstruction_loss = sum(reconstruction_losses)
    return {
        "loss": prediction_loss + lambda_recon * reconstruction_loss,
        "prediction_loss": prediction_loss,
        "reconstruction_loss": reconstruction_loss,
        "prediction_mse": prediction_losses,
        "copy_mse": copy_losses,
    }


def run_epoch(model, predictor, loader, optimizer, lambda_recon, training):
    model.eval()
    predictor.train(training)
    if training:
        model.multi_layer_adapter.train()
    totals = {"prediction_mse": [0.0] * 4, "copy_mse": [0.0] * 4, "count": 0}
    for images, _ in loader:
        images = images.cuda(non_blocking=True)
        losses = state_losses(model, predictor, images, lambda_recon)
        if training:
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
        for index in range(4):
            totals["prediction_mse"][index] += losses["prediction_mse"][index].detach().item()
            totals["copy_mse"][index] += losses["copy_mse"][index].detach().item()
        totals["count"] += 1
    prediction_mse = [value / totals["count"] for value in totals["prediction_mse"]]
    copy_mse = [value / totals["count"] for value in totals["copy_mse"]]
    return {
        "prediction_mse": prediction_mse,
        "copy_mse": copy_mse,
        "mean_prediction_mse": sum(prediction_mse) / 4,
        "mean_copy_mse": sum(copy_mse) / 4,
        "relative_improvement": (sum(copy_mse) - sum(prediction_mse)) / sum(copy_mse),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("State predictor training expects GPU 0.")
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    static_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_STATIC_CHECKPOINT", STATIC_CHECKPOINT_DEFAULT)
    )
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_STATE_PREDICTOR_OUTPUT_DIR",
            "/tmp/predify-storage/experiments/kitti_step_state_predictor",
        )
    )
    batch_size = int(os.environ.get("PREDIFY_KITTI_STEP_STATE_PREDICTOR_BATCH_SIZE", "1"))
    epochs = int(os.environ.get("PREDIFY_KITTI_STEP_STATE_PREDICTOR_EPOCHS", "1"))
    learning_rate = float(os.environ.get("PREDIFY_KITTI_STEP_STATE_PREDICTOR_LR", "0.0001"))
    lambda_recon = float(
        os.environ.get("PREDIFY_KITTI_STEP_STATE_PREDICTOR_LAMBDA_RECON", "0.1")
    )
    host = build_deeplabv3plus_resnet50_host(
        checkpoint_path=os.environ.get("PREDIFY_DEEPLABV3PLUS_CITYSCAPES_CHECKPOINT"),
    ).cuda()
    load_static_kitti_checkpoint(host, static_checkpoint)
    predictor = MultiLayerPredictor().cuda()
    trainable_parameters = configure_trainable_modules(host, predictor)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
    train_dataset = KITTISTEPTripletDataset.from_kitti_step_root(root, "train")
    val_dataset = KITTISTEPTripletDataset.from_kitti_step_root(root, "val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(host, predictor, train_loader, optimizer, lambda_recon, True)
        with torch.no_grad():
            val_metrics = run_epoch(host, predictor, val_loader, optimizer, lambda_recon, False)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    torch.save(
        {
            "adapter_state_dict": host.multi_layer_adapter.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "config": {
                "static_checkpoint": str(static_checkpoint),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "lambda_recon": lambda_recon,
                "state_channels": 128,
                "official_cityscapes_checkpoint": CITYSCAPES_CHECKPOINT_NAME,
            },
            "history": history,
        },
        output_dir / "state_predictor.pt",
    )


if __name__ == "__main__":
    main()
