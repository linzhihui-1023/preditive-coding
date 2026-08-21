import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from predify2021.datasets.kitti_step_triplets import KITTISTEPTripletDataset
from predify2021.mce_scores.train_kitti_step_state_predictor import (
    compute_state_prediction,
    load_static_kitti_checkpoint,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    MultiLayerPredictor,
    build_deeplabv3plus_resnet50_host,
)


STATIC_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/checkpoints/kitti_step_static_deeplabv3plus_epoch1/"
    "best_kitti_step_static_deeplabv3plus.pt"
)
ADAPTER_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_state_predictor_d655a0d/"
    "best_state_predictor.pt"
)


def configure_predictor_only(model, predictor):
    model.requires_grad_(False)
    predictor.requires_grad_(True)
    return list(predictor.parameters())


def prediction_losses(model, predictor, images):
    previous, current, predicted = compute_state_prediction(model, predictor, images)
    predictor_mse = [
        F.mse_loss(predicted_value, target_value)
        for predicted_value, target_value in zip(predicted.as_tuple(), current.as_tuple())
    ]
    copy_mse = [
        F.mse_loss(previous_value, target_value)
        for previous_value, target_value in zip(previous.as_tuple(), current.as_tuple())
    ]
    return sum(predictor_mse), predictor_mse, copy_mse


def run_epoch(model, predictor, loader, optimizer, training):
    model.eval()
    predictor.train(training)
    totals = {
        "prediction_loss": 0.0,
        "predictor_mse": [0.0] * 4,
        "copy_mse": [0.0] * 4,
        "sample_count": 0,
    }
    for images, _ in loader:
        images = images.cuda(non_blocking=True)
        prediction_loss, predictor_mse, copy_mse = prediction_losses(model, predictor, images)
        if training:
            optimizer.zero_grad(set_to_none=True)
            prediction_loss.backward()
            optimizer.step()
        batch_size = images.shape[0]
        totals["prediction_loss"] += prediction_loss.detach().item() * batch_size
        for index in range(4):
            totals["predictor_mse"][index] += predictor_mse[index].detach().item() * batch_size
            totals["copy_mse"][index] += copy_mse[index].detach().item() * batch_size
        totals["sample_count"] += batch_size
    predictor_mse = [
        value / totals["sample_count"] for value in totals["predictor_mse"]
    ]
    copy_mse = [value / totals["sample_count"] for value in totals["copy_mse"]]
    layer_improvement = [
        (copy_value - predictor_value) / copy_value
        for predictor_value, copy_value in zip(predictor_mse, copy_mse)
    ]
    mean_predictor_mse = sum(predictor_mse) / 4
    mean_copy_mse = sum(copy_mse) / 4
    return {
        "prediction_loss": totals["prediction_loss"] / totals["sample_count"],
        "predictor_mse": predictor_mse,
        "copy_mse": copy_mse,
        "layer_improvement": layer_improvement,
        "mean_predictor_mse": mean_predictor_mse,
        "mean_copy_mse": mean_copy_mse,
        "mean_improvement": (mean_copy_mse - mean_predictor_mse) / mean_copy_mse,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Fixed-adapter predictor training expects GPU 0.")
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
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_FIXED_ADAPTER_PREDICTOR_OUTPUT_DIR",
            "/home/lin/predify/experiments/kitti_step_fixed_adapter_predictor",
        )
    )
    batch_size = int(
        os.environ.get("PREDIFY_KITTI_STEP_FIXED_ADAPTER_PREDICTOR_BATCH_SIZE", "1")
    )
    epochs = int(
        os.environ.get("PREDIFY_KITTI_STEP_FIXED_ADAPTER_PREDICTOR_EPOCHS", "3")
    )
    learning_rate = float(
        os.environ.get("PREDIFY_KITTI_STEP_FIXED_ADAPTER_PREDICTOR_LR", "0.0001")
    )
    weight_decay = float(
        os.environ.get("PREDIFY_KITTI_STEP_FIXED_ADAPTER_PREDICTOR_WEIGHT_DECAY", "0.01")
    )
    num_workers = int(os.environ.get("PREDIFY_KITTI_STEP_NUM_WORKERS", "4"))
    model = build_deeplabv3plus_resnet50_host(
        checkpoint_path=os.environ.get("PREDIFY_DEEPLABV3PLUS_CITYSCAPES_CHECKPOINT"),
    ).cuda()
    load_static_kitti_checkpoint(model, static_checkpoint)
    adapter_payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.load_state_dict(adapter_payload["adapter_state_dict"], strict=True)
    predictor = MultiLayerPredictor().cuda()
    optimizer = torch.optim.AdamW(
        configure_predictor_only(model, predictor),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_dataset = KITTISTEPTripletDataset.from_kitti_step_root(root, "train")
    val_dataset = KITTISTEPTripletDataset.from_kitti_step_root(root, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "best_predictor.pt"
    best_epoch = None
    best_val_metrics = None
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, predictor, train_loader, optimizer, True)
        with torch.no_grad():
            val_metrics = run_epoch(model, predictor, val_loader, optimizer, False)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if best_val_metrics is None or (
            val_metrics["mean_predictor_mse"]
            < best_val_metrics["mean_predictor_mse"]
        ):
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save(
                {
                    "predictor_state_dict": predictor.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_checkpoint,
            )
    layer_wins = sum(
        predictor_mse < copy_mse
        for predictor_mse, copy_mse in zip(
            best_val_metrics["predictor_mse"],
            best_val_metrics["copy_mse"],
        )
    )
    summary = {
        "experiment": "kitti_step_fixed_adapter_predictor",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "static_host_checkpoint": str(static_checkpoint),
        "fixed_adapter_checkpoint": str(adapter_checkpoint),
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in predictor.parameters()
            ),
        },
        "dataset": {
            "train_triplet_count": len(train_dataset),
            "val_triplet_count": len(val_dataset),
        },
        "history": history,
        "best": {
            "epoch": best_epoch,
            **best_val_metrics,
            "checkpoint_path": str(best_checkpoint),
        },
        "decision": {
            "go": (
                best_val_metrics["mean_predictor_mse"]
                < best_val_metrics["mean_copy_mse"]
                and layer_wins >= 3
            ),
            "layer_wins": layer_wins,
        },
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
