import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from predify2021.datasets.kitti_step import (
    KITTI_STEP_IGNORE_LABEL,
    KITTI_STEP_NUM_CLASSES,
    KITTISTEPSegmentationDataset,
    pil_rgb_to_unit_tensor,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    CITYSCAPES_CLASSES,
    compute_iou,
    sequence_counts,
    update_confusion_matrix,
)
from predify2021.model_factory import get_model
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    CITYSCAPES_CHECKPOINT_NAME,
    CITYSCAPES_CHECKPOINT_URL,
    HostFeature,
)


CITYSCAPES_ZERO_TRAIN_MIOU = 0.6040115051765923


class KITTISTEPTrainTransform:
    def __init__(self, resize_size=(512, 1700), crop_size=(512, 1024), flip_prob=0.5):
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.flip_prob = flip_prob

    def __call__(self, image, mask):
        image = TF.resize(image, self.resize_size, interpolation=InterpolationMode.BILINEAR)
        mask_image = Image.fromarray(mask.numpy().astype("uint8"), mode="L")
        mask_image = TF.resize(
            mask_image,
            self.resize_size,
            interpolation=InterpolationMode.NEAREST,
        )
        crop_h, crop_w = self.crop_size
        image_w, image_h = image.size
        top = random.randint(0, image_h - crop_h)
        left = random.randint(0, image_w - crop_w)
        image = TF.crop(image, top, left, crop_h, crop_w)
        mask_image = TF.crop(mask_image, top, left, crop_h, crop_w)
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask_image = TF.hflip(mask_image)
        return pil_rgb_to_unit_tensor(image), torch.from_numpy(
            np.array(mask_image, dtype="int64")
        )


class KITTISTEPJointTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, joint_transform=None):
        self.dataset = dataset
        self.joint_transform = joint_transform
        self.samples = dataset.samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path = self.dataset.samples[index]["image_path"]
        mask_path = self.dataset.samples[index]["mask_path"]
        image = Image.open(image_path).convert("RGB")
        _, mask, metadata = self.dataset[index]
        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)
        return image, mask, metadata


def collate_train(samples):
    images, masks, metadata = zip(*samples)
    return torch.stack(images), torch.stack(masks), metadata


def collate_val(samples):
    if len(samples) != 1:
        raise ValueError("Validation uses batch size 1 for whole-frame inference.")
    image, mask, metadata = samples[0]
    return image.unsqueeze(0), mask, metadata


def model_segmentation_logits(model, images):
    c1, _, c3, c4 = model.backbone(model.preprocess_images(images))
    host_feature = HostFeature(
        tensor=c4,
        low_level=c1,
        output_size=tuple(images.shape[-2:]),
    )
    main_logits = model.decode_head(host_feature)
    aux_logits = model.auxiliary_head(c3, tuple(images.shape[-2:]))
    return main_logits, aux_logits


def train_one_epoch(model, dataloader, optimizer, scaler, device, aux_weight):
    model.train()
    total_loss = 0.0
    total_frames = 0
    criterion = nn.CrossEntropyLoss(ignore_index=KITTI_STEP_IGNORE_LABEL)
    for images, masks, _ in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, aux_logits = model_segmentation_logits(model, images)
            loss = criterion(logits, masks) + aux_weight * criterion(aux_logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(images.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_frames += batch_size
    return total_loss / max(total_frames, 1)


def evaluate(model, dataloader, device):
    model.eval()
    confusion = torch.zeros((KITTI_STEP_NUM_CLASSES, KITTI_STEP_NUM_CLASSES), dtype=torch.int64)
    frame_count = 0
    with torch.inference_mode():
        for images, masks, _ in dataloader:
            logits = model(images.to(device, non_blocking=True))
            prediction = logits.argmax(dim=1).squeeze(0).cpu().to(torch.int64)
            update_confusion_matrix(confusion, prediction, masks.to(torch.int64))
            frame_count += 1
    iou = compute_iou(confusion)
    return {
        "mIoU": float(torch.nanmean(iou).item()),
        "per_class_iou": {
            class_name: None if torch.isnan(iou[index]) else float(iou[index].item())
            for index, class_name in enumerate(CITYSCAPES_CLASSES)
        },
        "confusion_matrix": confusion.tolist(),
        "valid_frame_count": frame_count,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("KITTI-STEP static fine-tuning expects GPU 0.")
    seed = int(os.environ.get("PREDIFY_SEED", "0"))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_FINETUNE_OUTPUT_DIR",
            "/tmp/predify-storage/experiments/kitti_step_static_finetune",
        )
    )
    checkpoint_path = Path(
        os.environ.get(
            "PREDIFY_DEEPLABV3PLUS_CITYSCAPES_CHECKPOINT",
            str(Path(torch.hub.get_dir()) / "checkpoints" / CITYSCAPES_CHECKPOINT_NAME),
        )
    )
    epochs = int(os.environ.get("PREDIFY_KITTI_STEP_EPOCHS", "5"))
    batch_size = int(os.environ.get("PREDIFY_KITTI_STEP_BATCH_SIZE", "2"))
    lr = float(os.environ.get("PREDIFY_KITTI_STEP_LR", "0.00002"))
    weight_decay = float(os.environ.get("PREDIFY_KITTI_STEP_WEIGHT_DECAY", "0.01"))
    aux_weight = float(os.environ.get("PREDIFY_KITTI_STEP_AUX_WEIGHT", "0.4"))
    train_base = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_dataset = KITTISTEPJointTransformDataset(
        train_base,
        joint_transform=KITTISTEPTrainTransform(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(os.environ.get("PREDIFY_KITTI_STEP_NUM_WORKERS", "4")),
        pin_memory=True,
        collate_fn=collate_train,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(os.environ.get("PREDIFY_KITTI_STEP_NUM_WORKERS", "4")),
        pin_memory=True,
        collate_fn=collate_val,
    )
    model = get_model(
        "deeplabv3plus_resnet50",
        pretrained=True,
        segmentation_checkpoint_path=str(checkpoint_path),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    best_epoch = None
    best_checkpoint = output_dir / "best_kitti_step_static_deeplabv3plus.pt"
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, aux_weight)
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mIoU": val_metrics["mIoU"],
            "val_per_class_iou": val_metrics["per_class_iou"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "train_loss": train_loss,
                    "config": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "aux_weight": aux_weight,
                        "crop_size": (512, 1024),
                        "resize_size": (512, 1700),
                        "flip_prob": 0.5,
                        "seed": seed,
                    },
                },
                best_checkpoint,
            )
    best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    summary = {
        "experiment": "kitti_step_static_deeplabv3plus_resnet50_finetune",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "cityscapes_zero_train_mIoU": CITYSCAPES_ZERO_TRAIN_MIOU,
        "model": "deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024",
        "checkpoint_initialization": {
            "path": str(checkpoint_path),
            "source": CITYSCAPES_CHECKPOINT_URL,
            "sha256": sha256_file(checkpoint_path),
            "loaded_strict": True,
        },
        "dataset": {
            "root": str(root),
            "train_split": "train",
            "val_split": "val",
            "train_sequence_count": len(sequence_counts(train_base)),
            "train_sequence_frame_counts": sequence_counts(train_base),
            "train_frame_count": len(train_base),
            "val_sequence_count": len(sequence_counts(val_dataset)),
            "val_sequence_frame_counts": sequence_counts(val_dataset),
            "val_frame_count": len(val_dataset),
        },
        "training": {
            "static_single_frame": True,
            "temporal_state": False,
            "optimizer": "AdamW",
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "loss": "CrossEntropyLoss(ignore_index=255) + 0.4 auxiliary CrossEntropyLoss",
            "augmentation": {
                "resize_size": [512, 1700],
                "crop_size": [512, 1024],
                "horizontal_flip_prob": 0.5,
                "mask_interpolation": "nearest",
            },
        },
        "best": {
            "epoch": best_epoch,
            "checkpoint_path": str(best_checkpoint),
            "checkpoint_sha256": sha256_file(best_checkpoint),
            "mIoU": float(best_payload["val_metrics"]["mIoU"]),
            "per_class_iou": best_payload["val_metrics"]["per_class_iou"],
            "improvement_over_cityscapes_zero_train_mIoU": (
                float(best_payload["val_metrics"]["mIoU"]) - CITYSCAPES_ZERO_TRAIN_MIOU
            ),
        },
        "history": history,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary["best"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
