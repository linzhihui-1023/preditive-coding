import argparse
import json
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

from predify2021.datasets.vspw import (
    VSPW_IGNORE_LABEL,
    VSPW_NUM_CLASSES,
    VSPWSegmentationDataset,
    pil_rgb_to_unit_tensor,
    semantic_mask_from_vspw_png,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    CITYSCAPES_CHECKPOINT_NAME,
    HostFeature,
    build_deeplabv3plus_resnet50_host,
)


class VSPWTrainTransform:
    def __init__(self, crop_size=(480, 768), scale_range=(0.75, 1.5), flip_prob=0.5):
        self.crop_size = tuple(crop_size)
        self.scale_range = tuple(scale_range)
        self.flip_prob = flip_prob

    def __call__(self, image, mask):
        scale = random.uniform(*self.scale_range)
        short_side = max(1, round(480 * scale))
        image_w, image_h = image.size
        target_w = max(self.crop_size[1], round(image_w * short_side / image_h))
        image = TF.resize(image, (short_side, target_w), interpolation=InterpolationMode.BILINEAR)
        mask_image = Image.fromarray(mask.numpy().astype("uint8"), mode="L")
        mask_image = TF.resize(mask_image, (short_side, target_w), interpolation=InterpolationMode.NEAREST)
        crop_h, crop_w = self.crop_size
        pad_h = max(0, crop_h - image.height)
        pad_w = max(0, crop_w - image.width)
        if pad_h or pad_w:
            image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
            mask_image = TF.pad(mask_image, (0, 0, pad_w, pad_h), fill=VSPW_IGNORE_LABEL)
        top = random.randint(0, image.height - crop_h)
        left = random.randint(0, image.width - crop_w)
        image = TF.crop(image, top, left, crop_h, crop_w)
        mask_image = TF.crop(mask_image, top, left, crop_h, crop_w)
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask_image = TF.hflip(mask_image)
        return pil_rgb_to_unit_tensor(image), torch.from_numpy(np.asarray(mask_image, dtype=np.int64))


class VSPWJointDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        self.samples = dataset.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = semantic_mask_from_vspw_png(sample["mask_path"])
        image, mask = self.transform(image, mask)
        return image, mask


def collate_val(samples):
    if len(samples) != 1:
        raise ValueError("VSPW validation uses batch size 1 to retain frame order.")
    image, mask, metadata = samples[0]
    return image.unsqueeze(0), mask.unsqueeze(0), metadata


def segmentation_logits(model, images):
    c1, _, c3, c4 = model.backbone(model.preprocess_images(images))
    host_feature = HostFeature(tensor=c4, low_level=c1, output_size=tuple(images.shape[-2:]))
    return model.decode_head(host_feature), model.auxiliary_head(c3, tuple(images.shape[-2:]))


def update_confusion(confusion, logits, masks):
    prediction = logits.argmax(dim=1).to(torch.int64).cpu().numpy()
    target = masks.to(torch.int64).cpu().numpy()
    for pred, gt in zip(prediction, target):
        valid = (gt != VSPW_IGNORE_LABEL) & (gt >= 0) & (gt < VSPW_NUM_CLASSES)
        encoded = VSPW_NUM_CLASSES * gt[valid] + pred[valid]
        confusion += np.bincount(encoded, minlength=VSPW_NUM_CLASSES**2).reshape(
            VSPW_NUM_CLASSES, VSPW_NUM_CLASSES
        )


def miou_from_confusion(confusion):
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection
    iou = np.divide(intersection, union, out=np.full(VSPW_NUM_CLASSES, np.nan), where=union != 0)
    return float(np.nanmean(iou)), iou


def evaluate(model, dataloader, device):
    model.eval()
    confusion = np.zeros((VSPW_NUM_CLASSES, VSPW_NUM_CLASSES), dtype=np.int64)
    criterion = nn.CrossEntropyLoss(ignore_index=VSPW_IGNORE_LABEL)
    loss_sum = 0.0
    frame_count = 0
    with torch.inference_mode():
        for batch_index, (images, masks, _) in enumerate(dataloader, 1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, _ = segmentation_logits(model, images)
                loss = criterion(logits, masks)
            update_confusion(confusion, logits, masks)
            loss_sum += float(loss.item())
            frame_count += int(images.shape[0])
            if batch_index % 2000 == 0:
                print(json.dumps({"val_batch": batch_index}, sort_keys=True), flush=True)
    miou, per_class_iou = miou_from_confusion(confusion)
    return {
        "mIoU": miou,
        "val_loss": loss_sum / max(frame_count, 1),
        "per_class_iou": per_class_iou.tolist(),
        "confusion_matrix": confusion.tolist(),
        "frame_count": frame_count,
    }


def train_one_epoch(model, dataloader, optimizer, scaler, device, aux_weight):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=VSPW_IGNORE_LABEL)
    loss_sum = 0.0
    frame_count = 0
    for batch_index, (images, masks) in enumerate(dataloader, 1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            main_logits, aux_logits = segmentation_logits(model, images)
            loss = criterion(main_logits, masks) + aux_weight * criterion(aux_logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch = int(images.shape[0])
        loss_sum += float(loss.detach().item()) * batch
        frame_count += batch
        if batch_index % 1000 == 0:
            print(
                json.dumps(
                    {"train_batch": batch_index, "train_loss_so_far": loss_sum / max(frame_count, 1)},
                    sort_keys=True,
                ),
                flush=True,
            )
    return loss_sum / max(frame_count, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a static 124-class VSPW DeepLabV3+ R50 host.")
    parser.add_argument("--data-root", default="/home/lin/datasets/VSPW_480p")
    parser.add_argument("--output-dir", default="/home/lin/predify/experiments/vspw_static_deeplabv3plus_r50")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--aux-weight", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("VSPW Host training requires a CUDA device.")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_base = VSPWSegmentationDataset(data_root, "train")
    val_dataset = VSPWSegmentationDataset(data_root, "val")
    train_dataset = VSPWJointDataset(train_base, VSPWTrainTransform())
    workers = max(0, args.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=collate_val,
    )
    model = build_deeplabv3plus_resnet50_host(
        num_classes=VSPW_NUM_CLASSES,
        checkpoint_path=args.checkpoint,
        load_cityscapes_checkpoint=True,
    ).to(device)
    trainable_modules = (model.backbone, model.decode_head, model.auxiliary_head)
    optimizer = torch.optim.AdamW(
        [parameter for module in trainable_modules for parameter in module.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda")
    best_miou = -1.0
    best_epoch = None
    stale_epochs = 0
    history = []
    best_path = output_dir / "best_vspw_static_deeplabv3plus_r50.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args.aux_weight)
        val = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["val_loss"],
            "val_mIoU": val["mIoU"],
            "val_frame_count": val["frame_count"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        improved = val["mIoU"] > best_miou or (
            val["mIoU"] == best_miou and (best_epoch is None or val["val_loss"] < history[best_epoch - 1]["val_loss"])
        )
        if improved:
            best_miou = val["mIoU"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val,
                    "config": vars(args),
                },
                best_path,
            )
        elif epoch >= args.warmup_epochs:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(json.dumps({"early_stopping": True, "epoch": epoch}), flush=True)
                break
    summary = {
        "experiment": "vspw_static_deeplabv3plus_resnet50_iss_host",
        "data_root": str(data_root),
        "model": "DeepLabV3+-ResNet50",
        "num_classes": VSPW_NUM_CLASSES,
        "ignore_index": VSPW_IGNORE_LABEL,
        "checkpoint_initialization": model.checkpoint_load_report,
        "trainable_params": sum(p.numel() for module in trainable_modules for p in module.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "training": {
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "aux_weight": args.aux_weight,
            "crop_size": [480, 768],
            "scale_range": [0.75, 1.5],
            "horizontal_flip_prob": 0.5,
            "mask_interpolation": "nearest",
            "amp": True,
        },
        "dataset": {
            "train_videos": len(train_base.video_ids),
            "train_frames": len(train_base),
            "val_videos": len(val_dataset.video_ids),
            "val_frames": len(val_dataset),
        },
        "best": {"epoch": best_epoch, "checkpoint_path": str(best_path), "val_mIoU": best_miou},
        "history": history,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary["best"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
