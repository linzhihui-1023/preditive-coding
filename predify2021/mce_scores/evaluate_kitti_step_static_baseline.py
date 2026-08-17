import json
import os
from pathlib import Path
import hashlib

import torch
from torch.utils.data import DataLoader

from predify2021.datasets.kitti_step import (
    KITTI_STEP_IGNORE_LABEL,
    KITTI_STEP_NUM_CLASSES,
    KITTISTEPSegmentationDataset,
)
from predify2021.model_factory import get_model
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    CITYSCAPES_CHECKPOINT_NAME,
    CITYSCAPES_CHECKPOINT_URL,
)


CITYSCAPES_CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)


def collate_batch(samples):
    if len(samples) != 1:
        raise ValueError("Static KITTI-STEP baseline expects batch size 1.")
    image, mask, metadata = samples[0]
    return image.unsqueeze(0), mask, metadata


def update_confusion_matrix(confusion, prediction, target):
    valid = ((target >= 0) & (target < KITTI_STEP_NUM_CLASSES)) | (
        target == KITTI_STEP_IGNORE_LABEL
    )
    if not bool(valid.all()):
        raise RuntimeError("KITTI-STEP mask contains labels outside 0..18 and 255.")
    keep = target != KITTI_STEP_IGNORE_LABEL
    encoded = KITTI_STEP_NUM_CLASSES * target[keep] + prediction[keep]
    confusion += torch.bincount(
        encoded,
        minlength=KITTI_STEP_NUM_CLASSES * KITTI_STEP_NUM_CLASSES,
    ).reshape(KITTI_STEP_NUM_CLASSES, KITTI_STEP_NUM_CLASSES).cpu()


def compute_iou(confusion):
    true_positive = torch.diag(confusion).double()
    ground_truth = confusion.sum(dim=1).double()
    predicted = confusion.sum(dim=0).double()
    union = ground_truth + predicted - true_positive
    iou = torch.full((KITTI_STEP_NUM_CLASSES,), float("nan"), dtype=torch.float64)
    present = union > 0
    iou[present] = true_positive[present] / union[present]
    return iou


def sequence_counts(dataset):
    counts = {}
    for sample in dataset.samples:
        sequence_id = sample["sequence_id"]
        counts[sequence_id] = counts.get(sequence_id, 0) + 1
    return dict(sorted(counts.items()))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("KITTI-STEP static baseline expects GPU 0.")
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_BASELINE_OUTPUT_DIR",
            "results/kitti_step_static_baseline_deeplabv3plus_cityscapes",
        )
    )
    checkpoint_path = Path(
        os.environ.get(
            "PREDIFY_DEEPLABV3PLUS_CITYSCAPES_CHECKPOINT",
            str(Path(torch.hub.get_dir()) / "checkpoints" / CITYSCAPES_CHECKPOINT_NAME),
        )
    )
    split = "val"
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(os.environ.get("PREDIFY_KITTI_STEP_NUM_WORKERS", "2")),
        collate_fn=collate_batch,
        pin_memory=True,
    )
    model = get_model(
        "deeplabv3plus_resnet50",
        pretrained=True,
        segmentation_checkpoint_path=str(checkpoint_path),
    ).to(device)
    model.eval()
    parameter_snapshot = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    confusion = torch.zeros(
        (KITTI_STEP_NUM_CLASSES, KITTI_STEP_NUM_CLASSES), dtype=torch.int64
    )
    frame_count = 0
    with torch.inference_mode():
        for image, mask, metadata in dataloader:
            logits = model(image.to(device, non_blocking=True))
            prediction = logits.argmax(dim=1).squeeze(0).cpu().to(torch.int64)
            update_confusion_matrix(confusion, prediction, mask.to(torch.int64))
            frame_count += 1
            if frame_count % 250 == 0:
                print(f"evaluated {frame_count}/{len(dataset)} frames", flush=True)

    parameters_unchanged = all(
        torch.equal(parameter.detach().cpu(), parameter_snapshot[name])
        for name, parameter in model.named_parameters()
    )
    iou = compute_iou(confusion)
    miou = float(torch.nanmean(iou).item())
    per_class_iou = {
        class_name: None if torch.isnan(iou[index]) else float(iou[index].item())
        for index, class_name in enumerate(CITYSCAPES_CLASSES)
    }
    summary = {
        "experiment": "kitti_step_static_deeplabv3plus_resnet50_cityscapes_baseline",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "model": "deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024",
        "checkpoint": {
            "path": str(checkpoint_path),
            "source": CITYSCAPES_CHECKPOINT_URL,
            "sha256": sha256_file(checkpoint_path),
            "loaded_strict": model.checkpoint_load_report["strict"],
            "backbone_key_count": model.checkpoint_load_report["backbone_key_count"],
            "decode_head_key_count": model.checkpoint_load_report["decode_head_key_count"],
        },
        "dataset": {
            "root": str(root),
            "split": split,
            "sequence_count": len(sequence_counts(dataset)),
            "sequence_frame_counts": sequence_counts(dataset),
            "frame_count": len(dataset),
        },
        "evaluation": {
            "mode": "static_single_frame",
            "training": False,
            "parameter_update": False,
            "parameters_unchanged_after_eval": parameters_unchanged,
            "valid_frame_count": frame_count,
            "num_classes": KITTI_STEP_NUM_CLASSES,
            "ignore_label": KITTI_STEP_IGNORE_LABEL,
            "mIoU": miou,
            "per_class_iou": per_class_iou,
            "confusion_matrix": confusion.tolist(),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary["evaluation"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
