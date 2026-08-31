"""Minimal Cityscapes semantic-segmentation loader."""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_IGNORE_LABEL = 255
CITYSCAPES_IMAGE_SIZE = (2048, 1024)


def label_ids_to_train_ids(label_ids):
    """Map official Cityscapes labelIds through cityscapesScripts labels."""
    from cityscapesscripts.helpers.labels import labels

    label_ids = np.asarray(label_ids, dtype=np.uint8)
    mapping = np.full(256, CITYSCAPES_IGNORE_LABEL, dtype=np.uint8)
    for label in labels:
        if 0 <= label.id < 256 and 0 <= label.trainId < CITYSCAPES_NUM_CLASSES:
            mapping[label.id] = label.trainId
    return mapping[label_ids]


def pil_rgb_to_unit_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class CityscapesSegmentationDataset(Dataset):
    """Cityscapes leftImg8bit plus official gtFine labelIds pairs."""

    def __init__(
        self,
        root,
        split="val",
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        image_root = self.root / "leftImg8bit" / split
        label_root = self.root / "gtFine" / split
        self.samples = tuple(
            self._make_sample(path, label_root)
            for path in sorted(image_root.glob("*/*_leftImg8bit.png"))
        )
        missing = [sample for sample in self.samples if not sample["label_path"].is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing gtFine labelIds for {len(missing)} {split} images under {label_root}"
            )

    @staticmethod
    def _make_sample(image_path, label_root):
        stem = image_path.name.removesuffix("_leftImg8bit.png")
        return {
            "image_path": image_path,
            "label_path": label_root / image_path.parent.name / f"{stem}_gtFine_labelIds.png",
            "city": image_path.parent.name,
            "frame_id": stem,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        label_ids = np.asarray(Image.open(sample["label_path"]), dtype=np.uint8)
        mask = torch.from_numpy(label_ids_to_train_ids(label_ids).astype(np.int64))
        image = self.image_transform(image) if self.image_transform else pil_rgb_to_unit_tensor(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        metadata = {
            "image_path": str(sample["image_path"]),
            "label_path": str(sample["label_path"]),
            "city": sample["city"],
            "frame_id": sample["frame_id"],
            "split": self.split,
        }
        return image, mask, metadata


def cityscapes_split_counts(root):
    root = Path(root)
    return {
        split: len(tuple((root / "leftImg8bit" / split).glob("*/*_leftImg8bit.png")))
        for split in ("train", "val", "test")
    }
