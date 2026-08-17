from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


KITTI_STEP_NUM_CLASSES = 19
KITTI_STEP_IGNORE_LABEL = 255


class KITTISTEPSegmentationDataset(Dataset):
    """KITTI-STEP single-frame semantic segmentation dataset entry.

    The loader is intentionally independent from the KITTI Raw pair loaders.
    It is driven by official KITTI-STEP panoptic annotation splits and returns
    RGB images in `[0, 1]`, semantic masks from the annotation PNG red channel,
    and sequence/frame metadata in true per-sequence frame order.
    """

    def __init__(
        self,
        image_root,
        mask_root,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.mask_paths = tuple(sorted(self.mask_root.glob("*/*.png")))
        self.samples = tuple(
            {
                "sequence_id": mask.parent.name,
                "frame_id": mask.stem,
                "image_path": self.image_root / mask.parent.name / mask.name,
                "mask_path": mask,
            }
            for mask in self.mask_paths
        )
        self.image_paths = tuple(sample["image_path"] for sample in self.samples)

    @classmethod
    def from_kitti_step_root(
        cls,
        root,
        split: str,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):
        root = Path(root)
        return cls(
            image_root=root / "training" / "image_02",
            mask_root=root / "kitti-step" / "panoptic_maps" / split,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
        image = Image.open(image_path).convert("RGB")
        mask = semantic_mask_from_panoptic_png(mask_path)
        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            image = pil_rgb_to_unit_tensor(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        metadata = {
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "sequence_id": sample["sequence_id"],
            "frame_id": sample["frame_id"],
        }
        return image, mask, metadata


def semantic_mask_from_panoptic_png(path):
    panoptic = Image.open(path).convert("RGB")
    red_channel = np.array(panoptic, dtype=np.uint8)[..., 0]
    return torch.from_numpy(red_channel.astype(np.int64))


def pil_rgb_to_unit_tensor(image):
    array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
