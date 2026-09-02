from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


VSPW_NUM_CLASSES = 124
VSPW_IGNORE_LABEL = 255


def semantic_mask_from_vspw_png(path):
    """Load a VSPW mask and map raw labels 0..124,255 to 124-class labels."""
    raw = np.asarray(Image.open(path).convert("L"), dtype=np.int64)
    mapped = raw - 1
    mapped[(raw == 0) | (raw == 255) | (raw > VSPW_NUM_CLASSES)] = VSPW_IGNORE_LABEL
    return torch.from_numpy(mapped)


def pil_rgb_to_unit_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class VSPWSegmentationDataset(Dataset):
    """Single-frame VSPW loader with video/frame metadata preserved."""

    def __init__(
        self,
        root,
        split: str,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported VSPW split: {split}")
        self.root = Path(root)
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        video_list = (self.root / f"{split}.txt").read_text().splitlines()
        self.video_ids = tuple(video for video in video_list if video)
        samples = []
        for video_id in self.video_ids:
            origin_dir = self.root / "data" / video_id / "origin"
            mask_dir = self.root / "data" / video_id / "mask"
            for image_path in sorted(origin_dir.glob("*.jpg")):
                mask_path = mask_dir / f"{image_path.stem}.png"
                if split != "test" and not mask_path.is_file():
                    raise FileNotFoundError(f"Missing VSPW mask for {image_path}")
                samples.append(
                    {
                        "image_path": image_path,
                        "mask_path": mask_path if split != "test" else None,
                        "video_id": video_id,
                        "frame_name": image_path.name,
                        "frame_id": image_path.stem,
                    }
                )
        self.samples = tuple(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            image = pil_rgb_to_unit_tensor(image)
        if self.split == "test":
            mask = None
        else:
            mask = semantic_mask_from_vspw_png(sample["mask_path"])
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
        metadata = {
            "image_path": str(sample["image_path"]),
            "mask_path": None if sample["mask_path"] is None else str(sample["mask_path"]),
            "video_id": sample["video_id"],
            "frame_name": sample["frame_name"],
            "frame_id": sample["frame_id"],
        }
        return image, mask, metadata


def build_vspw_dataset(root, split, image_transform=None, mask_transform=None):
    return VSPWSegmentationDataset(
        root=root,
        split=split,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
