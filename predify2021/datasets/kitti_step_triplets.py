from pathlib import Path
from typing import Callable, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset

from .kitti_step import KITTISTEPSegmentationDataset, pil_rgb_to_unit_tensor


class KITTISTEPTripletDataset(Dataset):
    """Chronological t-2, t-1, t RGB triplets within official split sequences."""

    def __init__(
        self,
        base_dataset: KITTISTEPSegmentationDataset,
        image_transform: Optional[Callable] = None,
    ):
        self.base_dataset = base_dataset
        self.image_transform = image_transform
        by_sequence = {}
        for sample in base_dataset.samples:
            by_sequence.setdefault(sample["sequence_id"], []).append(sample)
        self.triplets = []
        for sequence_id, samples in by_sequence.items():
            samples.sort(key=lambda sample: int(sample["frame_id"]))
            for index in range(2, len(samples)):
                window = samples[index - 2 : index + 1]
                frame_ids = [int(sample["frame_id"]) for sample in window]
                if frame_ids == list(range(frame_ids[0], frame_ids[0] + 3)):
                    self.triplets.append((sequence_id, tuple(window)))

    @classmethod
    def from_kitti_step_root(
        cls,
        root,
        split: str,
        image_transform: Optional[Callable] = None,
    ):
        return cls(
            KITTISTEPSegmentationDataset.from_kitti_step_root(root, split),
            image_transform=image_transform,
        )

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        sequence_id, samples = self.triplets[index]
        images = []
        frame_ids = []
        for sample in samples:
            with Image.open(Path(sample["image_path"])) as opened_image:
                image = opened_image.convert("RGB")
                image = self.image_transform(image) if self.image_transform else pil_rgb_to_unit_tensor(image)
            images.append(image)
            frame_ids.append(sample["frame_id"])
        return torch.stack(images), {
            "sequence_id": sequence_id,
            "frame_ids": tuple(frame_ids),
        }
