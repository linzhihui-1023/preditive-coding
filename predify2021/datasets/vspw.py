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


class VSPWSequentialClipDataset(Dataset):
    """Continuous clips from one VSPW video at a time.

    The dataset is deliberately map-style: a sequential sampler can still use
    several workers for decoding/prefetching while the main process receives
    clips in video order.  A clip never crosses a video boundary or a gap in
    the numeric frame ids.  The final clip of a run may be shorter than
    ``clip_length`` so every available frame remains addressable.
    """

    def __init__(
        self,
        root,
        split: str,
        clip_length: int = 16,
        stride: Optional[int] = None,
        drop_short_clips: bool = False,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported VSPW split: {split}")
        if clip_length < 1:
            raise ValueError("clip_length must be positive")
        self.root = Path(root)
        self.split = split
        self.clip_length = int(clip_length)
        self.stride = int(stride if stride is not None else clip_length)
        self.drop_short_clips = bool(drop_short_clips)
        if self.stride < 1:
            raise ValueError("stride must be positive")
        self.video_ids = tuple(
            video for video in (self.root / f"{split}.txt").read_text().splitlines() if video
        )
        self.clips = tuple(self._build_clips())

    def _frame_samples(self, video_id):
        origin_dir = self.root / "data" / video_id / "origin"
        mask_dir = self.root / "data" / video_id / "mask"
        samples = []
        for image_path in sorted(origin_dir.glob("*.jpg"), key=lambda path: int(path.stem)):
            mask_path = mask_dir / f"{image_path.stem}.png"
            if self.split != "test" and not mask_path.is_file():
                raise FileNotFoundError(f"Missing VSPW mask for {image_path}")
            samples.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_path if self.split != "test" else None,
                    "video_id": video_id,
                    "frame_name": image_path.name,
                    "frame_id": image_path.stem,
                    "frame_index": int(image_path.stem),
                }
            )
        return samples

    def _build_clips(self):
        for video_id in self.video_ids:
            frames = self._frame_samples(video_id)
            if not frames:
                continue
            run = []
            run_index = 0
            for sample in frames:
                if run and sample["frame_index"] != run[-1]["frame_index"] + 1:
                    yield from self._clips_for_run(video_id, run_index, run)
                    run_index += 1
                    run = []
                run.append(sample)
            yield from self._clips_for_run(video_id, run_index, run)

    def _clips_for_run(self, video_id, run_index, run):
        sequence_id = f"{video_id}#run{run_index}"
        for start in range(0, len(run), self.stride):
            frames = tuple(run[start : start + self.clip_length])
            if not frames:
                continue
            if self.drop_short_clips and len(frames) < self.clip_length:
                continue
            yield {
                "video_id": video_id,
                "sequence_id": sequence_id,
                "is_sequence_start": start == 0,
                "is_sequence_end": start + len(frames) >= len(run),
                "frame_ids": tuple(sample["frame_id"] for sample in frames),
                "samples": frames,
            }

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip = self.clips[index]
        images = []
        masks = []
        metadata = []
        for sample in clip["samples"]:
            images.append(pil_rgb_to_unit_tensor(Image.open(sample["image_path"]).convert("RGB")))
            if self.split != "test":
                masks.append(semantic_mask_from_vspw_png(sample["mask_path"]))
            metadata.append(
                {
                    "image_path": str(sample["image_path"]),
                    "mask_path": None if sample["mask_path"] is None else str(sample["mask_path"]),
                    "video_id": sample["video_id"],
                    "sequence_id": clip["sequence_id"],
                    "frame_name": sample["frame_name"],
                    "frame_id": sample["frame_id"],
                }
            )
        result = {
            "images": torch.stack(images, dim=0),
            "masks": None if self.split == "test" else torch.stack(masks, dim=0),
            "metadata": metadata,
            "video_id": clip["video_id"],
            "sequence_id": clip["sequence_id"],
            "is_sequence_start": clip["is_sequence_start"],
            "is_sequence_end": clip["is_sequence_end"],
            "frame_ids": clip["frame_ids"],
        }
        return result


def collate_vspw_sequential_clips(samples):
    """Keep one clip intact; temporal state is owned by the main process."""
    if len(samples) != 1:
        raise ValueError("Sequential VSPW loading requires batch_size=1.")
    return samples[0]


def build_vspw_dataset(root, split, image_transform=None, mask_transform=None):
    return VSPWSegmentationDataset(
        root=root,
        split=split,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
