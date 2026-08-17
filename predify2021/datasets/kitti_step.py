from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class KITTISTEPSegmentationDataset(Dataset):
    """Minimal KITTI-STEP single-frame semantic segmentation dataset entry.

    The loader is intentionally independent from the KITTI Raw pair loaders.
    It expects aligned image and semantic-mask paths under separate roots and
    returns `(image, mask, metadata)` for future segmentation experiments.
    """

    def __init__(
        self,
        image_root,
        mask_root,
        image_glob: str = "**/*.png",
        mask_suffix: str = ".png",
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_paths = tuple(sorted(self.image_root.glob(image_glob)))
        self.mask_paths = tuple(
            self.mask_root / image.relative_to(self.image_root).with_suffix(mask_suffix)
            for image in self.image_paths
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        metadata = {
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
        return image, mask, metadata
