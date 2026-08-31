"""Check one Cityscapes validation pair and five online published corruptions."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from predify2021.datasets.cityscapes import (
    CITYSCAPES_IMAGE_SIZE,
    CityscapesSegmentationDataset,
    label_ids_to_train_ids,
    cityscapes_split_counts,
)
from predify2021.datasets.cityscapes_corruptions import (
    apply_published_corruption,
    load_published_protocol,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=os.environ.get("CITYSCAPES_DATASET", "/home/lin/datasets/cityscapes"),
    )
    args = parser.parse_args()
    root = Path(args.root)
    dataset = CityscapesSegmentationDataset(root, split="val")
    sample = dataset.samples[0]
    image = np.asarray(Image.open(sample["image_path"]).convert("RGB"), dtype=np.uint8)
    label_ids = np.asarray(Image.open(sample["label_path"]), dtype=np.uint8)
    train_ids = label_ids_to_train_ids(label_ids)
    assert image.shape[:2][::-1] == CITYSCAPES_IMAGE_SIZE
    assert train_ids.shape == image.shape[:2]
    assert set(np.unique(train_ids)).issubset(set(range(19)) | {255})

    protocol = load_published_protocol()
    assert len(protocol["corruptions"]) == 19
    for severity in range(1, 6):
        corrupted = apply_published_corruption(image, "gaussian_blur", severity)
        assert corrupted.shape == image.shape
        assert corrupted.dtype == np.uint8
        assert np.array_equal(train_ids, label_ids_to_train_ids(label_ids))

    result = {
        "root": str(root),
        "sample": str(sample["image_path"]),
        "image_size": [image.shape[1], image.shape[0]],
        "label_mapping": "official cityscapesscripts trainId mapping",
        "split_counts": cityscapes_split_counts(root),
        "corruption": "gaussian_blur",
        "severity_levels": [1, 2, 3, 4, 5],
        "passed": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
