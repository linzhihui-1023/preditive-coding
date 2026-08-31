import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.kitti_step_cityscapes_c import (
    CITYSCAPES_C_SEVERITIES,
    DEFAULT_CACHED_CORRUPTIONS,
    apply_cityscapes_c_corruption_uint8,
    corruption_cache_path,
    corruption_seed,
)


def parse_csv(value, allowed, cast=str):
    if not value:
        return tuple(allowed)
    selected = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    unknown = tuple(item for item in selected if item not in allowed)
    if unknown:
        raise ValueError(f"Unsupported values: {unknown}")
    return selected


def build_frame_records(groups):
    records = []
    for sequence_id, samples in groups.items():
        for sample in samples:
            records.append(
                (
                    str(sample["image_path"]),
                    str(sequence_id),
                    len(records),
                )
            )
    return records


def generate_one(task):
    (
        image_path,
        sequence_id,
        global_frame_index,
        frames_per_condition,
        corruption,
        severity,
        base_seed,
        cache_root,
    ) = task
    output_path = corruption_cache_path(
        cache_root,
        corruption,
        severity,
        sequence_id,
        image_path,
    )
    if output_path.exists():
        return 0

    with Image.open(image_path) as image:
        image_uint8 = np.array(image.convert("RGB"), dtype=np.uint8)

    sample_seed = corruption_seed(
        base_seed,
        frames_per_condition,
        global_frame_index,
        corruption,
        severity,
    )
    corrupted = apply_cityscapes_c_corruption_uint8(
        image_uint8,
        corruption,
        severity,
        seed=sample_seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(corrupted, mode="RGB").save(
        output_path,
        format="PNG",
        compress_level=1,
    )
    return 1


def main():
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    cache_root = Path(
        os.environ.get(
            "PREDIFY_CITYSCAPES_C_CACHE_ROOT",
            "/home/lin/predify/kitti_step_cityscapes_c_cache",
        )
    )
    corruptions = parse_csv(
        os.environ.get("PREDIFY_CITYSCAPES_C_PREGENERATE_CORRUPTIONS"),
        DEFAULT_CACHED_CORRUPTIONS,
    )
    severities = parse_csv(
        os.environ.get("PREDIFY_CITYSCAPES_C_SEVERITIES"),
        CITYSCAPES_C_SEVERITIES,
        int,
    )
    workers = int(os.environ.get("PREDIFY_CITYSCAPES_C_PREGENERATE_WORKERS", "8"))
    max_sequences = int(os.environ.get("PREDIFY_CITYSCAPES_C_MAX_SEQUENCES", "0"))
    base_seed = int(os.environ.get("PREDIFY_SEED", "0"))

    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    if max_sequences > 0:
        groups = dict(list(groups.items())[:max_sequences])

    frame_records = build_frame_records(groups)
    frames_per_condition = len(frame_records)
    tasks = [
        (
            image_path,
            sequence_id,
            global_frame_index,
            frames_per_condition,
            corruption,
            severity,
            base_seed,
            str(cache_root),
        )
        for corruption in corruptions
        for severity in severities
        for image_path, sequence_id, global_frame_index in frame_records
    ]

    print(
        f"Pregenerating {len(tasks)} images: corruptions={corruptions}, "
        f"severities={severities}, sequences={len(groups)}, workers={workers}",
        flush=True,
    )
    start = time.perf_counter()
    generated = 0
    processed = 0

    with Pool(processes=workers) as pool:
        for made in pool.imap_unordered(generate_one, tasks, chunksize=1):
            processed += 1
            generated += made
            if processed % 100 == 0 or processed == len(tasks):
                elapsed = time.perf_counter() - start
                rate = processed / elapsed if elapsed > 0 else 0.0
                print(
                    f"Processed {processed}/{len(tasks)} images "
                    f"({rate:.2f} images/s)",
                    flush=True,
                )

    elapsed = time.perf_counter() - start
    print(
        f"Completed pregeneration in {elapsed:.1f}s; "
        f"generated={generated}, reused={len(tasks) - generated}, "
        f"cache_root={cache_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
