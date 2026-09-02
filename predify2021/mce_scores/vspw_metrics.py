from pathlib import Path

import numpy as np
from PIL import Image

from predify2021.datasets.vspw import (
    VSPW_IGNORE_LABEL,
    VSPW_NUM_CLASSES,
    semantic_mask_from_vspw_png,
)


def update_confusion_matrix(confusion, prediction, target):
    valid = (target != VSPW_IGNORE_LABEL) & (target >= 0) & (target < VSPW_NUM_CLASSES)
    encoded = VSPW_NUM_CLASSES * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    confusion += np.bincount(encoded, minlength=VSPW_NUM_CLASSES**2).reshape(
        VSPW_NUM_CLASSES, VSPW_NUM_CLASSES
    )


def compute_miou(confusion):
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection
    iou = np.divide(intersection, union, out=np.full_like(intersection, np.nan), where=union != 0)
    return float(np.nanmean(iou)), iou


def official_vc_window_scores(gt_frames, prediction_frames, clip_size):
    """Exact VSPW VC_perclip.py window convention (range(T-clip_size))."""
    if len(gt_frames) != len(prediction_frames):
        raise ValueError("GT and prediction frame counts differ")
    scores = []
    for start in range(len(gt_frames) - clip_size):
        gt = np.stack(gt_frames[start : start + clip_size])
        prediction = np.stack(prediction_frames[start : start + clip_size])
        gt_common = np.ones(gt.shape[1:], dtype=bool)
        pred_common = np.ones(prediction.shape[1:], dtype=bool)
        for offset in range(1, clip_size):
            gt_common &= gt[0] == gt[offset]
            pred_common &= prediction[0] == prediction[offset]
        denominator = gt_common.sum()
        if denominator:
            scores.append(float((gt_common & pred_common).sum() / denominator))
    return scores


def compute_vspw_vc(root, prediction_root, split="val"):
    root = Path(root)
    prediction_root = Path(prediction_root)
    results = {}
    for clip_size in (8, 16):
        all_scores = []
        for video_id in (root / f"{split}.txt").read_text().splitlines():
            if not video_id:
                continue
            mask_paths = sorted((root / "data" / video_id / "mask").glob("*.png"))
            if len(mask_paths) <= clip_size:
                continue
            gt_frames = [np.asarray(Image.open(path).convert("L"), dtype=np.uint8) for path in mask_paths]
            prediction_frames = [
                np.asarray(Image.open(prediction_root / video_id / path.name).convert("L"), dtype=np.uint8)
                for path in mask_paths
            ]
            all_scores.extend(official_vc_window_scores(gt_frames, prediction_frames, clip_size))
        results[clip_size] = {
            "score": float(np.nanmean(all_scores)) if all_scores else float("nan"),
            "window_count": len(all_scores),
        }
    return results


def compute_vspw_miou(root, prediction_root, split="val"):
    root = Path(root)
    prediction_root = Path(prediction_root)
    confusion = np.zeros((VSPW_NUM_CLASSES, VSPW_NUM_CLASSES), dtype=np.int64)
    frame_count = 0
    for video_id in (root / f"{split}.txt").read_text().splitlines():
        if not video_id:
            continue
        for mask_path in sorted((root / "data" / video_id / "mask").glob("*.png")):
            target = semantic_mask_from_vspw_png(mask_path).numpy()
            prediction_path = prediction_root / video_id / mask_path.name
            prediction = np.asarray(Image.open(prediction_path).convert("L"), dtype=np.int64)
            update_confusion_matrix(confusion, prediction, target)
            frame_count += 1
    miou, per_class_iou = compute_miou(confusion)
    return {"mIoU": miou, "per_class_iou": per_class_iou.tolist(), "confusion_matrix": confusion.tolist(), "frame_count": frame_count}
