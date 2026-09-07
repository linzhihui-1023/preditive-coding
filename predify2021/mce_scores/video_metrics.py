from collections import deque

import torch

from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou


NUM_CLASSES = 19
IGNORE_LABEL = 255


def weighted_iou(confusion):
    frequency = confusion.sum(dim=1).float()
    frequency /= frequency.sum()
    return float(torch.nansum(compute_iou(confusion) * frequency).item())


def video_consistency(gt_window, prediction_window):
    gt = torch.stack(tuple(gt_window)).to(torch.int64)
    prediction = torch.stack(tuple(prediction_window)).to(torch.int64)
    gt_common = (gt == gt[0]).all(dim=0) & (gt[0] != IGNORE_LABEL)
    prediction_common = (prediction == prediction[0]).all(dim=0)
    consistent = gt_common & prediction_common
    return float(consistent.sum().item() / gt_common.sum().item()) if gt_common.any() else float("nan")


def video_effective_consistency(
    gt_window,
    prediction_window,
    num_classes=NUM_CLASSES,
    ignore_label=IGNORE_LABEL,
):
    """DTERN VEC_C (ICCV 2025, Eq. 17) for one C-frame clip.

    中文：对每个语义类别 k，分别求 C 帧中持续存在的 GT 区域与持续
    存在的预测区域，然后计算这两个持续区域的 IoU，最后对有效类别做
    宏平均。GT ignore 像素在任一帧出现时，该空间位置从整个 clip 排除。

    The paper writes 1/N_c over semantic categories.  As with standard mIoU,
    categories whose persistent GT/pred union is empty have undefined 0/0 IoU
    and are excluded from the macro mean.  This convention is reported by the
    prediction-only evaluator so it is explicit and reproducible.
    """
    gt = torch.stack(tuple(gt_window)).to(torch.int64)
    prediction = torch.stack(tuple(prediction_window)).to(torch.int64)
    if gt.shape != prediction.shape:
        raise ValueError("VEC GT/prediction windows must have identical shapes")
    if gt.ndim != 3:
        raise ValueError("VEC expects CxHxW windows")

    valid_all = (gt != int(ignore_label)).all(dim=0)
    class_ious = []
    for class_index in range(int(num_classes)):
        gt_persistent = (gt == class_index).all(dim=0) & valid_all
        pred_persistent = (prediction == class_index).all(dim=0) & valid_all
        union = gt_persistent | pred_persistent
        if not bool(union.any()):
            continue
        intersection = gt_persistent & pred_persistent
        class_ious.append(
            intersection.sum(dtype=torch.float64) / union.sum(dtype=torch.float64)
        )

    if not class_ious:
        return float("nan"), 0
    return float(torch.stack(class_ious).mean().item()), len(class_ious)


class VideoConsistency:
    def __init__(self, names):
        self.names = tuple(names)
        self.values = {8: {key: [] for key in self.names}, 16: {key: [] for key in self.names}}
        self.reset_sequence()

    def reset_sequence(self):
        self.gt = deque(maxlen=16)
        self.predictions = {key: deque(maxlen=16) for key in self.names}

    def append(self, gt, predictions):
        self.gt.append(gt.cpu().to(torch.uint8))
        for key in self.names:
            self.predictions[key].append(predictions[key].cpu().to(torch.uint8))
        for length in (8, 16):
            if len(self.gt) >= length:
                gt_window = list(self.gt)[-length:]
                for key in self.predictions:
                    value = video_consistency(gt_window, list(self.predictions[key])[-length:])
                    if value == value:
                        self.values[length][key].append(value)

    def means(self):
        return {length: {key: sum(values) / len(values) if values else float("nan") for key, values in series.items()} for length, series in self.values.items()}

    def window_counts(self):
        return {length: len(next(iter(series.values()))) for length, series in self.values.items()}


class VideoEffectiveConsistency:
    """Sliding-window VEC8/VEC16 accumulator aligned with existing mVC windows."""

    def __init__(self, names, num_classes=NUM_CLASSES, ignore_label=IGNORE_LABEL):
        self.names = tuple(names)
        self.num_classes = int(num_classes)
        self.ignore_label = int(ignore_label)
        self.values = {
            8: {key: [] for key in self.names},
            16: {key: [] for key in self.names},
        }
        self.valid_class_counts = {
            8: {key: [] for key in self.names},
            16: {key: [] for key in self.names},
        }
        self.reset_sequence()

    def reset_sequence(self):
        self.gt = deque(maxlen=16)
        self.predictions = {
            key: deque(maxlen=16)
            for key in self.names
        }

    def append(self, gt, predictions):
        self.gt.append(gt.cpu().to(torch.uint8))
        for key in self.names:
            self.predictions[key].append(predictions[key].cpu().to(torch.uint8))
        for length in (8, 16):
            if len(self.gt) < length:
                continue
            gt_window = list(self.gt)[-length:]
            for key in self.names:
                value, class_count = video_effective_consistency(
                    gt_window,
                    list(self.predictions[key])[-length:],
                    num_classes=self.num_classes,
                    ignore_label=self.ignore_label,
                )
                if value == value:
                    self.values[length][key].append(value)
                    self.valid_class_counts[length][key].append(class_count)

    def means(self):
        return {
            length: {
                key: sum(values) / len(values) if values else float("nan")
                for key, values in series.items()
            }
            for length, series in self.values.items()
        }

    def window_counts(self):
        return {
            length: {
                key: len(values)
                for key, values in series.items()
            }
            for length, series in self.values.items()
        }

    def mean_valid_class_counts(self):
        return {
            length: {
                key: (
                    sum(values) / len(values)
                    if values
                    else float("nan")
                )
                for key, values in series.items()
            }
            for length, series in self.valid_class_counts.items()
        }
