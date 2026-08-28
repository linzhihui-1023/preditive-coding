from collections import deque

import torch

from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou


def weighted_iou(confusion):
    frequency = confusion.sum(dim=1).float()
    frequency /= frequency.sum()
    return float(torch.nansum(compute_iou(confusion) * frequency).item())


def video_consistency(gt_window, prediction_window):
    gt = torch.stack(tuple(gt_window)).to(torch.int64)
    prediction = torch.stack(tuple(prediction_window)).to(torch.int64)
    gt_common = (gt == gt[0]).all(dim=0) & (gt[0] != 255)
    prediction_common = (prediction == prediction[0]).all(dim=0)
    consistent = gt_common & prediction_common
    return float(consistent.sum().item() / gt_common.sum().item()) if gt_common.any() else float("nan")


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
