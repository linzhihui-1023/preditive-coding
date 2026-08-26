from collections import deque

import torch

from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou


def weighted_iou(confusion):
    iou = compute_iou(confusion)
    frequency = confusion.sum(dim=0).float()
    frequency = frequency / frequency.sum()
    return float(torch.nansum(iou * frequency).item())


def video_consistency(gt_window, prediction_window):
    gt = torch.stack(tuple(gt_window)).to(torch.int64)
    prediction = torch.stack(tuple(prediction_window)).to(torch.int64)
    common = (gt == gt[0]).all(dim=0) & (gt[0] != 255)
    correct = common & (prediction == gt[0]).all(dim=0) & (prediction[0] == gt[0])
    return float(correct[common].float().mean().item()) if common.any() else float("nan")


class VideoConsistency:
    def __init__(self):
        self.gt = deque(maxlen=16)
        self.predictions = {"static": deque(maxlen=16), "persistence": deque(maxlen=16), "predictor": deque(maxlen=16)}
        self.values = {8: {key: [] for key in self.predictions}, 16: {key: [] for key in self.predictions}}

    def append(self, gt, predictions):
        self.gt.append(gt.cpu().to(torch.uint8))
        for key, value in predictions.items():
            self.predictions[key].append(value.cpu().to(torch.uint8))
        for length in (8, 16):
            if len(self.gt) >= length:
                gt_window = list(self.gt)[-length:]
                for key in self.predictions:
                    self.values[length][key].append(video_consistency(gt_window, list(self.predictions[key])[-length:]))

    def means(self):
        return {length: {key: float(torch.tensor(values).nanmean().item()) if values else float("nan") for key, values in series.items()} for length, series in self.values.items()}
