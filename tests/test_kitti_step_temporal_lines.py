import math

import torch

from predify2021.mce_scores.train_kitti_step_persistent_blur_direct_correction import persistent_blur
from predify2021.mce_scores.video_metrics import VideoConsistency, video_consistency


def test_persistent_blur_schedule_and_clean_identity(monkeypatch):
    image = torch.rand(1, 3, 12, 20)
    assert persistent_blur(image, 2, 9) is image
    sigmas = []
    class Filter:
        def __init__(self, sigma): sigmas.append(sigma)
    monkeypatch.setattr("predify2021.mce_scores.train_kitti_step_persistent_blur_direct_correction.ImageFilter.GaussianBlur", Filter)
    monkeypatch.setattr("predify2021.mce_scores.train_kitti_step_persistent_blur_direct_correction.Image.Image.filter", lambda self, _: self)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    for index in range(3, 10): persistent_blur(image, index, 9)
    assert sigmas == [0.4, 0.8, 1.2, 1.6, 2.0, 2.0, 2.0]


def test_video_consistency_and_sequence_reset():
    gt = [torch.zeros(2, 3, dtype=torch.uint8) for _ in range(16)]
    assert video_consistency(gt[:8], gt[:8]) == 1.0
    stable_wrong = [torch.ones(2, 3, dtype=torch.uint8) for _ in range(8)]
    assert video_consistency(gt[:8], stable_wrong) == 1.0
    wrong = list(gt[:8]); wrong[-1] = torch.ones(2, 3, dtype=torch.uint8)
    assert video_consistency(gt[:8], wrong) < 1.0
    changing_gt = list(gt[:8]); changing_gt[-1] = torch.ones(2, 3, dtype=torch.uint8)
    assert math.isnan(video_consistency(changing_gt, stable_wrong))
    metric = VideoConsistency(("static",))
    for mask in gt: metric.append(mask, {"static": mask})
    assert metric.window_counts()[8] == 9
    assert metric.window_counts()[16] == 1
    metric.reset_sequence()
    for mask in gt[:7]: metric.append(mask, {"static": mask})
    assert metric.window_counts()[8] == 9
