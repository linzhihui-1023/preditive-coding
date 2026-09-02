"""One evaluator for VSPW Host and FAST-B temporal modes."""

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from predify2021.datasets.vspw import (
    VSPW_IGNORE_LABEL,
    VSPW_NUM_CLASSES,
)
from predify2021.mce_scores.vspw_metrics import (
    compute_miou,
    official_vc_window_scores,
    update_confusion_matrix,
)
from predify2021.mce_scores.vspw_fast_b_common import (
    encode_vspw_frame,
    corrected_host_feature_with_size,
    restore_frame,
    reset_temporal_state,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature


class VSPWMetricAccumulator:
    """Streaming mIoU plus the exact Stage-V1 VC window convention."""

    def __init__(self):
        self.confusion = {
            name: np.zeros((VSPW_NUM_CLASSES, VSPW_NUM_CLASSES), dtype=np.int64)
            for name in ("host", "fast_b")
        }
        self.gt = []
        self.predictions = defaultdict(list)
        self.window_scores = {
            name: {8: [], 16: []} for name in ("host", "fast_b")
        }
        self.video_count = 0
        self.frame_count = 0
        self.current_video = None

    def _finish_video(self):
        if self.current_video is None:
            return
        for length in (8, 16):
            self.window_scores["host"][length].extend(
                score for score in official_vc_window_scores(
                    self.gt, self.predictions["host"], length
                ) if np.isfinite(score)
            )
            self.window_scores["fast_b"][length].extend(
                score for score in official_vc_window_scores(
                    self.gt, self.predictions["fast_b"], length
                ) if np.isfinite(score)
            )
        self.gt = []
        self.predictions = defaultdict(list)

    def add(self, video_id, target, host_prediction, fast_prediction):
        if self.current_video != video_id:
            self._finish_video()
            self.current_video = video_id
            self.video_count += 1
        target = target.to(torch.int64).cpu().numpy()
        host_prediction = host_prediction.to(torch.int64).cpu().numpy()
        fast_prediction = fast_prediction.to(torch.int64).cpu().numpy()
        update_confusion_matrix(self.confusion["host"], host_prediction, target)
        update_confusion_matrix(self.confusion["fast_b"], fast_prediction, target)
        self.gt.append(target)
        self.predictions["host"].append(host_prediction)
        self.predictions["fast_b"].append(fast_prediction)
        self.frame_count += 1

    def finish(self):
        self._finish_video()
        self.current_video = None
        result = {}
        for name, confusion in self.confusion.items():
            result[name] = {"mIoU": compute_miou(confusion)[0]}
        for name in ("host", "fast_b"):
            for length in (8, 16):
                values = self.window_scores[name][length]
                result[name][f"mVC{length}"] = (
                    float(np.mean(values)) if values else float("nan")
                )
                result[name][f"mVC{length}_window_count"] = len(values)
        return {
            "host": result["host"],
            "fast_b": result["fast_b"],
            "frame_count": self.frame_count,
            "video_count": self.video_count,
        }


def _single_frame_collate(samples):
    if len(samples) != 1:
        raise ValueError("VSPW validation requires batch_size=1 to preserve video order.")
    image, mask, metadata = samples[0]
    return image.unsqueeze(0), mask, metadata


def evaluate_vspw_fast_b(host, predictor, dataloader, device, temporal_mode="full"):
    if temporal_mode not in {"full", "reset"}:
        raise ValueError("temporal_mode must be 'full' or 'reset'")
    host.eval()
    predictor.eval()
    metrics = VSPWMetricAccumulator()
    state = reset_temporal_state()
    previous_video = None
    loss_sum = 0.0
    with torch.inference_mode():
        for images, masks, metadata in dataloader:
            video_id = metadata["video_id"]
            if video_id != previous_video:
                state = reset_temporal_state()
                previous_video = video_id
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                raw, observation = encode_vspw_frame(host, images)
                restored, state, _ = restore_frame(
                    predictor, observation, state, temporal_mode
                )
                host_logits = host.decode_from_host_feature(
                    HostFeature(raw.c4, raw.c1, tuple(images.shape[-2:]))
                )
                corrected = corrected_host_feature_with_size(
                    host, raw, observation, restored, tuple(images.shape[-2:])
                )
                fast_logits = host.decode_from_host_feature(corrected)
                loss_sum += float(
                    F.cross_entropy(
                        fast_logits,
                        masks.to(device, non_blocking=True),
                        ignore_index=VSPW_IGNORE_LABEL,
                    ).item()
                )
            metrics.add(
                video_id,
                masks,
                host_logits.argmax(dim=1)[0],
                fast_logits.argmax(dim=1)[0],
            )
    result = metrics.finish()
    result["fast_b"]["loss"] = loss_sum / max(result["frame_count"], 1)
    result["temporal_mode"] = temporal_mode
    return result


def evaluate_vspw_host(host, dataloader, device):
    """Evaluate ISS Host with the same accumulator used by FAST-B."""
    host.eval()
    metrics = VSPWMetricAccumulator()
    loss_sum = 0.0
    with torch.inference_mode():
        for images, masks, metadata in dataloader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                raw, _ = encode_vspw_frame(host, images)
                logits = host.decode_from_host_feature(
                    HostFeature(raw.c4, raw.c1, tuple(images.shape[-2:]))
                )
                loss_sum += float(
                    F.cross_entropy(
                        logits,
                        masks.to(device, non_blocking=True),
                        ignore_index=VSPW_IGNORE_LABEL,
                    ).item()
                )
            prediction = logits.argmax(dim=1)[0]
            metrics.add(metadata["video_id"], masks, prediction, prediction)
    result = metrics.finish()
    result["host"]["loss"] = loss_sum / max(result["frame_count"], 1)
    return result


def build_validation_loader(dataset, num_workers=8):
    from torch.utils.data import DataLoader

    workers = max(0, int(num_workers))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=_single_frame_collate,
    )
