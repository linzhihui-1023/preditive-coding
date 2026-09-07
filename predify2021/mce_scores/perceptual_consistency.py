"""Perceptual Consistency (PC) for video semantic segmentation.

中文：WACV 2022《Perceptual Consistency in Video Segmentation》的 SPC/PC
评测实现。

Protocol reproduced from the authors' public `yizhezhang2000/SPC/example.py`:
- ImageNet-pretrained ResNet-18 perceptual extractor;
- `torch.nn.Sequential(*list(resnet18.children())[:-4])`, i.e. through layer2;
- cosine all-pairs feature correlation;
- Eq.1 unconstrained maximum correlation;
- Eq.2 same-segmentation-label constrained maximum correlation;
- Eq.3 frame-average-normalized bidirectional score, taking the smaller direction;
- video PC is the mean over consecutive frame pairs (paper Eq.4).

The segmentation map is nearest-neighbour resized to the actual feature-map HxW.
This preserves the intended spatial alignment between the perceptual features and
semantic labels.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_resnet18_perceptual_extractor(device="cuda"):
    """Return the official ResNet-18 feature prefix (through layer2)."""
    try:
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except (AttributeError, TypeError):
        # Compatibility with older torchvision used by some project envs.
        backbone = models.resnet18(pretrained=True)
    extractor = nn.Sequential(*list(backbone.children())[:-4]).to(device)
    extractor.eval()
    extractor.requires_grad_(False)
    return extractor


def _image_tensor(path, device):
    with Image.open(Path(path)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


@torch.inference_mode()
def extract_perceptual_feature(extractor, image_path):
    device = next(extractor.parameters()).device
    feature = extractor(_image_tensor(image_path, device))[0]
    if feature.ndim != 3:
        raise RuntimeError("ResNet-18 perceptual feature must be CxHxW")
    return feature.detach()


def resize_semantic_to_feature(prediction, feature_hw, device=None):
    prediction = torch.as_tensor(prediction).long()
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError("semantic prediction must be HxW")
    target_device = prediction.device if device is None else torch.device(device)
    prediction = prediction.to(target_device)
    resized = F.interpolate(
        prediction[None, None].float(),
        size=tuple(feature_hw),
        mode="nearest",
    )[0, 0].long()
    return resized


def perceptual_correlation(feature_a, feature_b):
    """Cosine all-pairs correlation used by SPC Eq.1/Eq.2."""
    if feature_a.ndim != 3 or feature_b.ndim != 3:
        raise ValueError("features must be CxHxW")
    if feature_a.shape[0] != feature_b.shape[0]:
        raise ValueError("feature channels must match")
    a = feature_a.reshape(feature_a.shape[0], -1)
    b = feature_b.reshape(feature_b.shape[0], -1)
    a = F.normalize(a, p=2, dim=0).transpose(0, 1)
    b = F.normalize(b, p=2, dim=0)
    return torch.matmul(a, b)


def perceptual_consistency_from_correlation(correlation, labels_a, labels_b):
    """Authors' SPC Eq.1--Eq.3 for one predicted segmentation pair."""
    labels_a = torch.as_tensor(labels_a, device=correlation.device).reshape(-1)
    labels_b = torch.as_tensor(labels_b, device=correlation.device).reshape(-1)
    if correlation.shape != (labels_a.numel(), labels_b.numel()):
        raise ValueError("correlation and flattened semantic-map sizes do not match")

    max_a_unconstrained = correlation.max(dim=1).values
    max_b_unconstrained = correlation.max(dim=0).values

    # Equivalent to the authors' one-hot seg_A^T @ seg_B mask.  Multiplication
    # (rather than masked_fill(-inf)) deliberately reproduces their official
    # SPC behavior, where disallowed correlations become zero.
    same_label = labels_a[:, None] == labels_b[None, :]
    correlation_seg = correlation * same_label.to(correlation.dtype)
    max_a_constrained = correlation_seg.max(dim=1).values
    max_b_constrained = correlation_seg.max(dim=0).values

    mean_a = correlation.mean(dim=1)
    mean_b = correlation.mean(dim=0)
    pc_a_map = (
        (max_a_constrained - mean_a)
        / (max_a_unconstrained - mean_a)
    )
    pc_b_map = (
        (max_b_constrained - mean_b)
        / (max_b_unconstrained - mean_b)
    )
    pc_a = pc_a_map.mean()
    pc_b = pc_b_map.mean()
    pc = torch.minimum(pc_a, pc_b)
    return {
        "pc": float(pc.item()),
        "pc_a_to_b": float(pc_a.item()),
        "pc_b_to_a": float(pc_b.item()),
        "finite_a_fraction": float(torch.isfinite(pc_a_map).float().mean().item()),
        "finite_b_fraction": float(torch.isfinite(pc_b_map).float().mean().item()),
    }


@torch.inference_mode()
def perceptual_consistency_pair(feature_a, feature_b, prediction_a, prediction_b):
    """Convenience wrapper for one consecutive frame pair."""
    correlation = perceptual_correlation(feature_a, feature_b)
    labels_a = resize_semantic_to_feature(
        prediction_a,
        feature_a.shape[-2:],
        device=correlation.device,
    )
    labels_b = resize_semantic_to_feature(
        prediction_b,
        feature_b.shape[-2:],
        device=correlation.device,
    )
    return perceptual_consistency_from_correlation(correlation, labels_a, labels_b)
