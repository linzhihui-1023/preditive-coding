from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.segmentation.deeplabv3 import ASPP


@dataclass(frozen=True)
class HostFeature:
    tensor: torch.Tensor
    low_level: torch.Tensor
    output_size: Tuple[int, int]

    def replace(self, tensor: torch.Tensor) -> "HostFeature":
        return HostFeature(
            tensor=tensor,
            low_level=self.low_level,
            output_size=self.output_size,
        )


@dataclass(frozen=True)
class DeepLabV3PlusHostOutput:
    logits: torch.Tensor
    host_feature: HostFeature


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V2):
        super().__init__()
        backbone = resnet50(
            weights=weights,
            replace_stride_with_dilation=(False, False, True),
        )
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, images: torch.Tensor) -> HostFeature:
        output_size = tuple(images.shape[-2:])
        x = self.stem(images)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)
        return HostFeature(
            tensor=x,
            low_level=low_level,
            output_size=output_size,
        )


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        atrous_rates=(6, 12, 18),
        high_channels: int = 2048,
        low_channels: int = 256,
        aspp_channels: int = 256,
        low_projection_channels: int = 48,
    ):
        super().__init__()
        self.aspp = ASPP(high_channels, atrous_rates, aspp_channels)
        self.low_projection = nn.Sequential(
            nn.Conv2d(low_channels, low_projection_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_projection_channels),
            nn.ReLU(inplace=True),
        )
        decoder_channels = aspp_channels + low_projection_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, host_feature: HostFeature) -> torch.Tensor:
        high = self.aspp(host_feature.tensor)
        high = F.interpolate(
            high,
            size=host_feature.low_level.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        low = self.low_projection(host_feature.low_level)
        logits = self.decoder(torch.cat([high, low], dim=1))
        return F.interpolate(
            logits,
            size=host_feature.output_size,
            mode="bilinear",
            align_corners=False,
        )


class DeepLabV3PlusResNet50Host(nn.Module):
    """Static DeepLabV3+ ResNet-50 segmentation host.

    The host exposes a model-agnostic feature boundary for future adapters:
    callers can extract `HostFeature.tensor`, replace it with a modified tensor
    through `HostFeature.replace`, and continue through the segmentation head.
    """

    def __init__(
        self,
        num_classes: int = 21,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        self.feature_extractor = ResNet50FeatureExtractor(weights=weights)
        self.decoder = DeepLabV3PlusDecoder(num_classes=num_classes)
        if freeze_backbone:
            self.feature_extractor.requires_grad_(False)

    def extract_host_feature(self, images: torch.Tensor) -> HostFeature:
        return self.feature_extractor(images)

    def decode_from_host_feature(self, host_feature: HostFeature) -> torch.Tensor:
        return self.decoder(host_feature)

    def forward_with_host_feature(self, images: torch.Tensor) -> DeepLabV3PlusHostOutput:
        host_feature = self.extract_host_feature(images)
        logits = self.decode_from_host_feature(host_feature)
        return DeepLabV3PlusHostOutput(logits=logits, host_feature=host_feature)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_with_host_feature(images).logits


def build_deeplabv3plus_resnet50_host(
    num_classes: int = 21,
    pretrained_backbone: bool = True,
    freeze_backbone: bool = False,
) -> DeepLabV3PlusResNet50Host:
    return DeepLabV3PlusResNet50Host(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        freeze_backbone=freeze_backbone,
    )
