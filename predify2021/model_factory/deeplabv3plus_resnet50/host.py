from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_CONFIG_NAME = "deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024"
CITYSCAPES_CHECKPOINT_NAME = (
    "deeplabv3plus_r50-d8_512x1024_80k_cityscapes_"
    "20200606_114049-f9fb496d.pth"
)
CITYSCAPES_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/"
    "deeplabv3plus_r50-d8_512x1024_80k_cityscapes/"
    f"{CITYSCAPES_CHECKPOINT_NAME}"
)
CITYSCAPES_RGB_MEAN = (123.675, 116.28, 103.53)
CITYSCAPES_RGB_STD = (58.395, 57.12, 57.375)


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


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        dilation=1,
    ):
        super().__init__()
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise_conv = ConvModule(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class MMSegResNetV1cBackbone(nn.Module):
    """ResNetV1c R-50 D8 backbone layout used by MMSegmentation v0.x."""

    def __init__(self):
        super().__init__()
        reference = resnet50(
            weights=None,
            replace_stride_with_dilation=(False, True, True),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = reference.maxpool
        self.layer1 = reference.layer1
        self.layer2 = reference.layer2
        self.layer3 = reference.layer3
        self.layer4 = reference.layer4

    def forward(self, images):
        x = self.stem(images)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4


class MMSegDepthwiseSeparableASPPHead(nn.Module):
    def __init__(self, num_classes=CITYSCAPES_NUM_CLASSES):
        super().__init__()
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(2048, 512, kernel_size=1),
        )
        self.aspp_modules = nn.ModuleList(
            [
                ConvModule(2048, 512, kernel_size=1),
                DepthwiseSeparableConvModule(
                    2048, 512, kernel_size=3, padding=12, dilation=12
                ),
                DepthwiseSeparableConvModule(
                    2048, 512, kernel_size=3, padding=24, dilation=24
                ),
                DepthwiseSeparableConvModule(
                    2048, 512, kernel_size=3, padding=36, dilation=36
                ),
            ]
        )
        self.bottleneck = ConvModule(512 * 5, 512, kernel_size=3, padding=1)
        self.c1_bottleneck = ConvModule(256, 48, kernel_size=1)
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(560, 512, kernel_size=3, padding=1),
            DepthwiseSeparableConvModule(512, 512, kernel_size=3, padding=1),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, host_feature: HostFeature):
        high = host_feature.tensor
        image_pool = self.image_pool(high)
        image_pool = F.interpolate(
            image_pool,
            size=high.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        aspp = [image_pool]
        aspp.extend(module(high) for module in self.aspp_modules)
        output = self.bottleneck(torch.cat(aspp, dim=1))
        output = F.interpolate(
            output,
            size=host_feature.low_level.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        low = self.c1_bottleneck(host_feature.low_level)
        output = self.sep_bottleneck(torch.cat([output, low], dim=1))
        output = self.conv_seg(self.dropout(output))
        return F.interpolate(
            output,
            size=host_feature.output_size,
            mode="bilinear",
            align_corners=False,
        )


class MMSegFCNAuxiliaryHead(nn.Module):
    def __init__(self, num_classes=CITYSCAPES_NUM_CLASSES):
        super().__init__()
        self.convs = nn.Sequential(ConvModule(1024, 256, kernel_size=3, padding=1))
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, c3, output_size):
        output = self.conv_seg(self.dropout(self.convs(c3)))
        return F.interpolate(
            output,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class DeepLabV3PlusResNet50Host(nn.Module):
    """MMSegmentation DeepLabV3+ R-50 D8 Cityscapes static host."""

    def __init__(
        self,
        num_classes: int = CITYSCAPES_NUM_CLASSES,
        checkpoint_path: Optional[str] = None,
        load_cityscapes_checkpoint: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = MMSegResNetV1cBackbone()
        self.decode_head = MMSegDepthwiseSeparableASPPHead(num_classes=num_classes)
        self.auxiliary_head = MMSegFCNAuxiliaryHead(num_classes=num_classes)
        self.register_buffer(
            "input_mean",
            torch.tensor(CITYSCAPES_RGB_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(CITYSCAPES_RGB_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        self.cityscapes_checkpoint_loaded = False
        self.checkpoint_load_report = None
        if load_cityscapes_checkpoint:
            self.load_cityscapes_checkpoint(checkpoint_path)
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        return (images * 255.0 - self.input_mean) / self.input_std

    def extract_host_feature(self, images: torch.Tensor) -> HostFeature:
        c1, _, _, c4 = self.backbone(self.preprocess_images(images))
        return HostFeature(
            tensor=c4,
            low_level=c1,
            output_size=tuple(images.shape[-2:]),
        )

    def decode_from_host_feature(self, host_feature: HostFeature) -> torch.Tensor:
        return self.decode_head(host_feature)

    def forward_with_host_feature(self, images: torch.Tensor) -> DeepLabV3PlusHostOutput:
        host_feature = self.extract_host_feature(images)
        logits = self.decode_from_host_feature(host_feature)
        return DeepLabV3PlusHostOutput(logits=logits, host_feature=host_feature)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_with_host_feature(images).logits

    def load_cityscapes_checkpoint(self, checkpoint_path: Optional[str] = None):
        checkpoint = load_official_cityscapes_checkpoint(checkpoint_path)
        state_dict = checkpoint["state_dict"]
        self.load_state_dict(state_dict, strict=True)
        backbone_keys = sum(key.startswith("backbone.") for key in state_dict)
        decode_keys = sum(key.startswith("decode_head.") for key in state_dict)
        auxiliary_keys = sum(key.startswith("auxiliary_head.") for key in state_dict)
        if backbone_keys == 0 or decode_keys == 0:
            raise RuntimeError("Official checkpoint did not contain backbone and decode_head weights.")
        self.cityscapes_checkpoint_loaded = True
        self.checkpoint_load_report = {
            "config": CITYSCAPES_CONFIG_NAME,
            "checkpoint": CITYSCAPES_CHECKPOINT_NAME,
            "source": CITYSCAPES_CHECKPOINT_URL,
            "strict": True,
            "backbone_key_count": backbone_keys,
            "decode_head_key_count": decode_keys,
            "auxiliary_head_key_count": auxiliary_keys,
            "num_classes": self.num_classes,
        }
        return self.checkpoint_load_report


def default_checkpoint_path() -> Path:
    hub_dir = Path(torch.hub.get_dir())
    return hub_dir / "checkpoints" / CITYSCAPES_CHECKPOINT_NAME


def load_official_cityscapes_checkpoint(checkpoint_path: Optional[str] = None):
    if checkpoint_path is None:
        checkpoint = torch.hub.load_state_dict_from_url(
            CITYSCAPES_CHECKPOINT_URL,
            model_dir=str(default_checkpoint_path().parent),
            map_location="cpu",
            file_name=CITYSCAPES_CHECKPOINT_NAME,
        )
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError("Expected an MMSegmentation checkpoint with a state_dict.")
    return checkpoint


def build_deeplabv3plus_resnet50_host(
    num_classes: int = CITYSCAPES_NUM_CLASSES,
    checkpoint_path: Optional[str] = None,
    load_cityscapes_checkpoint: bool = True,
    freeze_backbone: bool = False,
) -> DeepLabV3PlusResNet50Host:
    return DeepLabV3PlusResNet50Host(
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
        load_cityscapes_checkpoint=load_cityscapes_checkpoint,
        freeze_backbone=freeze_backbone,
    )
