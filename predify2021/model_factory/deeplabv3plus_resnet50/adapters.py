from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


HOST_STAGE_CHANNELS = (256, 512, 1024, 2048)
UNIFIED_STATE_CHANNELS = 128


@dataclass(frozen=True)
class BackboneFeatures:
    """The four spatial ResNet stage outputs in layer order."""

    c1: torch.Tensor
    c2: torch.Tensor
    c3: torch.Tensor
    c4: torch.Tensor

    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.c1, self.c2, self.c3, self.c4


@dataclass(frozen=True)
class UnifiedFeatures:
    """Four spatial features in the shared adapter channel space."""

    z1: torch.Tensor
    z2: torch.Tensor
    z3: torch.Tensor
    z4: torch.Tensor

    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.z1, self.z2, self.z3, self.z4


class InputAdapter(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            UNIFIED_STATE_CHANNELS,
            kernel_size=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(32, UNIFIED_STATE_CHANNELS)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(feature))


class OutputAdapter(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.projection = nn.Conv2d(
            UNIFIED_STATE_CHANNELS,
            out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.projection(delta)


class HostConditionedResidualWriteback(nn.Module):
    """Convert a unified-state correction into a host-conditioned residual."""

    def __init__(self, host_channels: int):
        super().__init__()
        self.host_projection = nn.Conv2d(
            host_channels,
            UNIFIED_STATE_CHANNELS,
            kernel_size=1,
            bias=False,
        )
        self.delta_projection = nn.Conv2d(
            UNIFIED_STATE_CHANNELS,
            UNIFIED_STATE_CHANNELS,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.output_projection = nn.Conv2d(
            UNIFIED_STATE_CHANNELS,
            host_channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.zeros_(self.output_projection.weight)

    def forward(
        self,
        host_feature: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        host_context = self.host_projection(host_feature)
        correction_command = self.delta_projection(delta)
        return self.output_projection(host_context * correction_command)


class MultiLayerAdapter(nn.Module):
    """Independent spatial adapters for ResNet layers 1 through 4."""

    def __init__(self):
        super().__init__()
        self.input_adapters = nn.ModuleList(
            InputAdapter(channels) for channels in HOST_STAGE_CHANNELS
        )
        self.output_adapters = nn.ModuleList(
            OutputAdapter(channels) for channels in HOST_STAGE_CHANNELS
        )
        self.gain = nn.Parameter(torch.zeros(len(HOST_STAGE_CHANNELS)))

    def encode(self, features: BackboneFeatures) -> UnifiedFeatures:
        values = tuple(
            adapter(feature)
            for adapter, feature in zip(self.input_adapters, features.as_tuple())
        )
        return UnifiedFeatures(*values)

    def decode_deltas(self, deltas: UnifiedFeatures) -> BackboneFeatures:
        values = tuple(
            adapter(delta)
            for adapter, delta in zip(self.output_adapters, deltas.as_tuple())
        )
        return BackboneFeatures(*values)

    def apply_deltas(
        self,
        features: BackboneFeatures,
        deltas: UnifiedFeatures,
    ) -> BackboneFeatures:
        values = tuple(
            feature + self.gain[index] * adapter(delta)
            for index, (feature, adapter, delta) in enumerate(
                zip(features.as_tuple(), self.output_adapters, deltas.as_tuple())
            )
        )
        return BackboneFeatures(*values)
