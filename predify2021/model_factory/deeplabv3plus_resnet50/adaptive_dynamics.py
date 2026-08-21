import math

import torch
from torch import nn

from .adapters import UNIFIED_STATE_CHANNELS


INITIAL_UPDATE_RATE = 0.207


class AdaptiveUpdateRate(nn.Module):
    """Predict a spatial, channel-wise error-state update rate."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(
            UNIFIED_STATE_CHANNELS * 3,
            UNIFIED_STATE_CHANNELS,
            kernel_size=1,
        )
        nn.init.zeros_(self.projection.weight)
        nn.init.constant_(
            self.projection.bias,
            math.log(INITIAL_UPDATE_RATE / (1.0 - INITIAL_UPDATE_RATE)),
        )

    def forward(
        self,
        error: torch.Tensor,
        predicted: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat((error, predicted, previous_state), dim=1)
        return torch.sigmoid(self.projection(inputs))
