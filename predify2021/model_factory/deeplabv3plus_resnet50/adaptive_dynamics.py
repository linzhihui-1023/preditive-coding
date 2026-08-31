import math
import os

import torch
from torch import nn

from .adapters import UNIFIED_STATE_CHANNELS


# Provisional experiment defaults expressed in the original dynamic-error
# parameters. The learned update rate is initialized from Ts / tau_e.
DYNAMIC_ERROR_SAMPLE_TIME = float(
    os.environ.get("PREDIFY_DYNAMIC_ERROR_SAMPLE_TIME", "0.1035")
)
DYNAMIC_ERROR_TIME_CONSTANT = float(
    os.environ.get("PREDIFY_DYNAMIC_ERROR_TIME_CONSTANT", "0.5")
)
DYNAMIC_ERROR_GAIN = float(
    os.environ.get("PREDIFY_DYNAMIC_ERROR_GAIN", "1.0")
)
INITIAL_UPDATE_RATE = (
    DYNAMIC_ERROR_SAMPLE_TIME / DYNAMIC_ERROR_TIME_CONSTANT
)


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
