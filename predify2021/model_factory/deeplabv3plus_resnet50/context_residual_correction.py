import torch
from torch import nn

from .adapters import UNIFIED_STATE_CHANNELS


class ContextResidualCorrection(nn.Module):
    """Predict a zero-initialized gain-logit residual around a frozen gain."""

    def __init__(self):
        super().__init__()
        self.first = nn.Conv2d(UNIFIED_STATE_CHANNELS * 5, UNIFIED_STATE_CHANNELS, 1)
        self.middle = nn.Conv2d(
            UNIFIED_STATE_CHANNELS,
            UNIFIED_STATE_CHANNELS,
            3,
            padding=1,
        )
        self.last = nn.Conv2d(
            UNIFIED_STATE_CHANNELS,
            UNIFIED_STATE_CHANNELS,
            1,
        )
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(
        self,
        predicted,
        observation,
        error,
        dynamic_error,
        legacy_gain,
    ):
        context = torch.cat(
            (predicted, observation, error, dynamic_error, legacy_gain), dim=1
        )
        residual = self.last(
            torch.relu(self.middle(torch.relu(self.first(context))))
        )
        return residual
