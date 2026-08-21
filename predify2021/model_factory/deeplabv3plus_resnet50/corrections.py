import torch
from torch import nn

from .adapters import UNIFIED_STATE_CHANNELS


class ErrorGainCorrection(nn.Module):
    """Map an instantaneous state error to an elementwise correction gain."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(UNIFIED_STATE_CHANNELS, UNIFIED_STATE_CHANNELS, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(UNIFIED_STATE_CHANNELS, UNIFIED_STATE_CHANNELS, 1),
            nn.Sigmoid(),
        )

    def forward(self, error: torch.Tensor) -> torch.Tensor:
        return self.network(error)
