import torch
from torch import nn

from .adapters import UNIFIED_STATE_CHANNELS, UnifiedFeatures


class SpatialPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(UNIFIED_STATE_CHANNELS * 2, UNIFIED_STATE_CHANNELS, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(UNIFIED_STATE_CHANNELS, UNIFIED_STATE_CHANNELS, 3, padding=1),
        )

    def forward(self, previous_state, previous_delta):
        return self.network(torch.cat((previous_state, previous_delta), dim=1))


class MultiLayerPredictor(nn.Module):
    """Independent identical-capacity predictors for the four host stages."""

    def __init__(self):
        super().__init__()
        self.predictors = nn.ModuleList(SpatialPredictor() for _ in range(4))

    def forward(
        self,
        previous_state: UnifiedFeatures,
        previous_delta: UnifiedFeatures,
    ) -> UnifiedFeatures:
        values = tuple(
            predictor(state, delta)
            for predictor, state, delta in zip(
                self.predictors,
                previous_state.as_tuple(),
                previous_delta.as_tuple(),
            )
        )
        return UnifiedFeatures(*values)
