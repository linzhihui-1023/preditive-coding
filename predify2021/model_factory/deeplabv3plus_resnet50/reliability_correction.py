import torch
from torch import nn
from torch.nn import functional as F

from .adapters import UNIFIED_STATE_CHANNELS


class ReliabilityAwareCorrection(nn.Module):
    """Estimate prediction and observation uncertainty and their correction gain."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(UNIFIED_STATE_CHANNELS * 4, UNIFIED_STATE_CHANNELS, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(UNIFIED_STATE_CHANNELS, UNIFIED_STATE_CHANNELS * 2, 1),
        )

    def forward(self, predicted, observation, error, dynamic_error):
        values = torch.cat((predicted, observation, error, dynamic_error), dim=1)
        raw_prediction, raw_observation = self.network(values).chunk(2, dim=1)
        prediction_uncertainty = F.softplus(raw_prediction) + 1e-6
        observation_uncertainty = F.softplus(raw_observation) + 1e-6
        gain = prediction_uncertainty / (
            prediction_uncertainty + observation_uncertainty
        )
        return prediction_uncertainty, observation_uncertainty, gain
