"""Small causal gain head for Prediction--Observation fusion."""

import torch
from torch import nn
from torch.nn import functional as F


class Z4AdaptiveGainHead(nn.Module):
    """Map four scalar error statistics to one frame-wise gain in [0, 1]."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def statistics(observation_z4, prediction_z4, previous_magnitude=None):
        error = observation_z4 - prediction_z4
        magnitude = error.abs().mean(dim=(1, 2, 3))
        relative = error.square().mean(dim=(1, 2, 3)).sqrt() / (
            observation_z4.square().mean(dim=(1, 2, 3)).sqrt() + 1e-8
        )
        if previous_magnitude is None:
            change = torch.zeros_like(magnitude)
        else:
            change = (magnitude - previous_magnitude).abs()
        observation_gap = observation_z4.mean(dim=(2, 3))
        prediction_gap = prediction_z4.mean(dim=(2, 3))
        similarity = 1.0 - F.cosine_similarity(
            observation_gap, prediction_gap, dim=1, eps=1e-8
        )
        stats = torch.stack((magnitude, relative, change, similarity), dim=1)
        return error, stats

    def forward(self, observation_z4, prediction_z4, previous_magnitude=None):
        error, stats = self.statistics(
            observation_z4, prediction_z4, previous_magnitude
        )
        gain = torch.sigmoid(self.network(stats)).reshape(-1, 1, 1, 1)
        return gain, stats, error
