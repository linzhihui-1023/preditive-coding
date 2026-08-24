import torch
from torch import nn
from torch.nn import functional as F

from .adapters import UNIFIED_STATE_CHANNELS


class LocalErrorMemoryAlignment(nn.Module):
    """Move prior error memory with single-head local cross-attention."""

    def __init__(self, projection_channels=32, window_size=7):
        super().__init__()
        self.projection_channels = projection_channels
        self.window_size = window_size
        self.padding = window_size // 2
        self.query_projection = nn.Conv2d(
            UNIFIED_STATE_CHANNELS, projection_channels, kernel_size=1
        )
        self.key_projection = nn.Conv2d(
            UNIFIED_STATE_CHANNELS, projection_channels, kernel_size=1
        )

    def forward(
        self,
        observation: torch.Tensor,
        previous_posterior: torch.Tensor,
        previous_error: torch.Tensor,
        return_attention=False,
    ):
        batch_size, _, height, width = observation.shape
        query = F.normalize(self.query_projection(observation), dim=1)
        key = F.normalize(self.key_projection(previous_posterior), dim=1)
        neighborhood_size = self.window_size * self.window_size
        key_windows = F.unfold(
            key,
            kernel_size=self.window_size,
            padding=self.padding,
        ).reshape(
            batch_size,
            self.projection_channels,
            neighborhood_size,
            height,
            width,
        )
        scores = (query.unsqueeze(2) * key_windows).sum(dim=1)
        valid = F.unfold(
            torch.ones(
                batch_size,
                1,
                height,
                width,
                device=observation.device,
                dtype=observation.dtype,
            ),
            kernel_size=self.window_size,
            padding=self.padding,
        ).reshape(batch_size, neighborhood_size, height, width).bool()
        weights = torch.softmax(scores.masked_fill(~valid, -torch.inf), dim=1)
        value_windows = F.unfold(
            previous_error,
            kernel_size=self.window_size,
            padding=self.padding,
        ).reshape(
            batch_size,
            UNIFIED_STATE_CHANNELS,
            neighborhood_size,
            height,
            width,
        )
        aligned = (weights.unsqueeze(1) * value_windows).sum(dim=2)
        if return_attention:
            return aligned, weights, valid
        return aligned
