import torch
from torch import nn


class DirectStateCorrection(nn.Module):
    """Map observation, instant error, and dynamic error to a state delta."""

    def __init__(self, channels=128):
        super().__init__()
        self.input_projection = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.output_projection = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )

    def forward(
        self,
        observation: torch.Tensor,
        error: torch.Tensor,
        dynamic_error: torch.Tensor,
    ) -> torch.Tensor:
        return self.output_projection(
            self.activation(
                self.input_projection(torch.cat((observation, error, dynamic_error), dim=1))
            )
        )
