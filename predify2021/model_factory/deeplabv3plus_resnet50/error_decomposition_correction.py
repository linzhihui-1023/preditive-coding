import torch
from torch import nn

from .direct_state_correction import DirectStateCorrection


class ErrorInterpreter(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.input_projection = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, observation, error):
        return self.output_projection(
            self.activation(self.input_projection(torch.cat((observation, error), dim=1)))
        )


class ReliabilityGate(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.input_projection = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, estimated_corruption, estimated_mismatch, dynamic_corruption):
        return torch.sigmoid(
            self.output_projection(
                self.activation(
                    self.input_projection(
                        torch.cat(
                            (
                                estimated_corruption.abs(),
                                estimated_mismatch.abs(),
                                dynamic_corruption.abs(),
                            ),
                            dim=1,
                        )
                    )
                )
            )
        )


class ErrorDecompositionCorrection(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.interpreter = ErrorInterpreter(channels)
        self.reliability = ReliabilityGate(channels)
        self.direct = DirectStateCorrection(channels)

    def forward(self, observation, error, dynamic_corruption):
        estimated_corruption = self.interpreter(observation, error)
        estimated_mismatch = error - estimated_corruption
        reliability = self.reliability(
            estimated_corruption, estimated_mismatch, dynamic_corruption
        )
        direct_delta = self.direct(
            observation, estimated_corruption, dynamic_corruption
        )
        return (
            estimated_corruption,
            estimated_mismatch,
            reliability,
            reliability * direct_delta,
        )
