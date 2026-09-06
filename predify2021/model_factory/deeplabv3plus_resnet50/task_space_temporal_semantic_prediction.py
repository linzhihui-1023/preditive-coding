"""Temporal semantic prediction from a strictly historical C-V3 memory state.

The decoder receives only motion-transported previous semantic memory. It does
not receive current C1 appearance, Host logits, prediction error,
transportability, semantic correction state, or the current updated memory.

This makes its output a causal temporal semantic hypothesis rather than a
second current-frame segmentation head.
"""

from torch import nn


class TemporalSemanticDecoder(nn.Module):
    """Decode transported 64-D history into 19-class low-resolution logits."""

    def __init__(
        self,
        memory_channels=64,
        hidden_channels=64,
        bottleneck_channels=32,
        num_classes=19,
    ):
        super().__init__()
        self.memory_channels = int(memory_channels)
        self.hidden_channels = int(hidden_channels)
        self.bottleneck_channels = int(bottleneck_channels)
        self.num_classes = int(num_classes)

        hidden_groups = 8 if self.hidden_channels % 8 == 0 else 1
        bottleneck_groups = 8 if self.bottleneck_channels % 8 == 0 else 1
        self.decoder = nn.Sequential(
            nn.Conv2d(
                self.memory_channels,
                self.hidden_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(hidden_groups, self.hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                self.hidden_channels,
                self.bottleneck_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(bottleneck_groups, self.bottleneck_channels),
            nn.SiLU(),
            nn.Conv2d(self.bottleneck_channels, self.num_classes, 1),
        )

    def forward(self, predictive_memory):
        if predictive_memory.ndim != 4:
            raise ValueError("predictive_memory must be BCHW")
        if predictive_memory.shape[1] != self.memory_channels:
            raise ValueError(
                "Unexpected predictive-memory channels: "
                f"{predictive_memory.shape[1]}, expected {self.memory_channels}"
            )
        return self.decoder(predictive_memory)
