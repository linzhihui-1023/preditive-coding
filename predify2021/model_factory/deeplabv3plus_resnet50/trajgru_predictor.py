import torch
from torch import nn
from torch.nn import functional as F

from .adapters import UNIFIED_STATE_CHANNELS


class TrajGRUCell(nn.Module):
    """Trajectory GRU with learned location-variant recurrent connections."""

    def __init__(self, input_channels, hidden_channels=128, links=5):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.links = links

        self.flow_generator = nn.Sequential(
            nn.Conv2d(input_channels + hidden_channels, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2 * links, 5, padding=2),
        )
        nn.init.zeros_(self.flow_generator[-1].weight)
        nn.init.zeros_(self.flow_generator[-1].bias)

        self.input_projection = nn.Conv2d(
            input_channels, 3 * hidden_channels, 3, padding=1
        )
        self.hidden_projections = nn.ModuleList(
            nn.Conv2d(hidden_channels, 3 * hidden_channels, 1)
            for _ in range(links)
        )

    @staticmethod
    def _warp(hidden, flow_x, flow_y):
        batch, _, height, width = hidden.shape
        yy, xx = torch.meshgrid(
            torch.linspace(
                -1.0, 1.0, height, device=hidden.device, dtype=hidden.dtype
            ),
            torch.linspace(
                -1.0, 1.0, width, device=hidden.device, dtype=hidden.dtype
            ),
            indexing="ij",
        )
        base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        base_grid = base_grid.expand(batch, -1, -1, -1)

        normalized_x = 2.0 * flow_x / max(width - 1, 1)
        normalized_y = 2.0 * flow_y / max(height - 1, 1)
        sampling_grid = base_grid + torch.stack(
            (normalized_x, normalized_y), dim=-1
        )

        return F.grid_sample(
            hidden,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(
                x.shape[0],
                self.hidden_channels,
                x.shape[2],
                x.shape[3],
                device=x.device,
                dtype=x.dtype,
            )

        flows = self.flow_generator(torch.cat((x, hidden), dim=1))
        flow_x, flow_y = flows.chunk(2, dim=1)

        x_update, x_reset, x_candidate = self.input_projection(x).chunk(3, dim=1)
        hidden_update = torch.zeros_like(x_update)
        hidden_reset = torch.zeros_like(x_reset)
        hidden_candidate = torch.zeros_like(x_candidate)

        for link, projection in enumerate(self.hidden_projections):
            warped_hidden = self._warp(
                hidden,
                flow_x[:, link],
                flow_y[:, link],
            )
            h_update, h_reset, h_candidate = projection(warped_hidden).chunk(
                3, dim=1
            )
            hidden_update = hidden_update + h_update
            hidden_reset = hidden_reset + h_reset
            hidden_candidate = hidden_candidate + h_candidate

        update = torch.sigmoid(x_update + hidden_update)
        reset = torch.sigmoid(x_reset + hidden_reset)
        candidate = torch.tanh(x_candidate + reset * hidden_candidate)

        return (1.0 - update) * candidate + update * hidden


class AuxiliaryTemporalTrajPredictor(nn.Module):
    """Stage-T temporal predictor using TrajGRU instead of ConvGRU."""

    def __init__(self, channels=UNIFIED_STATE_CHANNELS, hidden_channels=128, links=5):
        super().__init__()
        self.recurrent = TrajGRUCell(
            2 * channels,
            hidden_channels=hidden_channels,
            links=links,
        )
        self.delta = nn.Conv2d(hidden_channels, channels, 3, padding=1)

        # Keep the existing Stage-T epoch-zero contract exactly:
        # predicted next state starts as persistence.
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def predict_next(self, state, error, hidden=None):
        hidden = self.recurrent(torch.cat((state, error), dim=1), hidden)
        return state + self.delta(hidden), hidden
