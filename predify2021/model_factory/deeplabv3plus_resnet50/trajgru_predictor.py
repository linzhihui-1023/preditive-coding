import torch
from torch import nn
from torch.nn import functional as F

from .adapters import UNIFIED_STATE_CHANNELS


class TrajGRUCell(nn.Module):
    """Trajectory GRU with learned location-variant recurrent connections.

    This experimental cell keeps the current ConvGRU gate convention and
    candidate nonlinearity unchanged.  The controlled variable is only the
    recurrent spatial connection: fixed 3x3 convolution is replaced by learned
    trajectory sampling followed by per-link 1x1 projections.
    """

    def __init__(self, input_channels, hidden_channels=128, links=5):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.links = links

        self.flow_generator = nn.Sequential(
            nn.Conv2d(input_channels + hidden_channels, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2 * links, 5, padding=2),
        )
        # Epoch zero starts from zero displacement for every trajectory link.
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
    def _base_grid(hidden):
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
        return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(
            batch, -1, -1, -1
        )

    @staticmethod
    def _warp(hidden, flow_x, flow_y, base_grid=None):
        _, _, height, width = hidden.shape
        if base_grid is None:
            base_grid = TrajGRUCell._base_grid(hidden)

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
        base_grid = self._base_grid(hidden)

        x_update, x_reset, x_candidate = self.input_projection(x).chunk(3, dim=1)
        hidden_update = torch.zeros_like(x_update)
        hidden_reset = torch.zeros_like(x_reset)
        hidden_candidate = torch.zeros_like(x_candidate)

        for link, projection in enumerate(self.hidden_projections):
            warped_hidden = self._warp(
                hidden,
                flow_x[:, link],
                flow_y[:, link],
                base_grid,
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

        # Match the existing ConvGRU convention exactly.  Therefore the only
        # intended architectural change in this experiment is spatial transport.
        return (1.0 - update) * hidden + update * candidate


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

        # Keep the existing Stage-T epoch-zero persistence contract exactly.
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def predict_next(self, state, error, hidden=None):
        hidden = self.recurrent(torch.cat((state, error), dim=1), hidden)
        return state + self.delta(hidden), hidden


def trajgru_sanity_checks(device):
    """Cheap pre-training checks for transport identity, direction and gradient."""
    dtype = torch.float32

    # 1) Zero-flow identity: warp(H, 0, 0) == H.
    hidden = torch.randn(1, 4, 8, 10, device=device, dtype=dtype)
    zeros = torch.zeros(1, 8, 10, device=device, dtype=dtype)
    base_grid = TrajGRUCell._base_grid(hidden)
    warped_zero = TrajGRUCell._warp(hidden, zeros, zeros, base_grid)
    identity_max_abs = float((warped_zero - hidden).abs().max().item())
    if identity_max_abs > 1e-4:
        raise RuntimeError(
            f"TrajGRU zero-flow identity failed: max_abs={identity_max_abs:.6g}"
        )

    # 2) A one-pixel displacement must change a non-constant feature map.
    impulse = torch.zeros(1, 1, 8, 10, device=device, dtype=dtype)
    impulse[:, :, 3, 4] = 1.0
    impulse_grid = TrajGRUCell._base_grid(impulse)
    plus_one_x = torch.ones(1, 8, 10, device=device, dtype=dtype)
    shifted = TrajGRUCell._warp(
        impulse,
        plus_one_x,
        torch.zeros_like(plus_one_x),
        impulse_grid,
    )
    displacement_change = float((shifted - impulse).abs().sum().item())
    if displacement_change <= 0.5:
        raise RuntimeError(
            "TrajGRU non-zero displacement check failed: warp did not move features"
        )

    # 3) The learned flow generator must receive gradient through grid_sample.
    cell = TrajGRUCell(input_channels=4, hidden_channels=4, links=2).to(
        device=device, dtype=dtype
    )
    x = torch.randn(1, 4, 8, 10, device=device, dtype=dtype)
    recurrent_hidden = torch.randn(1, 4, 8, 10, device=device, dtype=dtype)
    output = cell(x, recurrent_hidden)
    spatial_weight = torch.linspace(
        -1.0, 1.0, 10, device=device, dtype=dtype
    ).view(1, 1, 1, 10)
    loss = (output * spatial_weight).mean()
    loss.backward()
    flow_grad = cell.flow_generator[-1].weight.grad
    flow_grad_norm = 0.0 if flow_grad is None else float(flow_grad.norm().item())
    if not torch.isfinite(torch.tensor(flow_grad_norm)) or flow_grad_norm <= 0.0:
        raise RuntimeError(
            f"TrajGRU flow-gradient check failed: grad_norm={flow_grad_norm:.6g}"
        )

    return {
        "zero_flow_identity_max_abs": identity_max_abs,
        "one_pixel_displacement_change": displacement_change,
        "flow_generator_grad_norm": flow_grad_norm,
    }
