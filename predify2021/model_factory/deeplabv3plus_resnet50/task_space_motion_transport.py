"""C-V2 causal motion transport primitives for task-space temporal priors."""

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_recurrent_predictor import ConvGRUCell


class CausalMotionTransportPredictor(nn.Module):
    """Predict next-frame backward transport from causal semantic history only."""

    def __init__(
        self,
        num_classes=19,
        hidden_channels=64,
        max_displacement_low=32.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)
        self.max_displacement_low = float(max_displacement_low)
        self.recurrent = ConvGRUCell(2 * self.num_classes, self.hidden_channels)
        self.flow_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, 2, 1),
        )
        # Epoch zero is exact zero-motion persistence regardless of hidden state.
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)

    def predict_next(self, host_logits_low, probability_error, hidden=None):
        host_probability = F.softmax(host_logits_low, dim=1)
        recurrent_input = torch.cat((host_probability, probability_error), dim=1)
        hidden = self.recurrent(recurrent_input, hidden)
        raw_flow = self.flow_head(hidden)
        backward_flow_low = self.max_displacement_low * torch.tanh(raw_flow)
        return backward_flow_low, hidden


def low_flow_grid(backward_flow_low):
    """Build a grid that samples the previous map at current-grid + backward flow."""
    batch, _, height, width = backward_flow_low.shape
    if batch != 1:
        raise ValueError("C-V2 motion transport currently expects batch size 1")
    y, x = torch.meshgrid(
        torch.arange(height, device=backward_flow_low.device, dtype=backward_flow_low.dtype),
        torch.arange(width, device=backward_flow_low.device, dtype=backward_flow_low.dtype),
        indexing="ij",
    )
    source_x = x.unsqueeze(0) + backward_flow_low[:, 0]
    source_y = y.unsqueeze(0) + backward_flow_low[:, 1]
    valid = (
        (source_x >= 0)
        & (source_x <= width - 1)
        & (source_y >= 0)
        & (source_y <= height - 1)
    )
    grid = torch.stack(
        (
            2.0 * source_x / max(width - 1, 1) - 1.0,
            2.0 * source_y / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    )
    return grid, valid


def warp_low_logits(previous_logits_low, backward_flow_low):
    """History-only warp; invalid transport falls back to historical persistence."""
    if previous_logits_low.shape[-2:] != backward_flow_low.shape[-2:]:
        raise ValueError("Logits and backward flow must share the same spatial size")
    grid, valid = low_flow_grid(backward_flow_low)
    warped = F.grid_sample(
        previous_logits_low.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    # Never use the current Host as an invalid-flow fallback: that would leak
    # target-frame information into the historical prior. Use same-coordinate
    # previous logits instead.
    warped = torch.where(valid.unsqueeze(1), warped, previous_logits_low.float())
    return warped.to(previous_logits_low.dtype), valid


def downsample_backward_flow(full_flow, low_size):
    """Convert full-resolution pixel flow to low-resolution align-corners units."""
    low_h, low_w = low_size
    full_h, full_w = full_flow.shape[-2:]
    low = F.interpolate(full_flow, size=low_size, mode="bilinear", align_corners=True).clone()
    # flow_grid/grid_sample both use align_corners=True, so one endpoint-to-endpoint
    # pixel displacement scales with (size - 1), not size.
    x_scale = (low_w - 1) / max(full_w - 1, 1)
    y_scale = (low_h - 1) / max(full_h - 1, 1)
    low[:, 0] *= x_scale
    low[:, 1] *= y_scale
    return low


def teacher_reachable_mask(teacher_flow, max_displacement):
    """Teacher locations that are spatially valid and reachable by the predictor."""
    _, spatial_valid = low_flow_grid(teacher_flow)
    scale = max(float(max_displacement), 0.0)
    reachable = (
        spatial_valid
        & (teacher_flow[:, 0].abs() <= scale)
        & (teacher_flow[:, 1].abs() <= scale)
    )
    return reachable, spatial_valid


def normalized_flow_distillation_loss(predicted_flow, teacher_flow, max_displacement):
    """Smooth-L1 RAFT distillation only on spatially valid reachable teacher flow."""
    reachable, _ = teacher_reachable_mask(teacher_flow, max_displacement)
    valid = reachable.unsqueeze(1).expand_as(predicted_flow)
    if not bool(valid.any()):
        return predicted_flow.sum() * 0.0
    scale = max(float(max_displacement), 1e-6)
    teacher_target = teacher_flow.detach()
    return F.smooth_l1_loss(
        predicted_flow[valid] / scale,
        teacher_target[valid] / scale,
    )
