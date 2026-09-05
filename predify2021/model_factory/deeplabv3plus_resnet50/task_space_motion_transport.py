"""Task-space motion transport components for C-V2.

Stage 1 keeps the original causal semantic-history flow predictor. Stage 1B adds
an explicit pairwise Motion Observer and a residual future-motion predictor.
RAFT remains training supervision only; no class below depends on RAFT.
"""

import math

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
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)

    def predict_next(self, host_logits_low, probability_error, hidden=None):
        host_probability = F.softmax(host_logits_low, dim=1)
        recurrent_input = torch.cat((host_probability, probability_error), dim=1)
        hidden = self.recurrent(recurrent_input, hidden)
        raw_flow = self.flow_head(hidden)
        backward_flow_low = self.max_displacement_low * torch.tanh(raw_flow)
        return backward_flow_low, hidden


class LocalCorrelationMotionObserver(nn.Module):
    """Observe current backward motion from a frozen Host feature/probability pair.

    The observer estimates M_t = F_{t->t-1}. A local cost volume performs the
    explicit correspondence search that the original Stage-1 predictor lacked.
    """

    def __init__(
        self,
        c1_channels=256,
        num_classes=19,
        projected_channels=32,
        hidden_channels=64,
        correlation_radius=4,
        max_displacement_low=32.0,
    ):
        super().__init__()
        self.c1_channels = int(c1_channels)
        self.num_classes = int(num_classes)
        self.projected_channels = int(projected_channels)
        self.hidden_channels = int(hidden_channels)
        self.correlation_radius = int(correlation_radius)
        self.max_displacement_low = float(max_displacement_low)
        if self.correlation_radius < 0:
            raise ValueError("correlation_radius must be >= 0")

        groups = 8 if self.projected_channels % 8 == 0 else 1
        self.feature_projector = nn.Sequential(
            nn.Conv2d(self.c1_channels, self.projected_channels, 1, bias=False),
            nn.GroupNorm(groups, self.projected_channels),
            nn.SiLU(),
        )
        correlation_channels = (2 * self.correlation_radius + 1) ** 2
        input_channels = correlation_channels + 3 * self.num_classes
        self.flow_head = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, 2, 1),
        )
        # E0 is exact zero-motion persistence. The observer must earn every warp.
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)

    def _local_correlation(self, previous_c1, current_c1):
        previous = F.normalize(self.feature_projector(previous_c1), dim=1)
        current = F.normalize(self.feature_projector(current_c1), dim=1)
        batch, channels, height, width = current.shape
        radius = self.correlation_radius
        kernel = 2 * radius + 1
        patches = F.unfold(previous, kernel_size=kernel, padding=radius)
        patches = patches.view(batch, channels, kernel * kernel, height, width)
        correlation = (patches * current.unsqueeze(2)).sum(dim=1)
        correlation = correlation / math.sqrt(max(channels, 1))
        return correlation

    def forward(
        self,
        previous_c1,
        current_c1,
        previous_probability,
        current_probability,
    ):
        if previous_c1.shape[-2:] != current_c1.shape[-2:]:
            raise ValueError("Previous/current C1 features must share spatial size")
        if previous_probability.shape[-2:] != current_c1.shape[-2:]:
            raise ValueError("Host probabilities must be at C1 spatial resolution")
        correlation = self._local_correlation(previous_c1, current_c1)
        probability_delta = current_probability - previous_probability
        evidence = torch.cat(
            (correlation, previous_probability, current_probability, probability_delta),
            dim=1,
        )
        raw_flow = self.flow_head(evidence)
        return self.max_displacement_low * torch.tanh(raw_flow)


class MotionResidualPredictor(nn.Module):
    """Predict future motion as observed motion plus a bounded residual.

    Input at time t is the already observed M_t = F_{t->t-1} plus the semantic
    prediction error e_t. The output predicts M_{t+1} = F_{t+1->t}.
    """

    def __init__(
        self,
        num_classes=19,
        hidden_channels=64,
        max_observed_displacement_low=32.0,
        max_residual_displacement_low=16.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)
        self.max_observed_displacement_low = float(max_observed_displacement_low)
        self.max_residual_displacement_low = float(max_residual_displacement_low)
        self.recurrent = ConvGRUCell(2 + self.num_classes, self.hidden_channels)
        self.delta_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, 2, 1),
        )
        # E0 is exactly Lagged-Motion-Persistence: M_hat_{t+1} = M_t.
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def predict_next(self, observed_motion, probability_error, hidden=None):
        scale = max(self.max_observed_displacement_low, 1e-6)
        normalized_motion = observed_motion / scale
        recurrent_input = torch.cat((normalized_motion, probability_error), dim=1)
        hidden = self.recurrent(recurrent_input, hidden)
        raw_delta = self.delta_head(hidden)
        delta_motion = self.max_residual_displacement_low * torch.tanh(raw_delta)
        predicted_motion = observed_motion + delta_motion
        return predicted_motion, delta_motion, hidden


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
    warped = torch.where(valid.unsqueeze(1), warped, previous_logits_low.float())
    return warped.to(previous_logits_low.dtype), valid


def downsample_backward_flow(full_flow, low_size):
    """Convert full-resolution pixel flow to low-resolution align-corners units."""
    low_h, low_w = low_size
    full_h, full_w = full_flow.shape[-2:]
    low = F.interpolate(full_flow, size=low_size, mode="bilinear", align_corners=True).clone()
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


def normalized_motion_residual_l2(delta_motion, max_residual_displacement):
    """Dimensionless weak L2 regularizer for residual future motion."""
    scale = max(float(max_residual_displacement), 1e-6)
    return (delta_motion / scale).square().mean()
