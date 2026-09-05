"""Role-decoupled task-space correction components for C-V2.

These components sit on top of a frozen Stage-1B motion prior.  They never
modify the motion field.  Two correction roles are represented explicitly:

- transport correction: use history/motion evidence where semantics are
  transportable from the previous frame;
- semantic innovation: use the current observation where history cannot explain
  the current semantics (disocclusion/newly visible content, etc.).

A separate recurrent transportability predictor can route the two residuals at
inference.  It predicts a correction-routing variable, not a Host/Prior fusion
weight.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_recurrent_predictor import ConvGRUCell


class RoleSeparatedTaskCorrection(nn.Module):
    """Independent recurrent transport and semantic-innovation correction paths."""

    def __init__(
        self,
        c1_channels=256,
        num_classes=19,
        projected_channels=32,
        hidden_channels=64,
        motion_scale=32.0,
    ):
        super().__init__()
        self.c1_channels = int(c1_channels)
        self.num_classes = int(num_classes)
        self.projected_channels = int(projected_channels)
        self.hidden_channels = int(hidden_channels)
        self.motion_scale = float(motion_scale)

        groups = 8 if self.projected_channels % 8 == 0 else 1
        self.semantic_projector = nn.Sequential(
            nn.Conv2d(self.c1_channels, self.projected_channels, 1, bias=False),
            nn.GroupNorm(groups, self.projected_channels),
            nn.SiLU(),
        )

        # Transport correction deliberately has no direct C1 appearance path.
        # It must operate from Host/Prior disagreement and the frozen motion
        # estimate, which reduces the static-segmentation shortcut.
        transport_input_channels = 3 * self.num_classes + 2
        self.transport_recurrent = ConvGRUCell(
            transport_input_channels, self.hidden_channels
        )
        self.transport_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, self.num_classes, 1),
        )

        # Semantic innovation is the only path that receives current C1
        # appearance, because newly visible semantics cannot be synthesized from
        # a history-only warp.
        semantic_input_channels = 3 * self.num_classes + self.projected_channels
        self.semantic_recurrent = ConvGRUCell(
            semantic_input_channels, self.hidden_channels
        )
        self.semantic_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, self.num_classes, 1),
        )

        # E0 is exactly the frozen Host for both candidate outputs.
        nn.init.zeros_(self.transport_head[-1].weight)
        nn.init.zeros_(self.transport_head[-1].bias)
        nn.init.zeros_(self.semantic_head[-1].weight)
        nn.init.zeros_(self.semantic_head[-1].bias)

    def forward(
        self,
        current_c1,
        host_logits_low,
        prior_logits_low,
        transport_motion_low,
        transport_hidden=None,
        semantic_hidden=None,
    ):
        if host_logits_low.shape != prior_logits_low.shape:
            raise ValueError("Host and temporal-prior logits must share shape")
        if current_c1.shape[-2:] != host_logits_low.shape[-2:]:
            raise ValueError("C1 and task logits must share spatial size")
        if transport_motion_low.shape[-2:] != host_logits_low.shape[-2:]:
            raise ValueError("Motion and task logits must share spatial size")

        host_probability = F.softmax(host_logits_low, dim=1)
        prior_probability = F.softmax(prior_logits_low, dim=1)
        prediction_error = host_probability - prior_probability
        normalized_motion = transport_motion_low / max(self.motion_scale, 1e-6)

        transport_input = torch.cat(
            (
                host_probability,
                prior_probability,
                prediction_error,
                normalized_motion,
            ),
            dim=1,
        )
        transport_hidden = self.transport_recurrent(
            transport_input, transport_hidden
        )
        delta_transport = self.transport_head(transport_hidden)

        semantic_feature = self.semantic_projector(current_c1)
        semantic_input = torch.cat(
            (
                host_probability,
                prior_probability,
                prediction_error,
                semantic_feature,
            ),
            dim=1,
        )
        semantic_hidden = self.semantic_recurrent(semantic_input, semantic_hidden)
        delta_semantic = self.semantic_head(semantic_hidden)

        return {
            "delta_transport": delta_transport,
            "delta_semantic": delta_semantic,
            "prediction_error": prediction_error,
            "transport_hidden": transport_hidden,
            "semantic_hidden": semantic_hidden,
        }


class RecurrentTransportabilityMask(nn.Module):
    """Predict whether current semantics are transportable from history.

    Output T in [0,1]:
      T -> 1: route transport correction;
      T -> 0: route semantic-innovation correction.

    This is not a Host/Prior fusion gate.  The Host remains the additive anchor.
    """

    def __init__(
        self,
        c1_channels=256,
        num_classes=19,
        projected_channels=16,
        hidden_channels=32,
        motion_scale=32.0,
    ):
        super().__init__()
        self.c1_channels = int(c1_channels)
        self.num_classes = int(num_classes)
        self.projected_channels = int(projected_channels)
        self.hidden_channels = int(hidden_channels)
        self.motion_scale = float(motion_scale)

        groups = 8 if self.projected_channels % 8 == 0 else 1
        self.feature_projector = nn.Sequential(
            nn.Conv2d(self.c1_channels, self.projected_channels, 1, bias=False),
            nn.GroupNorm(groups, self.projected_channels),
            nn.SiLU(),
        )
        input_channels = 3 * self.num_classes + 2 + self.projected_channels
        self.recurrent = ConvGRUCell(input_channels, self.hidden_channels)
        self.mask_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, 1, 1),
        )
        # Neutral E0. The mask is trained by explicit role supervision rather
        # than by downstream CE, preventing the all-Host lazy shortcut.
        nn.init.zeros_(self.mask_head[-1].weight)
        nn.init.zeros_(self.mask_head[-1].bias)

    def forward(
        self,
        current_c1,
        host_logits_low,
        prior_logits_low,
        transport_motion_low,
        hidden=None,
    ):
        host_probability = F.softmax(host_logits_low, dim=1)
        prior_probability = F.softmax(prior_logits_low, dim=1)
        prediction_error = host_probability - prior_probability
        normalized_motion = transport_motion_low / max(self.motion_scale, 1e-6)
        appearance = self.feature_projector(current_c1)
        recurrent_input = torch.cat(
            (
                host_probability,
                prior_probability,
                prediction_error.abs(),
                normalized_motion,
                appearance,
            ),
            dim=1,
        )
        hidden = self.recurrent(recurrent_input, hidden)
        transportability_logit = self.mask_head(hidden)
        return transportability_logit, hidden
