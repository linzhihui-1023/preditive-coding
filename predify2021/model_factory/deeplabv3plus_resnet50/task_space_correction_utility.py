"""Correction-utility predictor for the current C-V2 task-space mainline.

U answers a different question from transportability T:

- T: can historical semantics be transported here?
- U: is the carried semantic correction worth writing back to the Host here?

U never gates transport correction and never controls semantic-state evolution.
It only gates the final writeback of the carried Semantic Correction State.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_recurrent_predictor import ConvGRUCell


class RecurrentCorrectionUtility(nn.Module):
    """Predict semantic-correction writeback utility from causal evidence."""

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

        # Evidence channels:
        # host probability, temporal-prior probability, absolute prediction error,
        # semantic-state candidate probability, transportability T, motion, appearance.
        input_channels = 4 * self.num_classes + 1 + 2 + self.projected_channels
        self.recurrent = ConvGRUCell(input_channels, self.hidden_channels)
        self.utility_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, 1, 1),
        )

        # At E0 semantic correction itself is zero-initialized, so U=0.5 still
        # preserves exact Host output. U learns only from its direct utility target.
        nn.init.zeros_(self.utility_head[-1].weight)
        nn.init.zeros_(self.utility_head[-1].bias)

    def forward(
        self,
        current_c1,
        host_logits_low,
        prior_logits_low,
        transport_motion_low,
        transportability_low,
        semantic_state_low,
        hidden=None,
    ):
        if host_logits_low.shape != prior_logits_low.shape:
            raise ValueError("Host and temporal-prior logits must share shape")
        if current_c1.shape[-2:] != host_logits_low.shape[-2:]:
            raise ValueError("C1 and task logits must share spatial size")
        if transport_motion_low.shape[-2:] != host_logits_low.shape[-2:]:
            raise ValueError("Motion and task logits must share spatial size")
        if semantic_state_low.shape != host_logits_low.shape:
            raise ValueError("Semantic state and Host logits must share shape")
        if transportability_low.shape[-2:] != host_logits_low.shape[-2:]:
            raise ValueError("Transportability and task logits must share spatial size")

        host_probability = F.softmax(host_logits_low, dim=1)
        prior_probability = F.softmax(prior_logits_low, dim=1)
        prediction_error_abs = (host_probability - prior_probability).abs()
        semantic_candidate_probability = F.softmax(
            host_logits_low + semantic_state_low, dim=1
        )
        normalized_motion = transport_motion_low / max(self.motion_scale, 1e-6)
        appearance = self.feature_projector(current_c1)

        recurrent_input = torch.cat(
            (
                host_probability,
                prior_probability,
                prediction_error_abs,
                semantic_candidate_probability,
                transportability_low,
                normalized_motion,
                appearance,
            ),
            dim=1,
        )
        hidden = self.recurrent(recurrent_input, hidden)
        utility_logit = self.utility_head(hidden)
        return utility_logit, hidden
