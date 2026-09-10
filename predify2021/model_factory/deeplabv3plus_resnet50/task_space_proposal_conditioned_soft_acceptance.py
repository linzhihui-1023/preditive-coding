"""C-V15 proposal-conditioned soft acceptance for semantic feature correction.

中文：C-V15 修正提案条件化软接受。

C-V15 keeps the complete aligned C-V14 proposal generator frozen and changes
only the acceptance decision. The new acceptance path explicitly observes:
1. frozen temporal latent;
2. frozen semantic latent from K=4 Prediction Error;
3. normalized bounded semantic Delta-c4 proposal;
4. current Host c4.

The frozen aligned C-V14 reliability remains a prior. A new proposal-conditioned
residual logit is initialized to zero, so before training C-V15 reproduces the
loaded C-V14 gate and final correction exactly (up to numerical precision).
"""

import torch
from torch import nn
from torch.nn import functional as F

from .task_space_post_writeback_reliability_feature_correction_aligned import (
    AlignedPostWritebackReliabilityFeatureCorrector,
)


class ProposalConditionedSoftAcceptanceCorrector(
    AlignedPostWritebackReliabilityFeatureCorrector
):
    """Frozen C-V14 proposal + proposal-conditioned soft acceptance."""

    def __init__(
        self,
        num_classes=19,
        history_length=4,
        semantic_channels=128,
        temporal_hidden_channels=32,
        host_channels=2048,
        residual_scale=0.10,
        proposal_descriptor_channels=32,
        host_descriptor_channels=32,
        acceptance_channels=128,
    ):
        super().__init__(
            num_classes=num_classes,
            history_length=history_length,
            semantic_channels=semantic_channels,
            temporal_hidden_channels=temporal_hidden_channels,
            host_channels=host_channels,
            residual_scale=residual_scale,
        )
        self.proposal_descriptor_channels = int(proposal_descriptor_channels)
        self.host_descriptor_channels = int(host_descriptor_channels)
        self.acceptance_channels = int(acceptance_channels)

        if self.proposal_descriptor_channels <= 0 or self.host_descriptor_channels <= 0:
            raise ValueError("descriptor channels must be positive")
        if self.acceptance_channels <= 0:
            raise ValueError("acceptance_channels must be positive")

        proposal_groups = min(8, self.proposal_descriptor_channels)
        while self.proposal_descriptor_channels % proposal_groups != 0:
            proposal_groups -= 1
        host_groups = min(8, self.host_descriptor_channels)
        while self.host_descriptor_channels % host_groups != 0:
            host_groups -= 1
        acceptance_groups = min(32, self.acceptance_channels)
        while self.acceptance_channels % acceptance_groups != 0:
            acceptance_groups -= 1

        self.proposal_encoder = nn.Sequential(
            nn.Conv2d(
                self.host_channels,
                self.proposal_descriptor_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(proposal_groups, self.proposal_descriptor_channels),
            nn.SiLU(),
        )
        self.host_encoder = nn.Sequential(
            nn.Conv2d(
                self.host_channels,
                self.host_descriptor_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(host_groups, self.host_descriptor_channels),
            nn.SiLU(),
        )

        fusion_channels = (
            self.semantic_channels
            + self.semantic_channels
            + self.proposal_descriptor_channels
            + self.host_descriptor_channels
        )
        self.acceptance_fusion = nn.Sequential(
            nn.Conv2d(
                fusion_channels,
                self.acceptance_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(acceptance_groups, self.acceptance_channels),
            nn.SiLU(),
        )
        self.acceptance_residual_head = nn.Conv2d(
            self.acceptance_channels,
            1,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.acceptance_residual_head.weight)
        nn.init.zeros_(self.acceptance_residual_head.bias)

        self._cv14_core_frozen = False

    def load_frozen_cv14_state_dict(self, state_dict):
        """Load aligned C-V14 E3 weights and freeze the inherited C-V14 core."""
        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = (
            "proposal_encoder.",
            "host_encoder.",
            "acceptance_fusion.",
            "acceptance_residual_head.",
        )
        unexpected = tuple(incompatible.unexpected_keys)
        illegal_missing = tuple(
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        )
        if unexpected:
            raise RuntimeError(f"Unexpected C-V14 checkpoint keys: {unexpected}")
        if illegal_missing:
            raise RuntimeError(f"Missing inherited C-V14 checkpoint keys: {illegal_missing}")
        self.freeze_cv14_core()
        return incompatible

    def freeze_cv14_core(self):
        """Freeze the entire inherited C-V14 proposal/temporal/reliability core."""
        for module in (
            self.semantic_encoder,
            self.temporal_pre,
            self.reliability_head,
            self.writeback,
        ):
            module.requires_grad_(False)
            module.eval()
        self._cv14_core_frozen = True
        return self

    def train(self, mode=True):
        super().train(mode)
        if self._cv14_core_frozen:
            for module in (
                self.semantic_encoder,
                self.temporal_pre,
                self.reliability_head,
                self.writeback,
            ):
                module.eval()
        return self

    def _proposal_conditioned_residual_logit(
        self,
        semantic_latent,
        temporal_latent,
        bounded_semantic_delta_c4,
        c4_channel_rms,
        current_c4,
    ):
        target_size = tuple(current_c4.shape[-2:])
        semantic_c4 = F.interpolate(
            semantic_latent.detach(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        temporal_c4 = F.interpolate(
            temporal_latent.detach(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        proposal_denominator = (
            self.residual_scale
            * c4_channel_rms.detach().clamp_min(1e-6)
        )
        normalized_proposal = (
            bounded_semantic_delta_c4.detach() / proposal_denominator
        ).clamp(-1.0, 1.0)
        proposal_descriptor = self.proposal_encoder(normalized_proposal)
        host_descriptor = self.host_encoder(current_c4.detach())

        acceptance_input = torch.cat(
            (
                temporal_c4,
                semantic_c4,
                proposal_descriptor,
                host_descriptor,
            ),
            dim=1,
        )
        acceptance_latent = self.acceptance_fusion(acceptance_input)
        residual_logit = self.acceptance_residual_head(acceptance_latent)
        return {
            "semantic_c4_for_acceptance": semantic_c4,
            "temporal_c4_for_acceptance": temporal_c4,
            "normalized_proposal": normalized_proposal,
            "proposal_descriptor": proposal_descriptor,
            "host_descriptor": host_descriptor,
            "acceptance_latent": acceptance_latent,
            "acceptance_residual_logit": residual_logit,
        }

    def forward(self, *args, **kwargs):
        current_c4 = kwargs.get("current_c4")
        if current_c4 is None:
            raise ValueError("C-V15 requires current_c4 as a keyword argument")

        row = super().forward(*args, **kwargs)
        base_reliability_c4 = row["reliability"].detach()
        base_reliability_low = row["reliability_low"].detach()

        acceptance = self._proposal_conditioned_residual_logit(
            semantic_latent=row["semantic_latent"],
            temporal_latent=row["temporal_latent"],
            bounded_semantic_delta_c4=row["bounded_semantic_delta_c4"],
            c4_channel_rms=row["c4_channel_rms"],
            current_c4=current_c4,
        )

        eps = 1e-6
        base_logit_c4 = torch.logit(
            base_reliability_c4.clamp(eps, 1.0 - eps)
        )
        reliability_logit_c4 = (
            base_logit_c4 + acceptance["acceptance_residual_logit"]
        )
        reliability_c4 = torch.sigmoid(reliability_logit_c4)

        current_c4 = current_c4.detach()
        delta_c4 = row["bounded_semantic_delta_c4"] * reliability_c4
        corrected_c4 = current_c4 + delta_c4

        row.update(
            {
                "base_reliability_low": base_reliability_low,
                "base_reliability_c4": base_reliability_c4,
                "base_reliability_logit_c4": base_logit_c4,
                "reliability_logit": reliability_logit_c4,
                "reliability_c4": reliability_c4,
                "reliability": reliability_c4,
                "delta_c4": delta_c4,
                "corrected_c4": corrected_c4,
                **acceptance,
            }
        )
        return row
