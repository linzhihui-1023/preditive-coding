"""C-V14 post-writeback reliability-controlled semantic feature correction.

中文：C-V14 后回写可靠性控制的语义特征修正。

Research boundary / 研究边界
----------------------------
1. K=4 signed Prediction Errors（预测误差）provide semantic correction content.
2. Semantic proposal is generated independently of temporal reliability.
3. Host-conditioned Writeback first produces a bounded semantic Delta-c4 proposal.
4. Frozen C-V4 temporal hidden + Dynamics Error（动力学误差）+ reliability evidence
   produce a single-channel pixel-wise acceptance probability g_t in [0,1].
5. Acceptance is applied AFTER Writeback and AFTER the 0.10 feature-relative bound:
      final_Delta-c4 = g_t * semantic_Delta-c4_proposal.
   Therefore g_t=0 strictly removes the deployed correction and downstream trainable
   layers cannot compensate for the suppression.
6. Raw History（原始历史）never directly enters correction generation.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .adapters import HostConditionedResidualWriteback


class PostWritebackReliabilitySemanticFeatureCorrector(nn.Module):
    """Prediction Error -> bounded c4 proposal; temporal evidence -> post-writeback acceptance."""

    def __init__(
        self,
        num_classes=19,
        history_length=4,
        semantic_channels=128,
        temporal_hidden_channels=32,
        host_channels=2048,
        residual_scale=0.10,
        acceptance_init=0.95,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.history_length = int(history_length)
        self.semantic_channels = int(semantic_channels)
        self.temporal_hidden_channels = int(temporal_hidden_channels)
        self.host_channels = int(host_channels)
        self.residual_scale = float(residual_scale)
        self.acceptance_init = float(acceptance_init)

        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        if self.semantic_channels % 8 != 0:
            raise ValueError("semantic_channels must be divisible by 8")
        if not (0.0 < self.residual_scale <= 1.0):
            raise ValueError("residual_scale must be in (0,1]")
        if not (0.0 < self.acceptance_init < 1.0):
            raise ValueError("acceptance_init must be in (0,1)")

        self.semantic_input_channels = self.history_length * self.num_classes
        groups = min(32, self.semantic_channels)
        while self.semantic_channels % groups != 0:
            groups -= 1

        # Structurally identical to C-V13: concat(e1..e4) -> 128D semantic command.
        self.semantic_encoder = nn.Sequential(
            nn.Conv2d(
                self.semantic_input_channels,
                self.semantic_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, self.semantic_channels),
            nn.SiLU(),
            nn.Conv2d(
                self.semantic_channels,
                self.semantic_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, self.semantic_channels),
            nn.SiLU(),
        )

        # Structurally identical to C-V13 up to the final head.
        # Inputs: frozen C-V4 H_t (32D), epsilon_t (19D), T, Q,
        # any-valid, valid-fraction, cross-history sign agreement.
        self.temporal_input_channels = self.temporal_hidden_channels + self.num_classes + 5
        self.temporal_pre = nn.Sequential(
            nn.Conv2d(
                self.temporal_input_channels,
                self.semantic_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, self.semantic_channels),
            nn.SiLU(),
        )

        # C-V14 replaces C-V13's 128D pre-writeback reliability with one
        # single-channel pixel-wise post-writeback acceptance probability.
        self.acceptance_head = nn.Conv2d(
            self.semantic_channels,
            1,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.acceptance_head.weight)
        # Start near pass-through rather than mechanically halving a transferred
        # C-V13 proposal. C-V13 E3 reliability was already near one; 0.95 keeps
        # the structural comparison clean while leaving useful sigmoid gradient.
        acceptance_bias = math.log(self.acceptance_init / (1.0 - self.acceptance_init))
        nn.init.constant_(self.acceptance_head.bias, acceptance_bias)

        # Same Host-conditioned semantic writeback structure as C-V13.
        self.writeback = HostConditionedResidualWriteback(self.host_channels)

    def _aggregate_prediction_errors(self, prediction_errors, history_validities_low):
        if len(prediction_errors) != self.history_length:
            raise ValueError("prediction_errors length must equal history_length")
        if len(history_validities_low) != self.history_length:
            raise ValueError("history_validities_low length must equal history_length")

        reference = prediction_errors[0]
        if reference.ndim != 4 or reference.shape[1] != self.num_classes:
            raise ValueError("prediction error must be BCHW with num_classes channels")
        spatial = reference.shape[-2:]

        masked_errors = []
        validities = []
        for error, validity in zip(prediction_errors, history_validities_low):
            if error.shape != reference.shape:
                raise ValueError("all prediction errors must share shape")
            if validity.ndim != 4 or validity.shape[1] != 1 or validity.shape[-2:] != spatial:
                raise ValueError("history validity must be Bx1xHxW at error resolution")
            validity = validity.to(error.dtype).clamp(0.0, 1.0)
            validities.append(validity)
            masked_errors.append(error * validity)

        validity_stack = torch.stack(validities, dim=1)
        error_stack = torch.stack(masked_errors, dim=1)
        valid_count = validity_stack.sum(dim=1).clamp_min(1.0)
        any_valid = (validity_stack.sum(dim=1) > 0.0).to(reference.dtype)
        valid_fraction = validity_stack.mean(dim=1)
        signed = torch.sign(error_stack) * validity_stack
        sign_mean = signed.sum(dim=1) / valid_count
        sign_agreement = sign_mean.abs().mean(dim=1, keepdim=True) * any_valid
        return {
            "semantic_input": torch.cat(masked_errors, dim=1),
            "any_history_valid": any_valid,
            "valid_fraction": valid_fraction,
            "sign_agreement": sign_agreement,
        }

    def forward(
        self,
        prediction_errors,
        history_validities_low,
        temporal_hidden,
        dynamics_error,
        transportability_low,
        memory_reliability_low,
        current_c4,
    ):
        aggregate = self._aggregate_prediction_errors(
            prediction_errors,
            history_validities_low,
        )
        spatial = aggregate["any_history_valid"].shape[-2:]

        if temporal_hidden.ndim != 4 or temporal_hidden.shape[1] != self.temporal_hidden_channels:
            raise ValueError("temporal_hidden must be BCHW with temporal_hidden_channels")
        if temporal_hidden.shape[-2:] != spatial:
            raise ValueError("temporal_hidden spatial size mismatch")
        if dynamics_error.ndim != 4 or dynamics_error.shape[1] != self.num_classes:
            raise ValueError("dynamics_error must be BCHW with num_classes channels")
        if dynamics_error.shape[-2:] != spatial:
            raise ValueError("dynamics_error spatial size mismatch")
        for name, tensor in (
            ("transportability_low", transportability_low),
            ("memory_reliability_low", memory_reliability_low),
        ):
            if tensor.ndim != 4 or tensor.shape[1] != 1 or tensor.shape[-2:] != spatial:
                raise ValueError(f"{name} must be Bx1xHxW at error resolution")
        if current_c4.ndim != 4 or current_c4.shape[1] != self.host_channels:
            raise ValueError("current_c4 must be BCHW with host_channels")

        # Semantic proposal is independent of temporal acceptance.
        semantic_latent = self.semantic_encoder(aggregate["semantic_input"])
        target_size = tuple(current_c4.shape[-2:])
        semantic_latent_c4 = F.interpolate(
            semantic_latent,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        current_c4 = current_c4.detach()
        raw_proposal_delta_c4 = self.writeback(current_c4, semantic_latent_c4)
        c4_channel_rms = (
            current_c4.square()
            .mean(dim=(-2, -1), keepdim=True)
            .sqrt()
            .clamp_min(1e-6)
        )
        proposal_delta_c4 = (
            self.residual_scale
            * c4_channel_rms
            * torch.tanh(raw_proposal_delta_c4)
        )
        proposal_corrected_c4 = current_c4 + proposal_delta_c4

        temporal_evidence = torch.cat(
            (
                temporal_hidden.detach(),
                dynamics_error.detach(),
                transportability_low.detach().clamp(0.0, 1.0),
                memory_reliability_low.detach().clamp(0.0, 1.0),
                aggregate["any_history_valid"],
                aggregate["valid_fraction"],
                aggregate["sign_agreement"],
            ),
            dim=1,
        )
        temporal_latent = self.temporal_pre(temporal_evidence)
        acceptance_logit = self.acceptance_head(temporal_latent)
        acceptance = torch.sigmoid(acceptance_logit)
        acceptance_c4 = F.interpolate(
            acceptance,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        # Hard causal control: no trainable layer follows this multiplication.
        final_delta_c4 = proposal_delta_c4 * acceptance_c4
        corrected_c4 = current_c4 + final_delta_c4

        # Same numerical head with detached temporal context: auxiliary BCE can
        # update only Acceptance Head; final CE still trains the deployed path.
        acceptance_logit_aux = self.acceptance_head(temporal_latent.detach())

        return {
            "corrected_c4": corrected_c4,
            "final_delta_c4": final_delta_c4,
            "proposal_corrected_c4": proposal_corrected_c4,
            "proposal_delta_c4": proposal_delta_c4,
            "raw_proposal_delta_c4": raw_proposal_delta_c4,
            "c4_channel_rms": c4_channel_rms,
            "semantic_latent": semantic_latent,
            "temporal_latent": temporal_latent,
            "acceptance_logit": acceptance_logit,
            "acceptance_logit_aux": acceptance_logit_aux,
            "acceptance": acceptance,
            "acceptance_c4": acceptance_c4,
            **aggregate,
        }
