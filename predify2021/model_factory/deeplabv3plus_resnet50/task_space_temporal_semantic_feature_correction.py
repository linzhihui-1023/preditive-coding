"""C-V12 prediction-error-driven temporal semantic feature correction.

中文：C-V12 预测误差驱动的时序条件语义特征修正。

Research boundary / 研究边界
----------------------------
1. Raw History（原始历史）never enters the correction generator directly.
   History -> motion-aligned Prediction -> Prediction Error -> Correction.
2. K=4 signed task-space Prediction Errors（预测误差）provide semantic
   correction content: what should be changed.
3. Frozen C-V4 temporal hidden state plus explicit Dynamics Error（动力学误差）
   provide temporal conditioning: when / where / which feature channels should
   accept the semantic correction.
4. The correction target is the frozen Host c4 semantic feature (2048D), not
   the final 19D segmentation logits.
5. The module predicts only a residual Delta-c4. It never reconstructs c4.
6. A zero-initialized HostConditionedResidualWriteback（宿主条件残差回写）
   guarantees exact zero-step equivalence to the frozen current feature path.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .adapters import HostConditionedResidualWriteback


class TemporalSemanticFeatureCorrector(nn.Module):
    """K=4 Prediction Error -> semantic residual, C-V4 state -> temporal modulation."""

    def __init__(
        self,
        num_classes=19,
        history_length=4,
        semantic_channels=128,
        temporal_hidden_channels=32,
        host_channels=2048,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.history_length = int(history_length)
        self.semantic_channels = int(semantic_channels)
        self.temporal_hidden_channels = int(temporal_hidden_channels)
        self.host_channels = int(host_channels)

        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        if self.semantic_channels % 8 != 0:
            raise ValueError("semantic_channels must be divisible by 8")

        # Semantic content is strictly concat(e1..eK). Validity only masks each
        # error before concatenation; raw probabilities/logits never enter here.
        self.semantic_input_channels = self.history_length * self.num_classes
        groups = min(32, self.semantic_channels)
        while self.semantic_channels % groups != 0:
            groups -= 1
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

        # Temporal evidence: frozen C-V4 hidden (32D), explicit epsilon (19D),
        # T, Q, any-valid, valid-fraction, and cross-history sign agreement.
        self.temporal_input_channels = (
            self.temporal_hidden_channels + self.num_classes + 5
        )
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
        self.temporal_modulation_head = nn.Conv2d(
            self.semantic_channels,
            self.semantic_channels,
            kernel_size=1,
            bias=True,
        )
        # 2*sigmoid(0)=1: temporal branch starts as an exact neutral modulation.
        nn.init.zeros_(self.temporal_modulation_head.weight)
        nn.init.zeros_(self.temporal_modulation_head.bias)

        # Converts the 128D error-driven command into a host-conditioned c4
        # residual. Its output projection is zero initialized by construction.
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

        validity_stack = torch.stack(validities, dim=1)  # B,K,1,H,W
        error_stack = torch.stack(masked_errors, dim=1)  # B,K,C,H,W
        valid_count = validity_stack.sum(dim=1).clamp_min(1.0)
        any_valid = (validity_stack.sum(dim=1) > 0.0).to(reference.dtype)
        valid_fraction = validity_stack.mean(dim=1)

        signed = torch.sign(error_stack) * validity_stack
        sign_mean = signed.sum(dim=1) / valid_count
        sign_agreement = sign_mean.abs().mean(dim=1, keepdim=True)
        sign_agreement = sign_agreement * any_valid

        return {
            "proposal_input": torch.cat(masked_errors, dim=1),
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

        semantic_latent = self.semantic_encoder(aggregate["proposal_input"])
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
        temporal_gain = 2.0 * torch.sigmoid(
            self.temporal_modulation_head(temporal_latent)
        )

        target_size = tuple(current_c4.shape[-2:])
        semantic_latent_c4 = F.interpolate(
            semantic_latent,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        temporal_gain_c4 = F.interpolate(
            temporal_gain,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        correction_command = semantic_latent_c4 * temporal_gain_c4

        current_c4 = current_c4.detach()
        delta_c4 = self.writeback(current_c4, correction_command)
        corrected_c4 = current_c4 + delta_c4

        return {
            "corrected_c4": corrected_c4,
            "delta_c4": delta_c4,
            "semantic_latent": semantic_latent,
            "temporal_latent": temporal_latent,
            "temporal_gain": temporal_gain,
            "correction_command": correction_command,
            **aggregate,
        }
