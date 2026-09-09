"""C-V11 reliability-conditioned predictive-error expansion.

中文：C-V11 可靠性条件幅度扩张预测误差修正器。

Research boundary / 研究边界
----------------------------
1. Preserve the predictive-coding causal path:
   History -> Prediction -> Prediction Error -> Proposal -> Correction.
2. Reuse the exact C-V9 Proposal Head（提议头）, Error Memory（误差记忆）, 95D->32D
   Control Pre（控制特征提取器）and Acceptance Head（接受头）.
3. Do not use a free Amplitude Head. Add one 32D->1 Expansion Reliability Head
   （幅度扩张可靠性头）that controls only the correction beyond the validated
   C-V9 base gain 0.25 * Acceptance.
4. Deployed gain:
      base_alpha = 0.25 * a
      expansion  = (1 - base_alpha) * a * r
      alpha      = base_alpha + expansion
   where a is Acceptance and r is Expansion Reliability. Therefore alpha is in
   [0,1], r=0 exactly recovers the C-V9 gain, and large expansion requires both
   high Acceptance and high Reliability.
5. The Reliability Head shares the validated C-V9 95D->32D Control Pre. Relative
   to C-V9 it adds only one 32D->1 head (33 trainable parameters).
6. tanh bounded semantic proposal（双曲正切有界语义提议）is retained.
"""

import math

import torch
from torch import nn

from .task_space_direct_error_proposal_corrector import DirectErrorProposalCorrector


class ReliabilityConditionedExpansionCorrector(DirectErrorProposalCorrector):
    """C-V9 safety path plus reliability-conditioned gain expansion."""

    def __init__(
        self,
        num_classes=19,
        history_length=4,
        hidden_channels=32,
        acceptance_bias=-2.0,
        base_gain=0.25,
        reliability_init=1e-3,
    ):
        # g_max=1.0 is evaluator compatibility metadata only. The forward path
        # below explicitly forms alpha and never multiplies by a fixed g_max.
        super().__init__(
            num_classes=num_classes,
            history_length=history_length,
            hidden_channels=hidden_channels,
            g_max=1.0,
            gate_bias=acceptance_bias,
        )
        self.acceptance_bias = float(acceptance_bias)
        self.base_gain = float(base_gain)
        self.reliability_init = float(reliability_init)
        if not 0.0 < self.base_gain < 1.0:
            raise ValueError("base_gain must be in (0,1)")
        if not 0.0 < self.reliability_init < 1.0:
            raise ValueError("reliability_init must be in (0,1)")
        self.reliability_bias = math.log(
            self.reliability_init / (1.0 - self.reliability_init)
        )
        self.decision_channels = (
            self.hidden_channels + 2 * self.num_classes + 6 + self.num_classes
        )

        # Reuse inherited gate_pre as shared Control Pre and inherited gate_head
        # as Acceptance Head. C-V11 adds only this 32D->1 Reliability Head.
        self.expansion_reliability_head = nn.Conv2d(
            self.hidden_channels,
            1,
            1,
            bias=True,
        )
        nn.init.zeros_(self.expansion_reliability_head.weight)
        nn.init.constant_(
            self.expansion_reliability_head.bias,
            self.reliability_bias,
        )

    @property
    def control_pre(self):
        return self.gate_pre

    @property
    def acceptance_head(self):
        return self.gate_head

    def forward(
        self,
        prediction_errors,
        dynamics_error,
        current_margin,
        history_margins,
        transportability_low,
        memory_reliability_low,
        history_validities_low,
        backward_motion_low,
        hidden=None,
    ):
        aggregate = self._aggregate_error_evidence(
            prediction_errors,
            history_validities_low,
        )
        spatial = aggregate["any_history_valid"].shape[-2:]

        if dynamics_error.ndim != 4 or dynamics_error.shape[1] != self.num_classes:
            raise ValueError("dynamics_error must be BCHW with num_classes channels")
        if dynamics_error.shape[-2:] != spatial:
            raise ValueError("dynamics_error spatial size mismatch")
        if len(history_margins) != self.history_length:
            raise ValueError("history_margins length must equal history_length")
        for margin in history_margins:
            if margin.ndim != 4 or margin.shape[1] != 1 or margin.shape[-2:] != spatial:
                raise ValueError("history margin must be Bx1xHxW")
        for name, tensor in (
            ("current_margin", current_margin),
            ("transportability_low", transportability_low),
            ("memory_reliability_low", memory_reliability_low),
        ):
            if tensor.ndim != 4 or tensor.shape[1] != 1 or tensor.shape[-2:] != spatial:
                raise ValueError(f"{name} must be Bx1xHxW")
        if backward_motion_low.ndim != 4 or backward_motion_low.shape[1] != 2:
            raise ValueError("backward_motion_low must be Bx2xHxW")
        if backward_motion_low.shape[-2:] != spatial:
            raise ValueError("backward_motion_low spatial size mismatch")

        # Temporal Error Memory: exact C-V9 path.
        encoded = self.error_pre(aggregate["evidence"])
        if hidden is None:
            warped_hidden = torch.zeros_like(encoded)
            hidden_valid = torch.zeros_like(aggregate["any_history_valid"])
        else:
            warped_hidden, valid = self._warp_hidden_zero_invalid(
                hidden,
                backward_motion_low,
            )
            hidden_valid = valid.unsqueeze(1).to(encoded.dtype)

        error_reliability = (
            memory_reliability_low.detach().clamp(0.0, 1.0)
            * aggregate["any_history_valid"]
            * hidden_valid
        )
        gated_history = error_reliability * warped_hidden
        hidden = self.error_recurrent(encoded, gated_history)

        # Prediction Error alone determines semantic correction content.
        delta_z_raw = self.proposal_head(aggregate["proposal_input"])
        delta_z = torch.tanh(delta_z_raw)

        # Exact C-V9 decision evidence and shared Control Pre.
        decision_evidence = self._build_gate_evidence(
            hidden,
            dynamics_error,
            current_margin,
            history_margins,
            transportability_low,
            memory_reliability_low,
            aggregate,
            delta_z_raw,
        )
        control_hidden = self.gate_pre(decision_evidence)
        acceptance_logit = self.gate_head(control_hidden)
        reliability_logit = self.expansion_reliability_head(control_hidden)
        acceptance = torch.sigmoid(acceptance_logit)
        expansion_reliability = torch.sigmoid(reliability_logit)

        base_alpha = self.base_gain * acceptance
        expansion = (
            (1.0 - base_alpha)
            * acceptance
            * expansion_reliability
        )
        alpha = base_alpha + expansion

        return {
            "hidden": hidden,
            "warped_hidden": warped_hidden,
            "error_reliability": error_reliability,
            "proposal_input": aggregate["proposal_input"],
            "proposal_for_decision": delta_z_raw,
            "decision_evidence": decision_evidence,
            "control_hidden": control_hidden,
            "delta_z_raw": delta_z_raw,
            "delta_z": delta_z,
            "acceptance_logit": acceptance_logit,
            "expansion_reliability_logit": reliability_logit,
            "acceptance": acceptance,
            "expansion_reliability": expansion_reliability,
            "base_alpha": base_alpha,
            "expansion": expansion,
            "alpha": alpha,
            # Shared C-V7/C-V9 evaluator multiplies row["gate"] by delta_z.
            "gate_logit": acceptance_logit,
            "gate": alpha,
            **aggregate,
        }
