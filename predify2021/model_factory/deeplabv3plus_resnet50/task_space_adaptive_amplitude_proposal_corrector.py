"""C-V10 adaptive-amplitude predictive-error correction.

中文：C-V10 自适应幅度预测误差修正器。

Research boundary / 研究边界
----------------------------
1. Raw Current / History semantic logits or probabilities（原始当前/历史语义）
   never enter Proposal / Acceptance / Amplitude directly.
2. K aligned history predictions first form Prediction Error（预测误差）.
3. Semantic correction content（语义修正内容）is generated only from
   concat(e1..eK), preserving History -> Prediction -> Prediction Error -> Correction.
4. Motion-aligned Error Memory H_err（运动对齐误差记忆）provides temporal
   reliability context for correction control.
5. C-V9's validated 95D->32D Gate feature extractor is retained as one shared
   Control Pre（共享控制特征提取器）. Two lightweight 32D->1 heads then predict:
      acceptance a_t = sigmoid(l_acc)  : whether to accept the proposal;
      amplitude  s_t = sigmoid(l_amp)  : how strongly to apply it.
   The deployed error gain is alpha_t = a_t * s_t in [0, 1].
6. The fixed C-V8/C-V9 g_max=0.25 amplitude cap is removed. Relative to C-V9,
   the control path adds only one 32D->1 Amplitude Head（幅度头）.
7. tanh bounded semantic proposal（双曲正切有界语义提议）is retained.
"""

import math

import torch
from torch import nn

from .task_space_direct_error_proposal_corrector import DirectErrorProposalCorrector


class AdaptiveAmplitudeProposalCorrector(DirectErrorProposalCorrector):
    """C-V9 proposal/control encoder plus a lightweight adaptive-amplitude head."""

    def __init__(
        self,
        num_classes=19,
        history_length=4,
        hidden_channels=32,
        acceptance_bias=-2.0,
        amplitude_init=0.25,
    ):
        # Reuse the exact C-V9 Proposal, Error Memory, decision evidence,
        # 95D->32D gate_pre, and 32D->1 gate_head. g_max=1.0 is compatibility
        # metadata only; forward below never applies a fixed scalar multiplier.
        super().__init__(
            num_classes=num_classes,
            history_length=history_length,
            hidden_channels=hidden_channels,
            g_max=1.0,
            gate_bias=acceptance_bias,
        )
        self.acceptance_bias = float(acceptance_bias)
        self.amplitude_init = float(amplitude_init)
        if not 0.0 < self.amplitude_init < 1.0:
            raise ValueError("amplitude_init must be in (0,1)")
        self.amplitude_bias = math.log(
            self.amplitude_init / (1.0 - self.amplitude_init)
        )

        self.decision_channels = (
            self.hidden_channels + 2 * self.num_classes + 6 + self.num_classes
        )

        # C-V9 gate_pre is the single shared Control Pre. The inherited gate_head
        # becomes the Acceptance Head; C-V10 adds only this 32D->1 Amplitude Head.
        self.amplitude_head = nn.Conv2d(
            self.hidden_channels,
            1,
            1,
            bias=True,
        )
        nn.init.zeros_(self.amplitude_head.weight)
        nn.init.constant_(self.amplitude_head.bias, self.amplitude_bias)

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

        # Temporal Error Memory: exact C-V9 motion-aligned recurrent path.
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

        # Exact C-V9 decision evidence. One shared 95D->32D Control Pre feeds two
        # scalar heads: Acceptance（是否修）and Amplitude（修多少）.
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
        amplitude_logit = self.amplitude_head(control_hidden)
        acceptance = torch.sigmoid(acceptance_logit)
        amplitude = torch.sigmoid(amplitude_logit)
        alpha = acceptance * amplitude

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
            "amplitude_logit": amplitude_logit,
            "acceptance": acceptance,
            "amplitude": amplitude,
            "alpha": alpha,
            # Shared C-V7/C-V9 training/evaluation code multiplies row["gate"]
            # by delta_z. In C-V10 this compatibility slot is adaptive alpha.
            "gate_logit": acceptance_logit,
            "gate": alpha,
            **aggregate,
        }
