"""C-V8 multi-hypothesis prediction-error direct semantic correction.

中文：C-V8 多假设预测误差直接语义修正。

The module removes Current/history hard selection. Historical semantics are used
only to form strict-validity-gated prediction errors. Those errors, the
motion-aligned recurrent Error State, explicit Dynamics Error and compact
CURRENT semantic state directly produce a 19-D logit residual:

    e_1..e_K -> H_error -> DeltaL_t
    L_out = L_C-V3 + DeltaL_t

Raw historical semantic probabilities never enter this module. The final
correction head is zero-initialized, so step zero is exactly frozen C-V3.
"""

import torch
from torch import nn

from .semantic_recurrent_predictor import ConvGRUCell
from .task_space_multihypothesis_error_selector import signed_error_channels


class MultiHypothesisErrorDirectCorrection(nn.Module):
    """Prediction-error-driven 19-class residual correction（直接误差修正）."""

    def __init__(
        self,
        num_classes=19,
        history_length=4,
        hidden_channels=32,
        current_state_channels=16,
        branch_channels=48,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.history_length = int(history_length)
        self.hidden_channels = int(hidden_channels)
        self.current_state_channels = int(current_state_channels)
        self.branch_channels = int(branch_channels)

        current_groups = 8 if self.current_state_channels % 8 == 0 else 1
        hidden_groups = 8 if self.hidden_channels % 8 == 0 else 1
        branch_groups = 8 if self.branch_channels % 8 == 0 else 1

        # CURRENT semantics are permitted only as a compact state used to
        # interpret prediction error. Historical probabilities are forbidden.
        self.current_state_encoder = nn.Sequential(
            nn.Conv2d(self.num_classes, self.current_state_channels, 1, bias=False),
            nn.GroupNorm(current_groups, self.current_state_channels),
            nn.SiLU(),
        )

        # K signed hypothesis errors + signed explicit Dynamics Error + compact
        # current state + current margin + T + Q + K strict-validity maps.
        signed_error_total = 2 * self.num_classes * (self.history_length + 1)
        scalar_context = 3 + self.history_length
        self.error_input_channels = (
            signed_error_total + self.current_state_channels + scalar_context
        )
        self.error_pre = nn.Sequential(
            nn.Conv2d(
                self.error_input_channels,
                self.hidden_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(hidden_groups, self.hidden_channels),
            nn.SiLU(),
        )
        self.error_recurrent = ConvGRUCell(self.hidden_channels, self.hidden_channels)

        # Readout sees the recurrent Error State and the complete signed error
        # evidence. It outputs a 19-D correction, not a candidate score.
        self.readout_input_channels = (
            self.hidden_channels
            + signed_error_total
            + self.current_state_channels
            + scalar_context
        )
        self.readout_pre = nn.Sequential(
            nn.Conv2d(
                self.readout_input_channels,
                self.branch_channels,
                1,
                bias=False,
            ),
            nn.GroupNorm(branch_groups, self.branch_channels),
            nn.SiLU(),
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(
                self.branch_channels,
                self.branch_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.SiLU(),
        )
        self.context_branch = nn.Sequential(
            nn.Conv2d(
                self.branch_channels,
                self.branch_channels,
                3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.SiLU(),
        )
        self.delta_head = nn.Conv2d(
            2 * self.branch_channels,
            self.num_classes,
            1,
            bias=True,
        )
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        prediction_errors,
        dynamics_error,
        current_probability,
        current_margin,
        transportability_low,
        memory_reliability_low,
        history_validities_low,
        previous_error_state=None,
    ):
        if len(prediction_errors) != self.history_length:
            raise ValueError("prediction_errors length must equal history_length")
        if len(history_validities_low) != self.history_length:
            raise ValueError("history_validities_low length must equal history_length")
        if current_probability.shape[1] != self.num_classes:
            raise ValueError("current_probability must have num_classes channels")
        if dynamics_error.shape[1] != self.num_classes:
            raise ValueError("dynamics_error must have num_classes channels")

        spatial = tuple(current_probability.shape[-2:])
        for name, value in (
            ("dynamics_error", dynamics_error),
            ("current_margin", current_margin),
            ("transportability_low", transportability_low),
            ("memory_reliability_low", memory_reliability_low),
        ):
            if tuple(value.shape[-2:]) != spatial:
                raise ValueError(f"{name} spatial size mismatch")
        for error in prediction_errors:
            if error.shape[1] != self.num_classes or tuple(error.shape[-2:]) != spatial:
                raise ValueError("prediction error shape mismatch")
        for validity in history_validities_low:
            if validity.shape[1] != 1 or tuple(validity.shape[-2:]) != spatial:
                raise ValueError("history validity shape mismatch")

        current_probability = current_probability.detach()
        current_state = self.current_state_encoder(current_probability)
        signed_errors = [signed_error_channels(error) for error in prediction_errors]
        signed_dynamics = signed_error_channels(dynamics_error)

        shared_evidence = torch.cat(
            [
                *signed_errors,
                signed_dynamics,
                current_state,
                current_margin,
                transportability_low,
                memory_reliability_low,
                *history_validities_low,
            ],
            dim=1,
        )
        if shared_evidence.shape[1] != self.error_input_channels:
            raise RuntimeError(
                f"error evidence channels mismatch: {shared_evidence.shape[1]} "
                f"!= {self.error_input_channels}"
            )

        encoded = self.error_pre(shared_evidence)
        error_state = self.error_recurrent(encoded, previous_error_state)

        readout_evidence = torch.cat((error_state, shared_evidence), dim=1)
        if readout_evidence.shape[1] != self.readout_input_channels:
            raise RuntimeError(
                f"readout channels mismatch: {readout_evidence.shape[1]} "
                f"!= {self.readout_input_channels}"
            )
        shared = self.readout_pre(readout_evidence)
        local = self.local_branch(shared)
        context = self.context_branch(shared)
        delta_logits = self.delta_head(torch.cat((local, context), dim=1))

        # No valid historical hypothesis means no predictive-coding correction.
        # This is a structural mask, not a learned gate.
        any_history_valid = torch.stack(
            [validity.detach() for validity in history_validities_low],
            dim=0,
        ).amax(dim=0).clamp(0.0, 1.0)
        delta_logits = delta_logits * any_history_valid

        return {
            "delta_logits": delta_logits,
            "error_state": error_state,
            "current_state": current_state,
            "any_history_valid": any_history_valid,
        }
