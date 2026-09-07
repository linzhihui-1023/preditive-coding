"""C-V8 multi-hypothesis prediction-error direct semantic correction.

中文：C-V8 多假设预测误差直接语义修正。

Historical semantics are used only to form strict-validity-gated prediction
errors. Two error representations have distinct roles:

    probability error e^P_k = P_current - P_history_k
      -> recurrent Error State / gain evidence

    centered-logit error e^L_k = C(L_current) - C(L_history_k)
      -> correction direction

where C(L) subtracts the per-pixel class mean. The network never predicts an
arbitrary semantic residual. It learns class-wise gains for each historical
hypothesis and applies them only along the centered-logit prediction-error
direction:

    DeltaL_k = - G_k * e^L_k
    DeltaL_t = sum_k DeltaL_k
    L_out = L_C-V3 + DeltaL_t

Prediction Error is therefore structurally necessary: if all correction errors
are zero, DeltaL_t is exactly zero even after training. Current semantics are
available only as a compact state for interpreting error; raw historical
probabilities/logits never enter the correction module.
"""

import torch
from torch import nn

from .semantic_recurrent_predictor import ConvGRUCell
from .task_space_multihypothesis_error_selector import signed_error_channels


class MultiHypothesisErrorDirectCorrection(nn.Module):
    """Class-wise multi-history prediction-error correction（逐类直接误差修正）."""

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

        self.current_state_encoder = nn.Sequential(
            nn.Conv2d(self.num_classes, self.current_state_channels, 1, bias=False),
            nn.GroupNorm(current_groups, self.current_state_channels),
            nn.SiLU(),
        )

        # Global recurrent Error State from all K signed probability errors,
        # explicit Dynamics Error and compact CURRENT context.
        signed_error_total = 2 * self.num_classes * (self.history_length + 1)
        global_scalar_context = 3 + self.history_length  # margin, T, Q, V1..VK
        self.error_input_channels = (
            signed_error_total
            + self.current_state_channels
            + global_scalar_context
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

        # One shared candidate-conditioned gain readout is reused for t-1..t-K.
        # It predicts 19 class-wise gains, not one scalar utility and not a free
        # 19-D semantic residual.
        self.candidate_input_channels = (
            2 * self.num_classes
            + self.hidden_channels
            + self.current_state_channels
            + 6  # current margin, T, Q, validity, normalized age, dynamics magnitude
        )
        self.candidate_pre = nn.Sequential(
            nn.Conv2d(
                self.candidate_input_channels,
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
        self.gain_head = nn.Conv2d(
            2 * self.branch_channels,
            self.num_classes,
            1,
            bias=True,
        )
        nn.init.zeros_(self.gain_head.weight)
        nn.init.zeros_(self.gain_head.bias)

    def forward(
        self,
        prediction_errors,
        correction_errors,
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
        if len(correction_errors) != self.history_length:
            raise ValueError("correction_errors length must equal history_length")
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
        for error in correction_errors:
            if error.shape[1] != self.num_classes or tuple(error.shape[-2:]) != spatial:
                raise ValueError("correction error shape mismatch")
        for validity in history_validities_low:
            if validity.shape[1] != 1 or tuple(validity.shape[-2:]) != spatial:
                raise ValueError("history validity shape mismatch")

        current_probability = current_probability.detach()
        current_state = self.current_state_encoder(current_probability)
        signed_errors = [signed_error_channels(error) for error in prediction_errors]
        signed_dynamics = signed_error_channels(dynamics_error)

        global_evidence = torch.cat(
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
        if global_evidence.shape[1] != self.error_input_channels:
            raise RuntimeError(
                f"error evidence channels mismatch: {global_evidence.shape[1]} "
                f"!= {self.error_input_channels}"
            )
        encoded = self.error_pre(global_evidence)
        error_state = self.error_recurrent(encoded, previous_error_state)

        dynamics_magnitude = dynamics_error.abs().mean(dim=1, keepdim=True)
        correction_terms = []
        gains = []
        for index in range(self.history_length):
            validity = history_validities_low[index].detach().clamp(0.0, 1.0)
            age = float(index + 1) / float(self.history_length)
            age_map = torch.full_like(current_margin, age)
            candidate = torch.cat(
                [
                    signed_errors[index],
                    error_state,
                    current_state,
                    current_margin,
                    transportability_low,
                    memory_reliability_low,
                    validity,
                    age_map,
                    dynamics_magnitude,
                ],
                dim=1,
            )
            if candidate.shape[1] != self.candidate_input_channels:
                raise RuntimeError(
                    f"candidate channels mismatch: {candidate.shape[1]} "
                    f"!= {self.candidate_input_channels}"
                )
            shared = self.candidate_pre(candidate)
            local = self.local_branch(shared)
            context = self.context_branch(shared)
            gain = self.gain_head(torch.cat((local, context), dim=1)) * validity
            gains.append(gain)
            # e^L_k = centered Current logits - centered History logits.
            # Negative e^L_k points from Current toward the historical hypothesis.
            correction_terms.append(-gain * correction_errors[index])

        candidate_corrections_low = torch.stack(correction_terms, dim=1)
        delta_logits = candidate_corrections_low.sum(dim=1)
        candidate_gains = torch.stack(gains, dim=1)
        any_history_valid = torch.stack(
            [validity.detach() for validity in history_validities_low],
            dim=0,
        ).amax(dim=0).clamp(0.0, 1.0)

        return {
            "delta_logits": delta_logits,
            "candidate_corrections_low": candidate_corrections_low,
            "candidate_gains": candidate_gains,
            "error_state": error_state,
            "current_state": current_state,
            "any_history_valid": any_history_valid,
        }
