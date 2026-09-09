"""C-V10 adaptive-amplitude predictive-error correction.

中文：C-V10 自适应幅度预测误差修正器。

Research boundary / 研究边界
----------------------------
1. Raw Current / History semantic logits or probabilities（原始当前/历史语义）
   never enter Proposal / Acceptance / Amplitude heads directly.
2. K aligned history predictions first form Prediction Error（预测误差）.
3. Semantic correction content（语义修正内容）is generated only from
   concat(e1..eK), preserving History -> Prediction -> Prediction Error -> Correction.
4. Motion-aligned Error Memory H_err（运动对齐误差记忆）provides temporal
   reliability context for correction control.
5. Correction control is split into two learned scalar fields:
      acceptance a_t = sigmoid(l_acc)  : whether to accept the proposal;
      amplitude  s_t = sigmoid(l_amp)  : how strongly to apply it.
   The deployed error gain is alpha_t = a_t * s_t in [0, 1].
6. The fixed C-V8/C-V9 g_max=0.25 amplitude cap is removed. This is a structural
   responsibility split, not a g_max sweep.
7. tanh bounded semantic proposal（双曲正切有界语义提议）is retained.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_recurrent_predictor import ConvGRUCell
from .task_space_motion_transport import low_flow_grid
from .task_space_multihypothesis_error_selector import signed_error_channels


class AdaptiveAmplitudeProposalCorrector(nn.Module):
    """Prediction-error proposal with separate acceptance and amplitude control."""

    def __init__(
        self,
        num_classes=19,
        history_length=4,
        hidden_channels=32,
        acceptance_bias=-2.0,
        amplitude_init=0.25,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.history_length = int(history_length)
        self.hidden_channels = int(hidden_channels)
        self.acceptance_bias = float(acceptance_bias)
        self.amplitude_init = float(amplitude_init)
        # Compatibility for the shared C-V7 evaluator metadata only. C-V10 has
        # no fixed scalar g_max multiplier; alpha itself is learned in [0,1].
        self.g_max = 1.0
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        if not 0.0 < self.amplitude_init < 1.0:
            raise ValueError("amplitude_init must be in (0,1)")
        self.amplitude_bias = math.log(
            self.amplitude_init / (1.0 - self.amplitude_init)
        )

        groups = 8 if self.hidden_channels % 8 == 0 else 1

        # Preserve C-V8/C-V9 Error Memory evidence and recurrent dynamics.
        error_channels = (
            2 * self.num_classes * self.history_length
            + 3 * self.num_classes
            + 2
        )
        self.error_pre = nn.Sequential(
            nn.Conv2d(
                error_channels,
                self.hidden_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, self.hidden_channels),
            nn.SiLU(),
        )
        self.error_recurrent = ConvGRUCell(
            self.hidden_channels,
            self.hidden_channels,
        )

        # Semantic correction content remains the exact C-V9 76D->19D path.
        self.proposal_head = nn.Conv2d(
            self.num_classes * self.history_length,
            self.num_classes,
            1,
            bias=False,
        )

        # Decision evidence is unchanged from C-V9: H_err + signed dynamics
        # error + current/history margins + transport/reliability/validity +
        # the actual 19D raw semantic proposal.
        decision_channels = (
            self.hidden_channels
            + 2 * self.num_classes
            + 6
            + self.num_classes
        )
        self.decision_channels = int(decision_channels)

        # Separate learned responsibility paths. Both consume the same evidence,
        # but they do not share their 95D->32D feature transform.
        self.acceptance_pre = nn.Sequential(
            nn.Conv2d(
                self.decision_channels,
                self.hidden_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, self.hidden_channels),
            nn.SiLU(),
        )
        self.acceptance_head = nn.Conv2d(
            self.hidden_channels,
            1,
            1,
            bias=True,
        )
        self.amplitude_pre = nn.Sequential(
            nn.Conv2d(
                self.decision_channels,
                self.hidden_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, self.hidden_channels),
            nn.SiLU(),
        )
        self.amplitude_head = nn.Conv2d(
            self.hidden_channels,
            1,
            1,
            bias=True,
        )

        # Zero proposal preserves exact C-V3 at E0. The two control heads are
        # initialized so their product exactly matches the former C-V9 initial
        # operating point: sigmoid(-2) * 0.25.
        nn.init.zeros_(self.proposal_head.weight)
        nn.init.zeros_(self.acceptance_head.weight)
        nn.init.constant_(self.acceptance_head.bias, self.acceptance_bias)
        nn.init.zeros_(self.amplitude_head.weight)
        nn.init.constant_(self.amplitude_head.bias, self.amplitude_bias)

    @staticmethod
    def _warp_hidden_zero_invalid(previous_hidden, backward_motion):
        if previous_hidden.shape[-2:] != backward_motion.shape[-2:]:
            raise ValueError("error hidden and backward motion must share spatial size")
        grid, valid = low_flow_grid(backward_motion)
        warped = F.grid_sample(
            previous_hidden.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        warped = warped * valid.unsqueeze(1).to(warped.dtype)
        return warped.to(previous_hidden.dtype), valid

    def _aggregate_error_evidence(self, prediction_errors, history_validities_low):
        if len(prediction_errors) != self.history_length:
            raise ValueError("prediction_errors length must equal history_length")
        if len(history_validities_low) != self.history_length:
            raise ValueError("history_validities_low length must equal history_length")

        reference = prediction_errors[0]
        if reference.ndim != 4 or reference.shape[1] != self.num_classes:
            raise ValueError("prediction error must be BCHW with num_classes channels")
        spatial = reference.shape[-2:]

        validities = []
        errors = []
        for error, validity in zip(prediction_errors, history_validities_low):
            if error.shape != reference.shape:
                raise ValueError("all prediction errors must share shape")
            if validity.ndim != 4 or validity.shape[1] != 1:
                raise ValueError("history validity must be Bx1xHxW")
            if validity.shape[-2:] != spatial:
                raise ValueError("history validity spatial size mismatch")
            valid = validity.to(reference.dtype).clamp(0.0, 1.0)
            validities.append(valid)
            errors.append(error * valid)

        valid_stack = torch.cat(validities, dim=1)
        valid_count = valid_stack.sum(dim=1, keepdim=True)
        denominator = valid_count.clamp_min(1.0)
        any_valid = (valid_count > 0.5).to(reference.dtype)
        valid_fraction = valid_count / float(self.history_length)

        stacked = torch.stack(errors, dim=1)
        mean_error = stacked.sum(dim=1) / denominator
        mean_abs_error = stacked.abs().sum(dim=1) / denominator

        signs = torch.sign(stacked)
        validity_5d = valid_stack.unsqueeze(2)
        signed_votes = (signs * validity_5d).sum(dim=1)
        sign_agreement = signed_votes.abs() / denominator

        evidence = torch.cat(
            (
                *[signed_error_channels(error) for error in errors],
                mean_error,
                mean_abs_error,
                sign_agreement,
                valid_fraction,
                any_valid,
            ),
            dim=1,
        )
        proposal_input = torch.cat(errors, dim=1)
        return {
            "evidence": evidence,
            "proposal_input": proposal_input,
            "mean_error": mean_error,
            "mean_abs_error": mean_abs_error,
            "sign_agreement": sign_agreement,
            "valid_fraction": valid_fraction,
            "any_history_valid": any_valid,
        }

    def _build_decision_evidence(
        self,
        hidden,
        dynamics_error,
        current_margin,
        history_margins,
        transportability_low,
        memory_reliability_low,
        aggregate,
        delta_z_raw,
    ):
        best_history_margin = torch.stack(history_margins, dim=1).max(dim=1).values
        return torch.cat(
            (
                hidden,
                signed_error_channels(dynamics_error),
                current_margin,
                best_history_margin,
                transportability_low,
                memory_reliability_low,
                aggregate["valid_fraction"],
                aggregate["any_history_valid"],
                delta_z_raw,
            ),
            dim=1,
        )

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

        # Prediction Error determines semantic correction content.
        delta_z_raw = self.proposal_head(aggregate["proposal_input"])
        delta_z = torch.tanh(delta_z_raw)

        # Temporal/reliability evidence determines whether and how strongly the
        # error-driven semantic proposal is applied.
        decision_evidence = self._build_decision_evidence(
            hidden,
            dynamics_error,
            current_margin,
            history_margins,
            transportability_low,
            memory_reliability_low,
            aggregate,
            delta_z_raw,
        )
        acceptance_hidden = self.acceptance_pre(decision_evidence)
        amplitude_hidden = self.amplitude_pre(decision_evidence)
        acceptance_logit = self.acceptance_head(acceptance_hidden)
        amplitude_logit = self.amplitude_head(amplitude_hidden)
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
            "delta_z_raw": delta_z_raw,
            "delta_z": delta_z,
            "acceptance_logit": acceptance_logit,
            "amplitude_logit": amplitude_logit,
            "acceptance": acceptance,
            "amplitude": amplitude,
            "alpha": alpha,
            # Shared C-V7/C-V9 training and evaluation path multiplies
            # row["gate"] by delta_z. In C-V10 this compatibility slot is alpha.
            "gate": alpha,
            **aggregate,
        }
