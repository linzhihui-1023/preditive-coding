"""Direct Prediction-Error Proposal + Temporal Error Memory + Proposal-Aware Gate.

中文：直接预测误差提议 + 时序误差记忆 + 提议感知门控。

Research boundary / 研究边界
----------------------------
1. Raw Current / History semantic logits or probabilities（原始当前/历史语义）
   never enter Proposal Head（提议头）or Gate（门控）directly.
2. K=4 aligned history predictions first form Prediction Error（预测误差）.
3. Semantic correction content（语义修正内容）comes directly from concat(e1..e4).
4. Error Memory H_err（误差记忆）keeps the existing motion-aligned recurrent path
   and serves temporal context / reliability, not the sole semantic readout.
5. Gate is proposal-aware: it explicitly observes DeltaZ_proposal together with
   H_err, Dynamics Error（动力学误差）and reliability evidence.
6. Proposal and Gate are jointly optimized by the final segmentation CE.  The
   inference responsibility remains separated because Proposal depends only on
   Prediction Error, while Gate decides proposal acceptance from temporal/safety
   context plus the proposal itself.
7. tanh bounded residual（双曲正切有界残差）and scalar g_max Gate are retained.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_recurrent_predictor import ConvGRUCell
from .task_space_motion_transport import low_flow_grid
from .task_space_multihypothesis_error_selector import signed_error_channels


class DirectErrorProposalCorrector(nn.Module):
    """Direct class-wise error proposal with recurrent temporal safety context."""

    def __init__(
        self,
        num_classes=19,
        history_length=4,
        hidden_channels=32,
        g_max=0.25,
        gate_bias=-2.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.history_length = int(history_length)
        self.hidden_channels = int(hidden_channels)
        self.g_max = float(g_max)
        self.gate_bias = float(gate_bias)
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        if not 0.0 < self.g_max <= 1.0:
            raise ValueError("g_max must be in (0,1]")

        groups = 8 if self.hidden_channels % 8 == 0 else 1

        # Existing C-V7 Error Memory evidence is intentionally preserved.
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

        # Diagnostic evidence showed concat(e1..e4) has strong class-wise linear
        # decodability. Keep the first deployable proposal path equally minimal:
        # 76D -> 19D for K=4/C=19, no bias, exact zero initialization.
        self.proposal_head = nn.Conv2d(
            self.num_classes * self.history_length,
            self.num_classes,
            1,
            bias=False,
        )

        # Gate evidence keeps the validated temporal/reliability context and adds
        # the actual 19D semantic proposal.  No extra stop-gradient is introduced:
        # the first architecture test changes the information path, not the
        # optimization graph beyond what follows naturally from that path.
        gate_channels = (
            self.hidden_channels
            + 2 * self.num_classes
            + 6
            + self.num_classes
        )
        self.gate_pre = nn.Sequential(
            nn.Conv2d(
                gate_channels,
                self.hidden_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, self.hidden_channels),
            nn.SiLU(),
        )
        self.gate_head = nn.Conv2d(
            self.hidden_channels,
            1,
            1,
            bias=True,
        )

        # E0 is exactly frozen C-V3 because the semantic proposal is zero.
        nn.init.zeros_(self.proposal_head.weight)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, self.gate_bias)

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
            # Caller already validity-gates e_k; repeat locally as a hard contract.
            errors.append(error * valid)

        valid_stack = torch.cat(validities, dim=1)
        valid_count = valid_stack.sum(dim=1, keepdim=True)
        denominator = valid_count.clamp_min(1.0)
        any_valid = (valid_count > 0.5).to(reference.dtype)
        valid_fraction = valid_count / float(self.history_length)

        stacked = torch.stack(errors, dim=1)  # B,K,C,H,W
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

    def _build_gate_evidence(
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

        # Temporal Error Memory: same C-V7 motion-aligned recurrent path.
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

        # Semantic content comes directly from the K class-wise Prediction Errors.
        delta_z_raw = self.proposal_head(aggregate["proposal_input"])
        delta_z = torch.tanh(delta_z_raw)

        # Temporal memory conditions acceptance; Gate also sees proposal content.
        gate_evidence = self._build_gate_evidence(
            hidden,
            dynamics_error,
            current_margin,
            history_margins,
            transportability_low,
            memory_reliability_low,
            aggregate,
            delta_z_raw,
        )
        gate_hidden = self.gate_pre(gate_evidence)
        gate_logit = self.gate_head(gate_hidden)
        gate = self.g_max * torch.sigmoid(gate_logit)

        return {
            "hidden": hidden,
            "warped_hidden": warped_hidden,
            "error_reliability": error_reliability,
            "proposal_input": aggregate["proposal_input"],
            "proposal_for_gate": delta_z_raw,
            "gate_evidence": gate_evidence,
            # Keep C-V7-compatible output names for the shared training/eval path.
            "delta_z_raw": delta_z_raw,
            "delta_z": delta_z,
            "gate_logit": gate_logit,
            "gate": gate,
            **aggregate,
        }
