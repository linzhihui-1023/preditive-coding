"""C-V7 multi-hypothesis prediction-error residual correction.

中文：C-V7 多假设预测误差残差修正。

核心边界：
1. Raw Current / History semantic probabilities（原始当前/历史语义概率）不直接进入修正头；
2. History（历史）必须先形成 motion-aligned temporal predictions（运动对齐时序预测），
   再形成 validity-gated Prediction Errors（有效性门控预测误差）；
3. Error State（误差状态）像 C-V3 Semantic Memory（语义记忆）一样先做运动对齐，
   再用 reliability（可靠性）门控后送入 ConvGRU（卷积门控循环单元）；
4. Correction Head（修正头）只产生 logit-space residual（分类得分空间残差）；
5. Protection Gate（保护门控）只控制修正幅度，不产生类别级语义方向；
6. 最终修正严格有界，并且零步严格等价于冻结 C-V3。
"""

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_recurrent_predictor import ConvGRUCell
from .task_space_motion_transport import low_flow_grid
from .task_space_multihypothesis_error_selector import signed_error_channels


class MotionAlignedErrorState(nn.Module):
    """Motion-aligned recurrent Prediction-Error state（运动对齐循环预测误差状态）."""

    def __init__(
        self,
        num_classes=19,
        history_length=4,
        hidden_channels=64,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.history_length = int(history_length)
        self.hidden_channels = int(hidden_channels)

        # K signed errors + mean error + mean absolute error + agreement
        # + K validity maps + valid fraction.
        semantic_channels = (
            2 * self.num_classes * self.history_length
            + self.num_classes
            + self.num_classes
            + self.num_classes
        )
        scalar_channels = self.history_length + 1
        self.input_channels = semantic_channels + scalar_channels

        groups = 8 if self.hidden_channels % 8 == 0 else 1
        self.pre = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, self.hidden_channels),
            nn.SiLU(),
        )
        self.recurrent = ConvGRUCell(self.hidden_channels, self.hidden_channels)

    @staticmethod
    def _warp_zero_invalid(previous_state, backward_motion):
        """Reuse the C-V3 state-transport rule（复用 C-V3 状态搬运规则）."""
        if previous_state.shape[-2:] != backward_motion.shape[-2:]:
            raise ValueError("Error state and motion must share spatial size")
        grid, valid = low_flow_grid(backward_motion)
        warped = F.grid_sample(
            previous_state.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        warped = warped * valid.unsqueeze(1).to(warped.dtype)
        return warped.to(previous_state.dtype), valid

    def _aggregate_error_evidence(self, prediction_errors, validities):
        if len(prediction_errors) != self.history_length:
            raise ValueError("prediction_errors length must equal history_length")
        if len(validities) != self.history_length:
            raise ValueError("validities length must equal history_length")

        reference = prediction_errors[0]
        if reference.shape[1] != self.num_classes:
            raise ValueError("Prediction Error must have num_classes channels")

        valid_stack = []
        weighted_errors = []
        signed = []
        for error, validity in zip(prediction_errors, validities):
            if error.shape != reference.shape:
                raise ValueError("all Prediction Errors must share shape")
            if validity.ndim != 4 or validity.shape[1] != 1:
                raise ValueError("history validity must have shape [N,1,H,W]")
            if validity.shape[-2:] != reference.shape[-2:]:
                raise ValueError("history validity must match Prediction Error size")
            valid = validity.to(reference.dtype).clamp(0.0, 1.0)
            valid_stack.append(valid)
            weighted_errors.append(error * valid)
            signed.append(signed_error_channels(error * valid))

        validity_sum = torch.stack(valid_stack, dim=0).sum(dim=0)
        denom = validity_sum.clamp_min(1.0)
        error_sum = torch.stack(weighted_errors, dim=0).sum(dim=0)
        mean_error = error_sum / denom
        mean_abs_error = (
            torch.stack([error.abs() * valid for error, valid in zip(prediction_errors, valid_stack)], dim=0)
            .sum(dim=0)
            / denom
        )

        # Class-wise sign agreement（类别级符号一致性） in [0,1].
        sign_votes = torch.stack(
            [torch.sign(error) * valid for error, valid in zip(prediction_errors, valid_stack)],
            dim=0,
        ).sum(dim=0)
        agreement = sign_votes.abs() / denom

        valid_fraction = validity_sum / float(self.history_length)
        evidence = torch.cat(
            (
                *signed,
                mean_error,
                mean_abs_error,
                agreement,
                *valid_stack,
                valid_fraction,
            ),
            dim=1,
        )
        return evidence, mean_error, mean_abs_error, agreement, valid_fraction

    def forward(
        self,
        prediction_errors,
        history_validities,
        backward_motion_low,
        reliability_low,
        previous_state=None,
    ):
        evidence, mean_error, mean_abs_error, agreement, valid_fraction = (
            self._aggregate_error_evidence(prediction_errors, history_validities)
        )
        encoded = self.pre(evidence)

        if previous_state is None:
            previous_state = torch.zeros_like(encoded)
        warped_state, motion_valid = self._warp_zero_invalid(
            previous_state,
            backward_motion_low,
        )

        if reliability_low.ndim != 4 or reliability_low.shape[1] != 1:
            raise ValueError("reliability_low must be single-channel BCHW")
        if reliability_low.shape[-2:] != encoded.shape[-2:]:
            raise ValueError("reliability_low must match Error State size")

        motion_valid = motion_valid.unsqueeze(1).to(encoded.dtype)
        any_history_valid = (valid_fraction > 0.0).to(encoded.dtype)
        reliability_effective = (
            reliability_low.detach().clamp(0.0, 1.0)
            * motion_valid
            * any_history_valid
        )
        gated_history = reliability_effective * warped_state
        state = self.recurrent(encoded, gated_history)

        return {
            "state": state,
            "encoded": encoded,
            "warped_state": warped_state,
            "reliability_effective": reliability_effective,
            "mean_error": mean_error,
            "mean_abs_error": mean_abs_error,
            "agreement": agreement,
            "valid_fraction": valid_fraction,
        }


class MultiHypothesisErrorResidualCorrector(nn.Module):
    """Predict bounded logit correction from Prediction Error only（仅由预测误差产生有界分类得分修正）."""

    def __init__(
        self,
        num_classes=19,
        history_length=4,
        hidden_channels=64,
        correction_channels=64,
        gate_max=0.25,
        gate_init_bias=-4.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.history_length = int(history_length)
        self.hidden_channels = int(hidden_channels)
        self.correction_channels = int(correction_channels)
        self.gate_max = float(gate_max)
        if not 0.0 < self.gate_max <= 1.0:
            raise ValueError("gate_max must be in (0,1]")

        self.error_state = MotionAlignedErrorState(
            num_classes=self.num_classes,
            history_length=self.history_length,
            hidden_channels=self.hidden_channels,
        )

        groups = 8 if self.correction_channels % 8 == 0 else 1
        self.correction_pre = nn.Sequential(
            nn.Conv2d(
                self.hidden_channels,
                self.correction_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, self.correction_channels),
            nn.SiLU(),
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(
                self.correction_channels,
                self.correction_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.SiLU(),
        )
        self.context_branch = nn.Sequential(
            nn.Conv2d(
                self.correction_channels,
                self.correction_channels,
                3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.SiLU(),
        )
        self.correction_head = nn.Conv2d(
            2 * self.correction_channels,
            self.num_classes,
            kernel_size=1,
            bias=True,
        )

        # Gate receives protection evidence only. It is intentionally single-channel.
        # state + signed Dynamics Error + current/best-history margin + T + Q
        # + any-history-valid + mean agreement.
        gate_channels = self.hidden_channels + 2 * self.num_classes + 6
        self.gate_pre = nn.Sequential(
            nn.Conv2d(gate_channels, self.hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(
                8 if self.hidden_channels % 8 == 0 else 1,
                self.hidden_channels,
            ),
            nn.SiLU(),
        )
        self.gate_head = nn.Conv2d(self.hidden_channels, 1, 1, bias=True)

        # E0（零步）双重保护：ΔZ_raw = 0，同时 g≈0。
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_init_bias))

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
        previous_error_state=None,
    ):
        if len(history_margins) != self.history_length:
            raise ValueError("history_margins length must equal history_length")
        if dynamics_error.shape[1] != self.num_classes:
            raise ValueError("Dynamics Error must have num_classes channels")

        state_row = self.error_state(
            prediction_errors=prediction_errors,
            history_validities=history_validities_low,
            backward_motion_low=backward_motion_low,
            reliability_low=memory_reliability_low,
            previous_state=previous_error_state,
        )
        error_state = state_row["state"]

        shared = self.correction_pre(error_state)
        local = self.local_branch(shared)
        context = self.context_branch(shared)
        raw_delta = self.correction_head(torch.cat((local, context), dim=1))
        bounded_delta = torch.tanh(raw_delta)

        validity_stack = torch.cat(history_validities_low, dim=1)
        any_history_valid = (
            validity_stack.max(dim=1, keepdim=True).values > 0.5
        ).to(current_margin.dtype)
        best_history_margin = torch.stack(history_margins, dim=1).max(dim=1).values
        mean_agreement = state_row["agreement"].mean(dim=1, keepdim=True)

        gate_evidence = torch.cat(
            (
                error_state,
                signed_error_channels(dynamics_error),
                current_margin,
                best_history_margin,
                transportability_low,
                memory_reliability_low,
                any_history_valid,
                mean_agreement,
            ),
            dim=1,
        )
        gate_hidden = self.gate_pre(gate_evidence)
        gate_logit = self.gate_head(gate_hidden)
        gate = self.gate_max * torch.sigmoid(gate_logit)
        gate = gate * any_history_valid

        correction_low = gate * bounded_delta
        return {
            "correction_low": correction_low,
            "raw_delta_low": raw_delta,
            "bounded_delta_low": bounded_delta,
            "gate_low": gate,
            "gate_logit_low": gate_logit,
            "error_state": error_state,
            "warped_error_state": state_row["warped_state"],
            "error_state_reliability": state_row["reliability_effective"],
            "mean_error": state_row["mean_error"],
            "mean_abs_error": state_row["mean_abs_error"],
            "agreement": state_row["agreement"],
            "valid_fraction": state_row["valid_fraction"],
        }
