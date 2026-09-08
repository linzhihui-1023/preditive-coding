"""C-V7 multi-hypothesis prediction-error residual corrector.

中文：C-V7 多假设预测误差残差修正器。

Core constraints / 核心约束：
1. Raw Current / History semantic probabilities（原始当前/历史语义概率）绝不进入修正头；
2. K 个历史候选先形成 validity-gated Prediction Error（有效性门控预测误差）；
3. Error Memory（误差记忆）先按当前 backward motion（反向运动）对齐，再由
   frozen C-V3 memory reliability（冻结 C-V3 记忆可靠度）与历史有效性门控；
4. Correction Head（修正头）最后一层严格零初始化，保证 E0（零步）修正为 0；
5. Gate Head（门控头）只输出单通道修正强度，不直接承担类别语义修正；
6. 最终在 logit space（分类得分空间）执行 bounded residual correction（有界残差修正）。
"""

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_recurrent_predictor import ConvGRUCell
from .task_space_motion_transport import low_flow_grid
from .task_space_multihypothesis_error_selector import signed_error_channels


class MultiHypothesisErrorResidualCorrector(nn.Module):
    """Motion-aligned recurrent error aggregation and bounded logit correction.

    中文：运动对齐循环误差聚合 + 有界分类得分残差修正。
    """

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

        # Current error evidence only. No raw current/history semantic probability.
        # K signed errors + mean signed error + mean absolute error
        # + cross-history sign agreement + valid fraction + any-valid.
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

        self.correction_head = nn.Sequential(
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                3,
                padding=1,
                bias=True,
            ),
            nn.SiLU(),
            nn.Conv2d(
                self.hidden_channels,
                self.num_classes,
                1,
                bias=True,
            ),
        )

        # Gate evidence: error memory + signed Dynamics Error + protection evidence.
        # Protection scalars: current margin, best-history margin, T, Q,
        # valid fraction, any-history-valid.
        gate_channels = self.hidden_channels + 2 * self.num_classes + 6
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

        # Exact E0 equality to frozen C-V3 comes from zero correction.
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

        # The gate starts conservative but not saturated, so the zero-initialized
        # correction head can still receive a usable first-step gradient.
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, self.gate_bias)

    @staticmethod
    def _warp_hidden_zero_invalid(previous_hidden, backward_motion):
        """Warp recurrent Error Memory into the current frame.

        中文：将上一帧误差记忆运动对齐到当前帧，并把越界位置清零。
        """
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
            # The caller already validity-gates e_k. Multiply again deliberately
            # to make this module robust to accidental non-zero invalid errors.
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
        return {
            "evidence": evidence,
            "mean_error": mean_error,
            "mean_abs_error": mean_abs_error,
            "sign_agreement": sign_agreement,
            "valid_fraction": valid_fraction,
            "any_history_valid": any_valid,
        }

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

        # Reuse frozen C-V3 reliability as the primary history trust signal.
        # Any-history-valid and warp validity prevent stale/error state leakage.
        error_reliability = (
            memory_reliability_low.detach().clamp(0.0, 1.0)
            * aggregate["any_history_valid"]
            * hidden_valid
        )
        gated_history = error_reliability * warped_hidden
        hidden = self.error_recurrent(encoded, gated_history)

        delta_z_raw = self.correction_head(hidden)
        delta_z = torch.tanh(delta_z_raw)

        best_history_margin = torch.stack(history_margins, dim=1).max(dim=1).values
        gate_evidence = torch.cat(
            (
                hidden,
                signed_error_channels(dynamics_error),
                current_margin,
                best_history_margin,
                transportability_low,
                memory_reliability_low,
                aggregate["valid_fraction"],
                aggregate["any_history_valid"],
            ),
            dim=1,
        )
        gate_hidden = self.gate_pre(gate_evidence)
        gate_logit = self.gate_head(gate_hidden)
        # Do not apply low-resolution validity here. The caller upsamples this
        # raw bounded gate first and applies the full-resolution validity mask
        # exactly once, matching the reviewed C-V6H no-mask-bleed rule.
        gate = self.g_max * torch.sigmoid(gate_logit)

        return {
            "hidden": hidden,
            "warped_hidden": warped_hidden,
            "error_reliability": error_reliability,
            "delta_z_raw": delta_z_raw,
            "delta_z": delta_z,
            "gate_logit": gate_logit,
            "gate": gate,
            **aggregate,
        }
