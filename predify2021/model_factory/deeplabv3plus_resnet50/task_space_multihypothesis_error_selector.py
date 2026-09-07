"""C-V6 multi-hypothesis prediction-error selector.

中文：C-V6 多假设预测误差选择器。

Core constraint / 核心约束：
- Controller（控制器）不接收 current/history semantic probabilities（当前/历史完整语义概率）。
- 每个历史候选先与当前语义形成 prediction error（预测误差），并由对应 validity（有效性）显式门控。
- Candidate Bank（候选库）只在控制器输出 selector logits 后执行最终选择，不向控制器泄露完整历史语义。
- Dynamics Error（动力学误差）继续作为显式误差动态证据。

This makes cross-frame semantic information enter the decision core only through
prediction-error representations and reliability scalars.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_recurrent_predictor import ConvGRUCell


def confidence_margin(probability: torch.Tensor) -> torch.Tensor:
    """Return top-1 minus top-2 confidence margin（置信度间隔）."""
    top2 = probability.topk(k=2, dim=1).values
    return top2[:, :1] - top2[:, 1:2]


def build_multihypothesis_error_evidence(
    current_probability: torch.Tensor,
    history_probabilities,
    history_validities,
):
    """Build validity-gated multi-hypothesis errors（有效性门控多假设误差）.

    For history candidate k:
        e_k = V_k * (P_current - P_history_k)

    Missing/invalid history therefore contributes exactly zero semantic error.
    The function also returns scalar confidence margins; raw semantic
    probabilities are not returned to the controller.
    """
    if len(history_probabilities) != len(history_validities):
        raise ValueError("history probabilities and validities must have equal length")

    errors = []
    history_margins = []
    for probability, validity in zip(history_probabilities, history_validities):
        if probability.shape != current_probability.shape:
            raise ValueError("all history probabilities must match current probability shape")
        if validity.shape[0] != current_probability.shape[0] or validity.shape[1] != 1:
            raise ValueError("history validity must have shape [N,1,H,W]")
        if validity.shape[-2:] != current_probability.shape[-2:]:
            raise ValueError("history validity must match semantic spatial size")

        valid = validity.to(current_probability.dtype).clamp(0.0, 1.0)
        error = (current_probability - probability) * valid
        errors.append(error)
        history_margins.append(confidence_margin(probability) * valid)

    return {
        "prediction_errors": errors,
        "current_margin": confidence_margin(current_probability),
        "history_margins": history_margins,
    }


def signed_error_channels(error: torch.Tensor) -> torch.Tensor:
    """Encode signed class-wise error without losing sign information.

    中文：将正误差与负误差拆成两组非负通道：
      [ReLU(e), ReLU(-e)].
    """
    return torch.cat((F.relu(error), F.relu(-error)), dim=1)


class MultiHypothesisErrorSelector(nn.Module):
    """Stateful error-only candidate controller（有状态仅误差候选控制器）.

    Semantic inputs / 语义输入：
      - K validity-gated class-wise prediction errors（K 个有效性门控类别级预测误差）;
      - one dynamics error state（一个动力学误差状态）.

    Reliability inputs / 可靠性输入：
      - current confidence margin（当前置信度间隔）;
      - K history confidence margins（K 个历史置信度间隔）;
      - transportability T（传输可行性）;
      - memory reliability Q_mem（记忆可靠度）;
      - K validity maps（K 个有效性图）;
      - K normalized ages（K 个归一化历史年龄）.

    Raw current/history semantic probabilities are intentionally excluded.
    Output index 0 always denotes Current; 1..K denote history candidates.
    """

    def __init__(self, num_classes=19, history_length=4, hidden_channels=32):
        super().__init__()
        self.num_classes = int(num_classes)
        self.history_length = int(history_length)
        self.hidden_channels = int(hidden_channels)

        # K prediction errors + one dynamics error, each split into positive and
        # negative class-wise channels. No raw semantic probability enters here.
        signed_semantic_channels = (
            2 * self.num_classes * (self.history_length + 1)
        )
        # current margin + K history margins + T + Q + K valid + K ages
        scalar_channels = 1 + self.history_length + 2 + self.history_length + self.history_length
        self.input_channels = signed_semantic_channels + scalar_channels

        groups = 8 if self.hidden_channels % 8 == 0 else 1
        self.pre = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_channels),
            nn.SiLU(),
        )
        self.recurrent = ConvGRUCell(self.hidden_channels, self.hidden_channels)
        self.selector_head = nn.Conv2d(
            self.hidden_channels,
            self.history_length + 1,
            1,
            bias=True,
        )

        # E0 strictly falls back to frozen C-V3 Current.
        nn.init.zeros_(self.selector_head.weight)
        nn.init.zeros_(self.selector_head.bias)

    def forward(
        self,
        prediction_errors,
        dynamics_error,
        current_margin,
        history_margins,
        transportability_low,
        memory_reliability_low,
        history_validities_low,
        hidden=None,
    ):
        if len(prediction_errors) != self.history_length:
            raise ValueError("prediction_errors length must equal history_length")
        if len(history_margins) != self.history_length:
            raise ValueError("history_margins length must equal history_length")
        if len(history_validities_low) != self.history_length:
            raise ValueError("history_validities_low length must equal history_length")

        spatial = dynamics_error.shape[-2:]
        for error in prediction_errors:
            if error.shape[1] != self.num_classes or error.shape[-2:] != spatial:
                raise ValueError("prediction error shape mismatch")
        if dynamics_error.shape[1] != self.num_classes:
            raise ValueError("dynamics_error must have num_classes channels")

        age_maps = []
        for index in range(self.history_length):
            age = float(index + 1) / float(self.history_length)
            age_maps.append(torch.full_like(current_margin, age))

        semantic_evidence = [
            signed_error_channels(error)
            for error in prediction_errors
        ]
        semantic_evidence.append(signed_error_channels(dynamics_error))

        evidence = torch.cat(
            [
                *semantic_evidence,
                current_margin,
                *history_margins,
                transportability_low,
                memory_reliability_low,
                *history_validities_low,
                *age_maps,
            ],
            dim=1,
        )
        if evidence.shape[1] != self.input_channels:
            raise RuntimeError(
                f"selector input channels mismatch: {evidence.shape[1]} != {self.input_channels}"
            )
        if evidence.shape[-2:] != spatial:
            raise RuntimeError("selector evidence spatial size mismatch")

        encoded = self.pre(evidence)
        hidden = self.recurrent(encoded, hidden)
        selector_logits = self.selector_head(hidden)
        return {
            "selector_logits": selector_logits,
            "hidden": hidden,
            "current_margin": current_margin,
            "history_margins": history_margins,
        }
