"""C-V5 non-autoregressive multi-frame semantic candidate selector.

中文：C-V5 非自回归多帧语义候选选择器。

Design / 设计：
- Current candidate（当前候选）始终来自冻结 C-V3 当前输出。
- History candidates（历史候选）始终来自过去冻结 C-V3 原始输出，经累计运动一次性 warp 到当前帧。
- Controller output（控制器输出）绝不回灌到历史库。
- Prediction Error（预测误差）与 Dynamics Error（动力学误差）仍只使用 t-1 历史，避免错误的深历史改变误差基准。
- Controller 只做 Current / t-1 / ... / t-K 的像素级候选选择，不生成新的语义 logits。
"""

import torch
from torch import nn

from .semantic_recurrent_predictor import ConvGRUCell


class MultiFrameSemanticSelector(nn.Module):
    """Stateful candidate selector（有状态候选选择器） for C-V5.

    Inputs / 输入：
      current probability（当前概率）,
      K history probabilities（K 个历史概率）,
      t-1 prediction error（一步预测误差）,
      dynamics error（动力学误差）,
      confidence margins（置信度间隔）,
      transportability T（传输可行性）,
      memory reliability Q_mem（记忆可靠度）,
      history validity（历史有效性）,
      normalized candidate ages（归一化历史年龄）.

    Output / 输出：K+1 类 selector logits（候选选择得分）。index 0 永远代表 Current。
    最后一层零初始化；torch.argmax 在全零时选择 index 0，因此 E0 严格退化为 C-V3 Current。
    """

    def __init__(self, num_classes=19, history_length=4, hidden_channels=32):
        super().__init__()
        self.num_classes = int(num_classes)
        self.history_length = int(history_length)
        self.hidden_channels = int(hidden_channels)

        # current P + K history P + e_t + epsilon_t
        semantic_channels = self.num_classes * (self.history_length + 3)
        # current margin + K history margins + T + Q + K valid + K ages
        scalar_channels = 1 + self.history_length + 2 + self.history_length + self.history_length
        self.input_channels = semantic_channels + scalar_channels

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
        nn.init.zeros_(self.selector_head.weight)
        nn.init.zeros_(self.selector_head.bias)

    @staticmethod
    def confidence_margin(probability):
        top2 = probability.topk(k=2, dim=1).values
        return top2[:, :1] - top2[:, 1:2]

    def forward(
        self,
        current_probability,
        history_probabilities,
        prediction_error,
        dynamics_error,
        transportability_low,
        memory_reliability_low,
        history_validities_low,
        hidden=None,
    ):
        if len(history_probabilities) != self.history_length:
            raise ValueError("history_probabilities length must equal history_length")
        if len(history_validities_low) != self.history_length:
            raise ValueError("history_validities_low length must equal history_length")

        spatial = current_probability.shape[-2:]
        current_margin = self.confidence_margin(current_probability)
        history_margins = [
            self.confidence_margin(probability)
            for probability in history_probabilities
        ]

        age_maps = []
        for index in range(self.history_length):
            age = float(index + 1) / float(self.history_length)
            age_maps.append(torch.full_like(current_margin, age))

        evidence = torch.cat(
            [
                current_probability,
                *history_probabilities,
                prediction_error,
                dynamics_error,
                current_margin,
                *history_margins,
                transportability_low,
                memory_reliability_low,
                *history_validities_low,
                *age_maps,
            ],
            dim=1,
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
