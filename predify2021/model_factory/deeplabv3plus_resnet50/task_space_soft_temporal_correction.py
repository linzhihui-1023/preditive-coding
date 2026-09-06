"""C-V6 set-based deep semantic evidence + soft stateful correction.

中文：集合式深历史语义证据 + 软有状态修正。

Design / 设计：
- K-frame history remains non-autoregressive. Only detached frozen C-V3 outputs
  enter the history bank; corrected outputs never feed back.
- A shared pixel-wise attention scorer keeps every history candidate separate
  until the final weighted semantic evidence is formed.
- The attention is not hard-coded to prefer recent history. Age is evidence,
  not a priority rule.
- A small ConvGRU stores correction-decision context only. It does not store
  semantic class content.
- The final output is a soft convex correction in probability space:
      P_out = (1-lambda) * P_current + lambda * P_history.
"""

import torch
from torch import nn

from .semantic_recurrent_predictor import ConvGRUCell


class SetBasedSoftTemporalCorrector(nn.Module):
    """Pixel-wise history attention + stateful soft correction gate.

    The module runs at the existing low controller resolution. Full-resolution
    history probabilities are mixed later with upsampled attention scores and
    full-resolution validity masks, so semantic logits are never recursively
    resampled through the history window.
    """

    def __init__(
        self,
        num_classes=19,
        history_length=4,
        attention_hidden_channels=16,
        correction_hidden_channels=32,
        gate_init_bias=-4.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.history_length = int(history_length)
        self.attention_hidden_channels = int(attention_hidden_channels)
        self.correction_hidden_channels = int(correction_hidden_channels)
        self.gate_init_bias = float(gate_init_bias)

        # Per-history evidence:
        # current P, history P, |current-history|, valid, normalized age, T, Q.
        attention_in = 3 * self.num_classes + 4
        attention_groups = (
            4 if self.attention_hidden_channels % 4 == 0 else 1
        )
        self.history_score_encoder = nn.Sequential(
            nn.Conv2d(
                attention_in,
                self.attention_hidden_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(attention_groups, self.attention_hidden_channels),
            nn.SiLU(),
        )
        self.history_score_head = nn.Conv2d(
            self.attention_hidden_channels,
            1,
            1,
            bias=True,
        )
        # Equal valid-history attention at initialization. The final soft gate is
        # near zero, so initialization stays close to frozen C-V3.
        nn.init.zeros_(self.history_score_head.weight)
        nn.init.zeros_(self.history_score_head.bias)

        # Correction evidence:
        # current P, attended history P, history-current delta,
        # one-step prediction error, Dynamics Error.
        semantic_channels = 5 * self.num_classes
        # current margin, history margin, T, Q, history available,
        # attention peak, weighted history age.
        scalar_channels = 7
        correction_in = semantic_channels + scalar_channels
        correction_groups = (
            8 if self.correction_hidden_channels % 8 == 0 else 1
        )
        self.correction_encoder = nn.Sequential(
            nn.Conv2d(
                correction_in,
                self.correction_hidden_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(correction_groups, self.correction_hidden_channels),
            nn.SiLU(),
        )
        self.correction_recurrent = ConvGRUCell(
            self.correction_hidden_channels,
            self.correction_hidden_channels,
        )
        self.lambda_head = nn.Conv2d(
            self.correction_hidden_channels,
            1,
            1,
            bias=True,
        )
        nn.init.zeros_(self.lambda_head.weight)
        nn.init.constant_(self.lambda_head.bias, self.gate_init_bias)

    @staticmethod
    def confidence_margin(probability):
        if probability.ndim != 4 or probability.shape[1] < 2:
            raise ValueError("probability must be BCHW with at least two classes")
        top2 = probability.topk(k=2, dim=1).values
        return top2[:, :1] - top2[:, 1:2]

    def _history_attention(
        self,
        current_probability,
        history_probabilities,
        history_validities,
        transportability_low,
        memory_reliability_low,
    ):
        if len(history_probabilities) != self.history_length:
            raise ValueError("history_probabilities length must equal history_length")
        if len(history_validities) != self.history_length:
            raise ValueError("history_validities length must equal history_length")

        current_shape = current_probability.shape
        scores = []
        ages = []
        valid_stack = []
        for index, (history_probability, validity) in enumerate(
            zip(history_probabilities, history_validities)
        ):
            if history_probability.shape != current_shape:
                raise ValueError("history probability must match current probability")
            if validity.ndim != 4 or validity.shape[1] != 1:
                raise ValueError("history validity must be Bx1xHxW")
            age_value = float(index + 1) / float(self.history_length)
            age = torch.full_like(validity, age_value)
            evidence = torch.cat(
                (
                    current_probability,
                    history_probability,
                    (current_probability - history_probability).abs(),
                    validity,
                    age,
                    transportability_low,
                    memory_reliability_low,
                ),
                dim=1,
            )
            score = self.history_score_head(self.history_score_encoder(evidence))
            scores.append(score)
            ages.append(age)
            valid_stack.append(validity)

        score_tensor = torch.cat(scores, dim=1)
        valid_tensor = torch.cat(valid_stack, dim=1).clamp(0.0, 1.0)
        masked_scores = score_tensor.masked_fill(valid_tensor <= 0.5, -1.0e4)
        weights = torch.softmax(masked_scores, dim=1) * valid_tensor
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

        history_probability = torch.zeros_like(current_probability)
        for index, probability in enumerate(history_probabilities):
            history_probability = (
                history_probability + weights[:, index : index + 1] * probability
            )

        age_tensor = torch.cat(ages, dim=1)
        weighted_age = (weights * age_tensor).sum(dim=1, keepdim=True)
        attention_peak = weights.max(dim=1, keepdim=True).values
        history_available = (valid_tensor.sum(dim=1, keepdim=True) > 0.5).to(
            current_probability.dtype
        )
        return {
            "history_scores": score_tensor,
            "history_weights": weights,
            "history_probability": history_probability,
            "history_available": history_available,
            "attention_peak": attention_peak,
            "weighted_age": weighted_age,
        }

    def forward(
        self,
        current_probability,
        history_probabilities,
        history_validities,
        prediction_error,
        dynamics_error,
        transportability_low,
        memory_reliability_low,
        correction_hidden=None,
    ):
        history = self._history_attention(
            current_probability,
            history_probabilities,
            history_validities,
            transportability_low,
            memory_reliability_low,
        )
        history_probability = history["history_probability"]

        current_margin = self.confidence_margin(current_probability)
        history_margin = self.confidence_margin(history_probability)
        correction_evidence = torch.cat(
            (
                current_probability,
                history_probability,
                history_probability - current_probability,
                prediction_error,
                dynamics_error,
                current_margin,
                history_margin,
                transportability_low,
                memory_reliability_low,
                history["history_available"],
                history["attention_peak"],
                history["weighted_age"],
            ),
            dim=1,
        )
        encoded = self.correction_encoder(correction_evidence)
        correction_hidden = self.correction_recurrent(encoded, correction_hidden)
        lambda_logit = self.lambda_head(correction_hidden)
        lambda_probability = torch.sigmoid(lambda_logit) * history["history_available"]
        return {
            **history,
            "lambda_logit": lambda_logit,
            "lambda_probability": lambda_probability,
            "correction_hidden": correction_hidden,
            "current_margin": current_margin,
            "history_margin": history_margin,
        }
