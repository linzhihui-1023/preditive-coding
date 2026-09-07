"""Semantic refinement under a frozen temporal correction budget.

中文：冻结时序修正预算下的语义细化模块。

C-V6S does not replace the C-V6R temporal history distribution from scratch.
It starts from the frozen C-V6R history distribution P_hist_temp and learns:

    P_hist_sem = softmax(log(P_hist_temp + eps) + Delta_sem)
    lambda_final = lambda_temp * (1 - clamp(d_sem, 0, 1))

Both Delta_sem and d_sem are exactly zero initialized, so step zero reproduces
C-V6R-E1 exactly. Validity is a hard mask for history composition; age is only
soft evidence. Prediction Error / Dynamics Error are deliberately excluded from
this module because they belong to the frozen temporal-decision branch.
"""

import torch
from torch import nn


class SemanticBudgetRefiner(nn.Module):
    """Class-wise K-frame semantic evidence + correction-budget attenuation."""

    def __init__(
        self,
        num_classes=19,
        history_length=4,
        score_hidden_channels=16,
        refine_hidden_channels=32,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.history_length = int(history_length)

        # Per-history semantic evidence:
        # P_cur, H_k, |H_k-P_cur|, validity, normalized age, T, Q.
        score_in = 3 * self.num_classes + 4
        score_groups = 4 if score_hidden_channels % 4 == 0 else 1
        self.score_encoder = nn.Sequential(
            nn.Conv2d(score_in, score_hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(score_groups, score_hidden_channels),
            nn.SiLU(),
        )
        # One score per semantic class, per history frame, per pixel.
        self.class_score_head = nn.Conv2d(
            score_hidden_channels,
            self.num_classes,
            1,
            bias=True,
        )
        nn.init.zeros_(self.class_score_head.weight)
        nn.init.zeros_(self.class_score_head.bias)

        # Refine the frozen C-V6R attended history distribution with set evidence.
        # P_cur, P_hist_temp, P_set, (P_set-P_hist_temp), T, Q, history_available.
        refine_in = 4 * self.num_classes + 3
        refine_groups = 8 if refine_hidden_channels % 8 == 0 else 1
        self.refine_encoder = nn.Sequential(
            nn.Conv2d(refine_in, refine_hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(refine_groups, refine_hidden_channels),
            nn.SiLU(),
        )
        self.delta_head = nn.Conv2d(
            refine_hidden_channels,
            self.num_classes,
            1,
            bias=True,
        )
        self.attenuation_head = nn.Conv2d(
            refine_hidden_channels,
            1,
            1,
            bias=True,
        )
        # Exact C-V6R-E1 identity at initialization.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.attenuation_head.weight)
        nn.init.zeros_(self.attenuation_head.bias)

    def forward(
        self,
        current_probability,
        temporal_history_probability,
        history_probabilities,
        history_validities,
        transportability,
        memory_reliability,
    ):
        if len(history_probabilities) != self.history_length:
            raise ValueError("history_probabilities length must equal history_length")
        if len(history_validities) != self.history_length:
            raise ValueError("history_validities length must equal history_length")

        scores = []
        validity_channels = []
        for index, (history_probability, validity) in enumerate(
            zip(history_probabilities, history_validities)
        ):
            if history_probability.shape != current_probability.shape:
                raise ValueError("history probability must match current probability")
            if validity.ndim != 4 or validity.shape[1] != 1:
                raise ValueError("history validity must be Bx1xHxW")
            age = torch.full_like(
                validity,
                float(index + 1) / float(self.history_length),
            )
            # Hard-mask semantic content before the trainable encoder sees it.
            masked_history = history_probability * validity
            evidence = torch.cat(
                (
                    current_probability,
                    masked_history,
                    (masked_history - current_probability).abs(),
                    validity,
                    age,
                    transportability,
                    memory_reliability,
                ),
                dim=1,
            )
            scores.append(self.class_score_head(self.score_encoder(evidence)))
            validity_channels.append(validity)

        # B x K x C x H x W. Invalid histories receive exactly zero attention.
        score_tensor = torch.stack(scores, dim=1)
        valid_tensor = torch.stack(validity_channels, dim=1)
        masked_scores = score_tensor.masked_fill(valid_tensor <= 0.5, -1.0e4)
        class_attention = torch.softmax(masked_scores, dim=1)
        class_attention = class_attention * valid_tensor.to(class_attention.dtype)
        class_attention = class_attention / class_attention.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)

        set_probability = torch.zeros_like(current_probability)
        for index, history_probability in enumerate(history_probabilities):
            set_probability = set_probability + (
                class_attention[:, index] * history_probability
            )
        set_probability = set_probability.clamp_min(0.0)
        set_probability = set_probability / set_probability.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-8)

        history_available = (
            torch.cat(history_validities, dim=1).sum(dim=1, keepdim=True) > 0.5
        ).to(current_probability.dtype)
        refine_evidence = torch.cat(
            (
                current_probability,
                temporal_history_probability,
                set_probability,
                set_probability - temporal_history_probability,
                transportability,
                memory_reliability,
                history_available,
            ),
            dim=1,
        )
        encoded = self.refine_encoder(refine_evidence)
        delta_semantic = self.delta_head(encoded)
        attenuation_raw = self.attenuation_head(encoded)

        return {
            "delta_semantic": delta_semantic,
            "attenuation_raw": attenuation_raw,
            "set_probability": set_probability,
            "class_attention": class_attention,
            "history_available": history_available,
        }
