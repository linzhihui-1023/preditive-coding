"""Hierarchical error-centric temporal selector for C-V6.

中文：分层误差中心时序选择器。

Structural split:
1. Stage-1 gate answers Current vs History.
   It sees all K validity-gated Prediction Errors so t-2..t-K-only rescue is
   observable, plus the explicit t-1 Dynamics Error and reliability evidence.
   Dynamics Error remains a temporal-persistence signal, not an age selector.
2. Stage-2 history selector answers t-1 / ... / t-K only after History is chosen.
   It uses the K validity-gated prediction errors and per-history reliability,
   but deliberately excludes Dynamics Error.

Raw Current / History semantic probabilities never enter either controller.
The final 1+K logits are composed so that argmax exactly implements the
hierarchy: first Current-vs-History, then the best valid history age.
"""

import torch
from torch import nn

from .semantic_recurrent_predictor import ConvGRUCell
from .task_space_multihypothesis_error_selector import signed_error_channels


class HierarchicalMultiHypothesisErrorSelector(nn.Module):
    """Two-stage error-only selector（两阶段仅误差选择器）."""

    def __init__(self, num_classes=19, history_length=4, hidden_channels=32):
        super().__init__()
        self.num_classes = int(num_classes)
        self.history_length = int(history_length)
        self.hidden_channels = int(hidden_channels)

        groups = 8 if self.hidden_channels % 8 == 0 else 1

        # Stage 1 / Current-vs-History.
        # All K errors are visible so a useful deep history cannot be hidden
        # merely because t-1 is unhelpful.  The single Dynamics Error remains
        # the explicit temporal state of strict t-1 Prediction Error.
        gate_semantic_channels = 2 * self.num_classes * (self.history_length + 1)
        # current margin + best history margin + T + Q + V1 + any-history-valid
        gate_scalar_channels = 6
        self.gate_pre = nn.Sequential(
            nn.Conv2d(
                gate_semantic_channels + gate_scalar_channels,
                self.hidden_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, self.hidden_channels),
            nn.SiLU(),
        )
        self.gate_recurrent = ConvGRUCell(self.hidden_channels, self.hidden_channels)
        self.gate_head = nn.Conv2d(self.hidden_channels, 2, 1, bias=True)

        # Stage 2 / History age selection.
        # Dynamics Error is intentionally excluded: it describes persistence of
        # the t-1 error and is not an age-selection signal for t-2..t-K.
        hist_semantic_channels = 2 * self.num_classes * self.history_length
        # K history margins + T + Q + K validity + K normalized ages.
        hist_scalar_channels = 3 * self.history_length + 2
        self.history_pre = nn.Sequential(
            nn.Conv2d(
                hist_semantic_channels + hist_scalar_channels,
                self.hidden_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, self.hidden_channels),
            nn.SiLU(),
        )
        self.history_recurrent = ConvGRUCell(
            self.hidden_channels,
            self.hidden_channels,
        )
        self.history_head = nn.Conv2d(
            self.hidden_channels,
            self.history_length,
            1,
            bias=True,
        )

        # E0 hard prediction falls back to Current. With all heads at zero,
        # Current and the best History tie at zero and output index 0 wins.
        for head in (self.gate_head, self.history_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _split_hidden(self, hidden):
        if hidden is None:
            return None, None
        if hidden.ndim != 4 or hidden.shape[1] != 2 * self.hidden_channels:
            raise ValueError(
                "hierarchical hidden must have 2*hidden_channels channels"
            )
        return hidden[:, : self.hidden_channels], hidden[:, self.hidden_channels :]

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

        gate_hidden, history_hidden = self._split_hidden(hidden)

        validity_stack = torch.cat(history_validities_low, dim=1)
        any_history_valid = (validity_stack.max(dim=1, keepdim=True).values > 0.5).to(
            current_margin.dtype
        )
        best_history_margin = torch.stack(history_margins, dim=1).max(dim=1).values
        gate_evidence = torch.cat(
            (
                *[signed_error_channels(error) for error in prediction_errors],
                signed_error_channels(dynamics_error),
                current_margin,
                best_history_margin,
                transportability_low,
                memory_reliability_low,
                history_validities_low[0],
                any_history_valid,
            ),
            dim=1,
        )
        gate_encoded = self.gate_pre(gate_evidence)
        gate_hidden = self.gate_recurrent(gate_encoded, gate_hidden)
        gate_logits = self.gate_head(gate_hidden)
        gate_logits = torch.cat(
            (
                gate_logits[:, :1],
                torch.where(
                    any_history_valid > 0.5,
                    gate_logits[:, 1:2],
                    torch.full_like(gate_logits[:, 1:2], -1.0e4),
                ),
            ),
            dim=1,
        )

        ages = [
            torch.full_like(
                current_margin,
                float(index + 1) / float(self.history_length),
            )
            for index in range(self.history_length)
        ]
        history_evidence = torch.cat(
            (
                *[signed_error_channels(error) for error in prediction_errors],
                *history_margins,
                transportability_low,
                memory_reliability_low,
                *history_validities_low,
                *ages,
            ),
            dim=1,
        )
        history_encoded = self.history_pre(history_evidence)
        history_hidden = self.history_recurrent(history_encoded, history_hidden)
        raw_history_logits = self.history_head(history_hidden)
        history_channels = []
        for index, validity in enumerate(history_validities_low):
            history_channels.append(
                torch.where(
                    validity > 0.5,
                    raw_history_logits[:, index : index + 1],
                    torch.full_like(
                        raw_history_logits[:, index : index + 1],
                        -1.0e4,
                    ),
                )
            )
        history_logits = torch.cat(history_channels, dim=1)

        # Normalize only the relative history-age score. The best valid history
        # receives zero offset; Stage 1 controls whether History beats Current,
        # while Stage 2 controls which valid age wins inside History.
        best_history_logit = history_logits.max(dim=1, keepdim=True).values
        history_relative = history_logits - best_history_logit
        history_relative = torch.where(
            any_history_valid > 0.5,
            history_relative,
            torch.full_like(history_relative, -1.0e4),
        )
        selector_logits = torch.cat(
            (
                gate_logits[:, :1],
                gate_logits[:, 1:2] + history_relative,
            ),
            dim=1,
        )

        return {
            "selector_logits": selector_logits,
            "gate_logits": gate_logits,
            "history_logits": history_logits,
            "hidden": torch.cat((gate_hidden, history_hidden), dim=1),
            "current_margin": current_margin,
            "history_margins": history_margins,
            "any_history_valid": any_history_valid,
        }
