"""C-V7 shared multi-hypothesis prediction-error utility estimator.

中文：C-V7 多假设预测误差共享效用估计器。

The controller no longer solves a K+1-way class-imbalanced classification
problem.  It estimates the relative utility of each historical prediction
against the frozen current C-V3 prediction:

    u_k = Utility(e_k, H_error, S_current, reliability_k, age_k)

Current has fixed utility zero.  Historical candidate k is selected only when
its estimated utility is positive and maximal.

Cross-frame semantic content still enters the controller only through the
validity-gated prediction error e_k.  A compact CURRENT semantic state is
allowed because predictive-coding error must be interpreted relative to the
state being corrected; raw historical probabilities are never supplied to the
utility scorer.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_recurrent_predictor import ConvGRUCell
from .task_space_multihypothesis_error_selector import signed_error_channels


class MultiHypothesisErrorUtilityEstimator(nn.Module):
    """Shared candidate utility estimator（共享候选效用估计器）.

    Architecture:
      1. current semantic probability -> compact current state S_t;
      2. all prediction errors + Dynamics Error -> recurrent Error State H_t;
      3. one shared scorer evaluates every historical hypothesis independently;
      4. Current utility is fixed at zero, so no separate correction gate exists.
    """

    def __init__(
        self,
        num_classes=19,
        history_length=4,
        hidden_channels=32,
        current_state_channels=16,
        scorer_channels=32,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.history_length = int(history_length)
        self.hidden_channels = int(hidden_channels)
        self.current_state_channels = int(current_state_channels)
        self.scorer_channels = int(scorer_channels)

        current_groups = 8 if self.current_state_channels % 8 == 0 else 1
        hidden_groups = 8 if self.hidden_channels % 8 == 0 else 1
        scorer_groups = 8 if self.scorer_channels % 8 == 0 else 1

        # Current representation is compacted before it reaches the candidate
        # scorer. Historical semantic probability is never passed here.
        self.current_state_encoder = nn.Sequential(
            nn.Conv2d(
                self.num_classes,
                self.current_state_channels,
                1,
                bias=False,
            ),
            nn.GroupNorm(current_groups, self.current_state_channels),
            nn.SiLU(),
        )

        # K hypothesis errors + one explicit Dynamics Error, all sign-preserving.
        signed_error_channels_total = 2 * self.num_classes * (
            self.history_length + 1
        )
        # compact current state + current margin + T + Q + K validity maps.
        context_scalar_channels = 1 + 2 + self.history_length
        self.context_input_channels = (
            signed_error_channels_total
            + self.current_state_channels
            + context_scalar_channels
        )
        self.context_pre = nn.Sequential(
            nn.Conv2d(
                self.context_input_channels,
                self.hidden_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(hidden_groups, self.hidden_channels),
            nn.SiLU(),
        )
        self.recurrent = ConvGRUCell(self.hidden_channels, self.hidden_channels)

        # The SAME scorer is reused for t-1..t-K.  This prevents sparse deep
        # history examples from having to train independent class heads.
        self.candidate_input_channels = (
            2 * self.num_classes  # candidate-specific signed prediction error
            + self.hidden_channels
            + self.current_state_channels
            + 6  # current margin, history margin, T, Q, validity, normalized age
        )
        self.candidate_pre = nn.Sequential(
            nn.Conv2d(
                self.candidate_input_channels,
                self.scorer_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(scorer_groups, self.scorer_channels),
            nn.SiLU(),
        )
        # Kept under the selector_head name so the existing zero-step invariant
        # checker can verify exact Current fallback at initialization.
        self.selector_head = nn.Conv2d(self.scorer_channels, 1, 1, bias=True)
        nn.init.zeros_(self.selector_head.weight)
        nn.init.zeros_(self.selector_head.bias)

    def forward(
        self,
        prediction_errors,
        dynamics_error,
        current_probability,
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
        if current_probability.shape[1] != self.num_classes:
            raise ValueError("current_probability must have num_classes channels")
        if dynamics_error.shape[1] != self.num_classes:
            raise ValueError("dynamics_error must have num_classes channels")

        spatial = tuple(current_probability.shape[-2:])
        if tuple(dynamics_error.shape[-2:]) != spatial:
            raise ValueError("dynamics_error spatial size mismatch")
        for error in prediction_errors:
            if error.shape[1] != self.num_classes or tuple(error.shape[-2:]) != spatial:
                raise ValueError("prediction error shape mismatch")

        # Upstream C-V3 and the candidate bank are frozen.  Explicit detach also
        # preserves that boundary if the upstream graph is changed later.
        current_probability = current_probability.detach()
        current_state = self.current_state_encoder(current_probability)

        signed_errors = [signed_error_channels(error) for error in prediction_errors]
        context = torch.cat(
            [
                *signed_errors,
                signed_error_channels(dynamics_error),
                current_state,
                current_margin,
                transportability_low,
                memory_reliability_low,
                *history_validities_low,
            ],
            dim=1,
        )
        if context.shape[1] != self.context_input_channels:
            raise RuntimeError(
                f"utility context channels mismatch: {context.shape[1]} "
                f"!= {self.context_input_channels}"
            )
        encoded = self.context_pre(context)
        hidden = self.recurrent(encoded, hidden)

        utilities = []
        for index in range(self.history_length):
            age = float(index + 1) / float(self.history_length)
            age_map = torch.full_like(current_margin, age)
            candidate = torch.cat(
                [
                    signed_errors[index],
                    hidden,
                    current_state,
                    current_margin,
                    history_margins[index],
                    transportability_low,
                    memory_reliability_low,
                    history_validities_low[index],
                    age_map,
                ],
                dim=1,
            )
            if candidate.shape[1] != self.candidate_input_channels:
                raise RuntimeError(
                    f"utility candidate channels mismatch: {candidate.shape[1]} "
                    f"!= {self.candidate_input_channels}"
                )
            utilities.append(self.selector_head(self.candidate_pre(candidate)))

        candidate_utilities = torch.cat(utilities, dim=1)
        current_utility = torch.zeros_like(candidate_utilities[:, :1])
        selector_logits = torch.cat((current_utility, candidate_utilities), dim=1)
        return {
            "selector_logits": selector_logits,
            "candidate_utilities": candidate_utilities,
            "hidden": hidden,
            "current_state": current_state,
            "current_margin": current_margin,
            "history_margins": history_margins,
        }


def balanced_utility_regression_loss(
    predicted_utility,
    target_utility,
    valid_mask,
    neutral_delta=0.10,
    neutral_weight=0.10,
):
    """Balanced Smooth-L1 utility loss（分组平衡效用回归损失）.

    Positive-gain and negative-gain hypotheses each receive equal aggregate
    weight regardless of their natural frequency.  Near-zero hypotheses are a
    low-weight neutral group, so abundant Current-like examples cannot dominate
    optimization.
    """
    if predicted_utility.shape != target_utility.shape:
        raise ValueError("predicted_utility and target_utility must match")
    if valid_mask.shape != predicted_utility.shape:
        raise ValueError("valid_mask must match utility tensors")

    valid = valid_mask.bool()
    positive = valid & (target_utility > float(neutral_delta))
    negative = valid & (target_utility < -float(neutral_delta))
    neutral = valid & ~(positive | negative)

    groups = (
        (positive, 0.5),
        (negative, 0.5),
        (neutral, float(neutral_weight)),
    )
    terms = []
    active_weight = 0.0
    counts = {}
    for name, (mask, weight) in zip(("positive", "negative", "neutral"), groups):
        count = int(mask.sum().item())
        counts[name] = count
        if count == 0 or weight <= 0.0:
            continue
        term = F.smooth_l1_loss(
            predicted_utility[mask],
            target_utility[mask],
            reduction="mean",
        )
        terms.append(float(weight) * term)
        active_weight += float(weight)

    if not terms:
        return predicted_utility.sum() * 0.0, counts
    return torch.stack(terms).sum() / max(active_weight, 1e-12), counts


def semantic_first_temporal_tiebreak_loss(
    candidate_utilities,
    semantic_gains,
    valid_mask,
    temporal_matches,
    temporal_valid,
    semantic_tie_delta=0.25,
    rank_margin=0.05,
):
    """Temporal ranking only when semantic utility is effectively tied.

    Includes Current as utility zero.  Temporal evidence is never allowed to
    override a semantic-gain difference larger than semantic_tie_delta.
    """
    if candidate_utilities.shape != semantic_gains.shape:
        raise ValueError("candidate utilities and semantic gains must match")
    if valid_mask.shape != candidate_utilities.shape:
        raise ValueError("valid_mask must match candidate utilities")
    expected_temporal_channels = candidate_utilities.shape[1] + 1
    if temporal_matches.ndim != 4 or temporal_matches.shape[1] != expected_temporal_channels:
        raise ValueError("temporal_matches must contain Current plus all histories")
    if temporal_matches.shape[0] != candidate_utilities.shape[0] or temporal_matches.shape[-2:] != candidate_utilities.shape[-2:]:
        raise ValueError("temporal_matches batch/spatial size mismatch")
    if temporal_valid.ndim == 2:
        temporal_valid = temporal_valid.unsqueeze(0)
    if temporal_valid.ndim == 3:
        temporal_valid = temporal_valid.unsqueeze(1)
    if temporal_valid.ndim != 4 or temporal_valid.shape[1] != 1:
        raise ValueError("temporal_valid must be [N,1,H,W] or [N,H,W]")

    current_utility = torch.zeros_like(candidate_utilities[:, :1])
    current_gain = torch.zeros_like(semantic_gains[:, :1])
    current_match = temporal_matches[:, :1]
    history_match = temporal_matches[:, 1:]
    if history_match.shape != candidate_utilities.shape:
        raise ValueError("temporal_matches history channels mismatch")

    utilities = torch.cat((current_utility, candidate_utilities), dim=1)
    gains = torch.cat((current_gain, semantic_gains), dim=1)
    current_valid = temporal_valid.expand(-1, 1, -1, -1).bool()
    all_valid = torch.cat((current_valid, valid_mask.bool() & current_valid), dim=1)

    pair_losses = []
    pair_count = 0
    channels = utilities.shape[1]
    for left in range(channels):
        for right in range(left + 1, channels):
            pair_valid = all_valid[:, left] & all_valid[:, right]
            tied = (gains[:, left] - gains[:, right]).abs() <= float(
                semantic_tie_delta
            )
            left_match = (
                current_match[:, 0]
                if left == 0
                else history_match[:, left - 1]
            )
            right_match = (
                current_match[:, 0]
                if right == 0
                else history_match[:, right - 1]
            )
            preference = left_match.to(torch.int8) - right_match.to(torch.int8)
            informative = pair_valid & tied & (preference != 0)
            count = int(informative.sum().item())
            if count == 0:
                continue
            # preference > 0 means left should have larger utility.
            signed_difference = (
                utilities[:, left] - utilities[:, right]
            ) * preference.to(utilities.dtype)
            pair_losses.append(
                F.relu(float(rank_margin) - signed_difference[informative]).mean()
            )
            pair_count += count

    if not pair_losses:
        return candidate_utilities.sum() * 0.0, 0
    return torch.stack(pair_losses).mean(), pair_count
