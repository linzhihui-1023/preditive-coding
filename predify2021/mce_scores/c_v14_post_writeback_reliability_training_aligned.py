"""Aligned training wrapper for C-V14 post-writeback reliability acceptance.

中文：C-V14 回写后可靠性接受控制的空间对齐训练层。

This module preserves the original C-V14 train/eval loops and changes only the
acceptance-supervision mapping. The auxiliary BCE now follows the deployed gate
path exactly:

    detached temporal context -> reliability logits -> sigmoid
    -> resize to c4 -> resize to output -> BCE

The aligned model exposes ``correction_row['reliability']`` at c4 resolution,
so the existing train/eval diagnostics also measure the same deployed gate.
"""

import torch
from torch.nn import functional as F

from predify2021.mce_scores import c_v14_post_writeback_reliability_training as _base
from predify2021.mce_scores.c_v14_post_writeback_reliability_training import *  # noqa: F401,F403


def aligned_acceptance_probability(
    corrector,
    correction_row,
    output_size,
):
    """Reconstruct the deployed gate from detached temporal context.

    Gradients flow only into ``reliability_head`` because the temporal latent is
    detached inside ``reliability_logits_from_detached_context``.
    """
    aux_low_logit = corrector.reliability_logits_from_detached_context(
        correction_row["temporal_latent"]
    )
    aux_low_probability = torch.sigmoid(aux_low_logit)
    c4_size = tuple(correction_row["bounded_semantic_delta_c4"].shape[-2:])
    aux_c4_probability = F.interpolate(
        aux_low_probability,
        size=c4_size,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    aux_full_probability = F.interpolate(
        aux_c4_probability,
        size=tuple(output_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].clamp(0.0, 1.0)
    return aux_full_probability


def acceptance_bce_loss(corrector, correction_row, output_size, target, supervised):
    """Acceptance BCE on the same probability field used by deployed gating."""
    if int(supervised.sum().item()) == 0:
        zero = correction_row["reliability_logit"].sum() * 0.0
        return zero, None
    aux_full_probability = aligned_acceptance_probability(
        corrector,
        correction_row,
        output_size,
    )
    probability = aux_full_probability[supervised].clamp(1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(
        probability,
        target[supervised],
        reduction="mean",
    )
    return bce, aux_full_probability


# The original train_sequence resolves acceptance_bce_loss from its module
# globals at runtime. Replace only that global function; all other C-V14
# training/evaluation code and fixed protocol stay unchanged.
_base.acceptance_bce_loss = acceptance_bce_loss

# Explicit re-exports for the entrypoint and contracts.
train_sequence = _base.train_sequence
train_epoch = _base.train_epoch
evaluate = _base.evaluate
decode_post_writeback_feature_update = _base.decode_post_writeback_feature_update
protection_kl_loss = _base.protection_kl_loss
proposal_acceptance_targets = _base.proposal_acceptance_targets
host_feature_observation = _base.host_feature_observation
legacy_observation = _base.legacy_observation
build_prediction_error_and_temporal_state = _base.build_prediction_error_and_temporal_state
cv4_baseline_logits = _base.cv4_baseline_logits
formal_rescue_mask = _base.formal_rescue_mask
