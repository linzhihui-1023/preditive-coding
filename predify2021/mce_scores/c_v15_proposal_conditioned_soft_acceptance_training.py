"""Training/evaluation wrapper for C-V15 proposal-conditioned soft acceptance.

中文：C-V15 修正提案条件化软接受训练与评测层。

The fixed C-V14 sequence construction, losses, K=4 history, C-V4 protection,
and metrics remain unchanged. C-V15 changes only the acceptance probability
used by the same beneficial/harmful labels. The deployed C-V15 gate already
lives at c4 resolution, therefore supervision follows exactly:

    proposal-conditioned reliability_c4 -> output resize -> BCE.

Because the inherited C-V14 proposal/temporal core is frozen, gradients from
CE, protection KL, and acceptance BCE update only the new C-V15 acceptance
modules (proposal encoder, host encoder, fusion, and residual head).
"""

from torch.nn import functional as F

from predify2021.mce_scores import (
    c_v14_post_writeback_reliability_training as _base,
)
from predify2021.mce_scores.c_v14_post_writeback_reliability_training import *  # noqa: F401,F403


def proposal_conditioned_acceptance_probability(correction_row, output_size):
    """Return the exact deployed C-V15 gate resized from c4 to output space."""
    return F.interpolate(
        correction_row["reliability"],
        size=tuple(output_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].clamp(0.0, 1.0)


def acceptance_bce_loss(corrector, correction_row, output_size, target, supervised):
    """Beneficial/harmful BCE on the deployed proposal-conditioned gate."""
    if int(supervised.sum().item()) == 0:
        zero = correction_row["reliability_logit"].sum() * 0.0
        return zero, None
    probability_full = proposal_conditioned_acceptance_probability(
        correction_row,
        output_size,
    )
    probability = probability_full[supervised].clamp(1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(
        probability,
        target[supervised],
        reduction="mean",
    )
    return bce, probability_full


# C-V14 train/eval resolves this symbol from its own module globals.
# Replace only the acceptance loss; sequence/history/metrics stay unchanged.
_base.acceptance_bce_loss = acceptance_bce_loss


def _add_acceptance_diagnostics(row):
    if row is None:
        return row
    row["reliability_separation"] = (
        row.get("positive_reliability_mean", 0.0)
        - row.get("negative_reliability_mean", 0.0)
    )
    row["damage_rescue_ratio"] = (
        row.get("baseline_correct_damaged", 0)
        / max(row.get("rescue_recovered", 0), 1)
    )
    return row


def train_sequence(*args, **kwargs):
    return _add_acceptance_diagnostics(_base.train_sequence(*args, **kwargs))


def train_epoch(*args, **kwargs):
    return _add_acceptance_diagnostics(_base.train_epoch(*args, **kwargs))


def evaluate(*args, **kwargs):
    metrics, diagnostics = _base.evaluate(*args, **kwargs)
    metrics["c_v15"] = metrics.pop("c_v14")
    _add_acceptance_diagnostics(diagnostics)
    diagnostics.update(
        {
            "acceptance_semantics": "proposal-conditioned soft acceptance coefficient",
            "acceptance_inputs": (
                "frozen temporal latent + frozen semantic latent + normalized "
                "bounded Delta-c4 descriptor + current c4 descriptor"
            ),
            "acceptance_supervision_mapping": (
                "deployed reliability_c4 -> output resize -> beneficial/harmful BCE"
            ),
            "cv14_proposal_generator_frozen": True,
            "cv14_reliability_prior_frozen": True,
        }
    )
    return metrics, diagnostics


# Explicit fixed-protocol re-exports.
decode_post_writeback_feature_update = _base.decode_post_writeback_feature_update
protection_kl_loss = _base.protection_kl_loss
proposal_acceptance_targets = _base.proposal_acceptance_targets
host_feature_observation = _base.host_feature_observation
legacy_observation = _base.legacy_observation
build_prediction_error_and_temporal_state = _base.build_prediction_error_and_temporal_state
cv4_baseline_logits = _base.cv4_baseline_logits
formal_rescue_mask = _base.formal_rescue_mask
