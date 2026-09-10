"""Synthetic contract for C-V15 Host/C-V4 dual-reference diagnostics.

This checker does not change training labels. It verifies that validation can
independently label the same frozen semantic proposal relative to Host and C-V4.
"""

import inspect

import torch

from predify2021.mce_scores import (
    c_v15_proposal_conditioned_soft_acceptance_training as training,
)


NUM_CLASSES = training.NUM_CLASSES


def _logits(predictions):
    logits = torch.full((1, NUM_CLASSES, 1, len(predictions)), -5.0)
    for x, cls in enumerate(predictions):
        logits[0, cls, 0, x] = 5.0
    return logits


def _check_reference_targets_are_independent():
    # GT        : [0,1,2,3]
    # Host      : [4,1,2,4] -> proposal rescue at x0, damage at x1
    # C-V4      : [0,4,2,3] -> proposal damage at x0, rescue at x1
    # Proposal  : [0,4,2,4]
    gt = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    host = _logits([4, 1, 2, 4])
    c_v4 = _logits([0, 4, 2, 3])
    proposal = _logits([0, 4, 2, 4])

    host_positive, host_negative = training.reference_acceptance_targets(
        host, proposal, gt
    )
    cv4_positive, cv4_negative = training.reference_acceptance_targets(
        c_v4, proposal, gt
    )

    if host_positive.tolist() != [[True, False, False, False]]:
        raise RuntimeError("Host-reference positive target is wrong")
    if host_negative.tolist() != [[False, True, False, False]]:
        raise RuntimeError("Host-reference negative target is wrong")
    if cv4_positive.tolist() != [[False, True, False, False]]:
        raise RuntimeError("C-V4-reference positive target is wrong")
    if cv4_negative.tolist() != [[True, False, False, True]]:
        raise RuntimeError("C-V4-reference negative target is wrong")


def _check_training_target_stays_host_relative():
    source = inspect.getsource(training)
    required = (
        "Training supervision deliberately remains Host-relative",
        "reference_acceptance_targets(c_v4_logits.detach(), proposal_logits, gt_gpu)",
        '"cv4_reference_labels_backpropagated": False',
        '"training_acceptance_reference": "Host"',
        '"damage_rescue_reference": "frozen C-V4"',
        '"host_reference_separation_gain_vs_source_cv14"',
        '"cv4_reference_separation_gain_vs_source_cv14"',
    )
    for token in required:
        if token not in source:
            raise RuntimeError(f"C-V15 dual-reference diagnostic contract lost: {token}")

    # The actual training label must still come from the inherited C-V14 target.
    if "_base.proposal_acceptance_targets" not in source:
        # Re-export below is also accepted as proof that the inherited target is used.
        if "proposal_acceptance_targets = _base.proposal_acceptance_targets" not in source:
            raise RuntimeError("C-V15 no longer exposes the unchanged Host-relative target")


def main():
    _check_reference_targets_are_independent()
    _check_training_target_stays_host_relative()
    print(
        {
            "passed": True,
            "training_reference": "Host",
            "validation_secondary_reference": "frozen C-V4",
            "secondary_reference_backpropagated": False,
            "reported_primary_diagnostics": [
                "host_reference_reliability_separation",
                "cv4_reference_reliability_separation",
                "host_reference_separation_gain_vs_source_cv14",
                "cv4_reference_separation_gain_vs_source_cv14",
                "damage_rescue_ratio",
            ],
        }
    )


if __name__ == "__main__":
    main()
