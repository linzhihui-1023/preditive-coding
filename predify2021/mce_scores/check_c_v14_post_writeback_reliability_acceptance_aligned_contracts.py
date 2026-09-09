"""CPU/synthetic contracts for the aligned C-V14 reliability path."""

import inspect

import torch
from torch.nn import functional as F

from predify2021.mce_scores import (
    c_v14_post_writeback_reliability_training as base_training,
)
from predify2021.mce_scores import (
    c_v14_post_writeback_reliability_training_aligned as training,
)
from predify2021.mce_scores import (
    check_c_v14_post_writeback_reliability_acceptance_contracts as base_contract,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v14_post_writeback_reliability_acceptance_aligned
    as c_v14_aligned,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_post_writeback_reliability_feature_correction_aligned import (
    AlignedPostWritebackReliabilityFeatureCorrector,
)


# Reuse the original C-V14 behavioral contracts with aligned dependencies.
base_contract.training = training
base_contract.PostWritebackReliabilityFeatureCorrector = (
    AlignedPostWritebackReliabilityFeatureCorrector
)


def _check_acceptance_matches_deployed_reliability():
    model = base_contract._model().train()
    with torch.no_grad():
        model.writeback.output_projection.weight.normal_(0.0, 1e-3)
        model.reliability_head.weight.normal_(0.0, 0.05)
        model.reliability_head.bias.normal_(0.0, 0.1)

    row = model(**base_contract._inputs(error_size=8, c4_size=4))
    output_size = (11, 13)
    auxiliary = training.aligned_acceptance_probability(
        model,
        row,
        output_size,
    )
    deployed_full = F.interpolate(
        row["reliability"],
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    if not torch.allclose(auxiliary, deployed_full, atol=1e-6, rtol=1e-6):
        raise RuntimeError(
            "Acceptance supervision probability must equal deployed "
            "reliability_c4 -> full probability field"
        )

    low_to_full_direct = F.interpolate(
        row["reliability_low"],
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    # With nontrivial spatial resizing, the contract should distinguish the old
    # low->full diagnostic path from the deployed low->c4->full path.
    if torch.allclose(auxiliary, low_to_full_direct, atol=1e-8, rtol=1e-8):
        raise RuntimeError(
            "Synthetic alignment case failed to distinguish low->full from "
            "deployed low->c4->full mapping"
        )


def _check_aligned_source_contract():
    aligned_training_source = inspect.getsource(training)
    aligned_entry_source = inspect.getsource(c_v14_aligned)
    aligned_model_source = inspect.getsource(AlignedPostWritebackReliabilityFeatureCorrector)
    base_helper_source = inspect.getsource(base_training)

    required_aligned_training = (
        "aux_low_probability = torch.sigmoid(aux_low_logit)",
        'c4_size = tuple(correction_row["bounded_semantic_delta_c4"].shape[-2:])',
        "aux_c4_probability = F.interpolate(",
        "aux_full_probability = F.interpolate(",
        "F.binary_cross_entropy(",
        "_base.acceptance_bce_loss = acceptance_bce_loss",
    )
    for token in required_aligned_training:
        if token not in aligned_training_source:
            raise RuntimeError(f"Aligned C-V14 training lost contract: {token}")
    if "binary_cross_entropy_with_logits" in aligned_training_source:
        raise RuntimeError(
            "Aligned C-V14 must not supervise interpolate(logit) with BCEWithLogits"
        )

    required_aligned_model = (
        'row["reliability_low"] = row["reliability"]',
        'row["reliability"] = row["reliability_c4"]',
    )
    for token in required_aligned_model:
        if token not in aligned_model_source:
            raise RuntimeError(f"Aligned C-V14 model lost contract: {token}")

    required_entry = (
        "_base_entry.training = training",
        "AlignedPostWritebackReliabilityFeatureCorrector",
    )
    for token in required_entry:
        if token not in aligned_entry_source:
            raise RuntimeError(f"Aligned C-V14 entry lost contract: {token}")

    # The original training loop remains the fixed protocol and must still own
    # the three losses, non-autoregressive history, and no Rescue CE.
    for token in (
        "PROTECTION_LOSS_WEIGHT * protect_kl",
        "ACCEPTANCE_LOSS_WEIGHT * accept_bce",
        'raw_history.insert(0, c_v3_logits.detach())',
    ):
        if token not in base_helper_source:
            raise RuntimeError(f"Base C-V14 fixed protocol drifted: {token}")
    for token in ("RESCUE_LOSS_WEIGHT", "rescue_ce ="):
        if token in base_helper_source:
            raise RuntimeError(f"Base C-V14 retained forbidden behavior: {token}")


def main():
    # Original behavioral contracts, except its old source-string contract.
    base_contract._check_predictive_coding_boundary()
    base_contract._check_zero_step()
    base_contract._check_post_writeback_control_and_bound()
    base_contract._check_deep_history_and_temporal_conditioning()
    base_contract._check_deployed_gradient_path()
    base_contract._check_acceptance_targets_and_detached_aux_gradient()
    base_contract._check_protection_supervision()
    base_contract._check_model_selection()

    _check_acceptance_matches_deployed_reliability()
    _check_aligned_source_contract()

    model = base_contract._model()
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print({
        "passed": True,
        "architecture": "C-V14 post-writeback reliability acceptance",
        "acceptance_mapping": "sigmoid(low logit) -> c4 resize -> full resize -> BCE",
        "deployment_mapping": "sigmoid(low logit) -> c4 resize -> bounded Delta-c4 gating",
        "diagnostic_mapping": "deployed reliability_c4 -> full resize",
        "acceptance_aux_gradient": "reliability_head only",
        "trainable_parameters": trainable,
        "fixed_protocol": True,
    })


if __name__ == "__main__":
    main()
