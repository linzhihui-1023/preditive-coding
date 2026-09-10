"""CPU/synthetic contracts for C-V15 Proposal-Conditioned Soft Acceptance."""

import inspect

import torch

from predify2021.mce_scores import (
    c_v15_proposal_conditioned_soft_acceptance_training as training,
)
from predify2021.mce_scores import (
    check_c_v14_post_writeback_reliability_acceptance_contracts as c_v14_contract,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v15_proposal_conditioned_soft_acceptance
    as c_v15_entry,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_post_writeback_reliability_feature_correction_aligned import (
    AlignedPostWritebackReliabilityFeatureCorrector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_proposal_conditioned_soft_acceptance import (
    ProposalConditionedSoftAcceptanceCorrector,
)


NUM_CLASSES = c_v14_contract.NUM_CLASSES
K = c_v14_contract.K
SEMANTIC = c_v14_contract.SEMANTIC
TEMPORAL = c_v14_contract.TEMPORAL
C4 = c_v14_contract.C4


def _aligned_cv14():
    return AlignedPostWritebackReliabilityFeatureCorrector(
        num_classes=NUM_CLASSES,
        history_length=K,
        semantic_channels=SEMANTIC,
        temporal_hidden_channels=TEMPORAL,
        host_channels=C4,
        residual_scale=training.RESIDUAL_SCALE,
    )


def _c_v15_from_synthetic_cv14():
    base = _aligned_cv14().eval()
    with torch.no_grad():
        base.writeback.output_projection.weight.normal_(0.0, 1e-3)
        base.reliability_head.weight.normal_(0.0, 0.02)
        base.reliability_head.bias.normal_(0.0, 0.05)
    model = ProposalConditionedSoftAcceptanceCorrector(
        num_classes=NUM_CLASSES,
        history_length=K,
        semantic_channels=SEMANTIC,
        temporal_hidden_channels=TEMPORAL,
        host_channels=C4,
        residual_scale=training.RESIDUAL_SCALE,
    )
    model.load_frozen_cv14_state_dict(base.state_dict())
    return base, model


def _check_zero_residual_reproduces_cv14():
    torch.manual_seed(0)
    base, model = _c_v15_from_synthetic_cv14()
    data = c_v14_contract._inputs(error_size=8, c4_size=4)
    with torch.no_grad():
        base_row = base(**data)
        row = model(**data)
    for key in ("reliability", "delta_c4", "corrected_c4", "bounded_semantic_delta_c4"):
        if not torch.allclose(row[key], base_row[key], atol=2e-6, rtol=2e-6):
            raise RuntimeError(
                f"C-V15 zero residual must reproduce aligned C-V14 for {key}"
            )
    if float(row["acceptance_residual_logit"].abs().max().item()) != 0.0:
        raise RuntimeError("C-V15 acceptance residual logit must initialize at exact zero")


def _check_only_new_acceptance_modules_trainable():
    _, model = _c_v15_from_synthetic_cv14()
    expected_prefixes = (
        "proposal_encoder.",
        "host_encoder.",
        "acceptance_fusion.",
        "acceptance_residual_head.",
    )
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("C-V15 has no trainable acceptance parameters")
    illegal = [name for name in trainable if not name.startswith(expected_prefixes)]
    if illegal:
        raise RuntimeError(f"Inherited C-V14 parameters remained trainable: {illegal}")
    for module_name, module in (
        ("semantic_encoder", model.semantic_encoder),
        ("temporal_pre", model.temporal_pre),
        ("reliability_head", model.reliability_head),
        ("writeback", model.writeback),
    ):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError(f"Frozen C-V14 module became trainable: {module_name}")


def _check_proposal_and_host_conditioning():
    torch.manual_seed(1)
    _, model = _c_v15_from_synthetic_cv14()
    model.train()
    with torch.no_grad():
        for module in (model.proposal_encoder, model.host_encoder, model.acceptance_fusion):
            for parameter in module.parameters():
                parameter.normal_(0.0, 0.02)
        model.acceptance_residual_head.weight.normal_(0.0, 0.02)
        model.acceptance_residual_head.bias.zero_()

    data = c_v14_contract._inputs(error_size=8, c4_size=4)
    with torch.no_grad():
        row = model(**data)
        baseline = model._proposal_conditioned_residual_logit(
            row["semantic_latent"],
            row["temporal_latent"],
            row["bounded_semantic_delta_c4"],
            row["c4_channel_rms"],
            data["current_c4"],
        )["acceptance_residual_logit"]
        changed_proposal = model._proposal_conditioned_residual_logit(
            row["semantic_latent"],
            row["temporal_latent"],
            row["bounded_semantic_delta_c4"] + 0.02,
            row["c4_channel_rms"],
            data["current_c4"],
        )["acceptance_residual_logit"]
        changed_host = model._proposal_conditioned_residual_logit(
            row["semantic_latent"],
            row["temporal_latent"],
            row["bounded_semantic_delta_c4"],
            row["c4_channel_rms"],
            data["current_c4"] + 0.1,
        )["acceptance_residual_logit"]
    if torch.allclose(baseline, changed_proposal):
        raise RuntimeError("C-V15 acceptance must respond to the specific Delta-c4 proposal")
    if torch.allclose(baseline, changed_host):
        raise RuntimeError("C-V15 acceptance must respond to current Host c4 context")
    normalized = row["normalized_proposal"]
    if float(normalized.min().item()) < -1.000001 or float(normalized.max().item()) > 1.000001:
        raise RuntimeError("Normalized proposal descriptor input must stay in [-1,1]")


def _check_soft_acceptance_composition():
    torch.manual_seed(2)
    _, model = _c_v15_from_synthetic_cv14()
    data = c_v14_contract._inputs(error_size=8, c4_size=4)
    with torch.no_grad():
        model.acceptance_residual_head.bias.fill_(-20.0)
        off = model(**data)
        model.acceptance_residual_head.bias.fill_(20.0)
        on = model(**data)
    if not torch.allclose(
        off["bounded_semantic_delta_c4"],
        on["bounded_semantic_delta_c4"],
        atol=1e-7,
        rtol=0.0,
    ):
        raise RuntimeError("C-V15 acceptance must not alter the frozen proposal")
    expected_on = on["bounded_semantic_delta_c4"] * on["reliability"]
    if not torch.allclose(on["delta_c4"], expected_on, atol=1e-7, rtol=1e-6):
        raise RuntimeError("C-V15 final Delta-c4 must equal g * bounded proposal")
    if float(off["delta_c4"].abs().max().item()) >= float(on["delta_c4"].abs().max().item()):
        raise RuntimeError("C-V15 soft acceptance failed to control proposal amplitude")


def _check_acceptance_bce_trains_only_new_path():
    torch.manual_seed(3)
    _, model = _c_v15_from_synthetic_cv14()
    model.train()
    with torch.no_grad():
        for module in (model.proposal_encoder, model.host_encoder, model.acceptance_fusion):
            for parameter in module.parameters():
                parameter.normal_(0.0, 0.02)
        model.acceptance_residual_head.weight.normal_(0.0, 0.02)

    row = model(**c_v14_contract._inputs(error_size=8, c4_size=4))
    target = torch.zeros((8, 8))
    target[:4] = 1.0
    supervised = torch.ones((8, 8), dtype=torch.bool)
    loss, probability = training.acceptance_bce_loss(
        model,
        row,
        (8, 8),
        target,
        supervised,
    )
    if probability is None:
        raise RuntimeError("C-V15 acceptance BCE did not return deployed probability")
    model.zero_grad(set_to_none=True)
    loss.backward()

    required = {
        "proposal_encoder": model.proposal_encoder[0].weight.grad,
        "host_encoder": model.host_encoder[0].weight.grad,
        "acceptance_fusion": model.acceptance_fusion[0].weight.grad,
        "acceptance_residual_head": model.acceptance_residual_head.weight.grad,
    }
    for name, gradient in required.items():
        if gradient is None or float(gradient.abs().sum().item()) <= 0.0:
            raise RuntimeError(f"C-V15 acceptance BCE did not train {name}")

    frozen = {
        "semantic_encoder": model.semantic_encoder[0].weight.grad,
        "temporal_pre": model.temporal_pre[0].weight.grad,
        "legacy_reliability_head": model.reliability_head.weight.grad,
        "writeback": model.writeback.output_projection.weight.grad,
    }
    for name, gradient in frozen.items():
        if gradient is not None and float(gradient.abs().sum().item()) > 0.0:
            raise RuntimeError(f"C-V15 acceptance BCE leaked into frozen {name}")


def _check_labels_unchanged():
    gt = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    host_logits = torch.full((1, NUM_CLASSES, 1, 4), -5.0)
    proposal_logits = torch.full_like(host_logits, -5.0)
    host_pred = torch.tensor([4, 1, 2, 4])
    proposal_pred = torch.tensor([0, 4, 2, 4])
    for x in range(4):
        host_logits[0, host_pred[x], 0, x] = 5.0
        proposal_logits[0, proposal_pred[x], 0, x] = 5.0
    target, supervised, beneficial, harmful = training.proposal_acceptance_targets(
        host_logits,
        proposal_logits,
        gt,
    )
    if beneficial.tolist() != [[True, False, False, False]]:
        raise RuntimeError("C-V15 changed C-V14 beneficial label definition")
    if harmful.tolist() != [[False, True, False, False]]:
        raise RuntimeError("C-V15 changed C-V14 harmful label definition")
    if supervised.tolist() != [[True, True, False, False]]:
        raise RuntimeError("C-V15 changed ignored acceptance pixels")
    if target.tolist() != [[1.0, 0.0, 0.0, 0.0]]:
        raise RuntimeError("C-V15 changed acceptance target values")


def _check_source_contract():
    model_source = inspect.getsource(ProposalConditionedSoftAcceptanceCorrector)
    training_source = inspect.getsource(training)
    entry_source = inspect.getsource(c_v15_entry)

    for token in (
        "normalized_proposal",
        "proposal_descriptor",
        "host_descriptor",
        "acceptance_residual_head",
        'base_reliability_c4 = row["reliability"].detach()',
        'delta_c4 = row["bounded_semantic_delta_c4"] * reliability_c4',
    ):
        if token not in model_source:
            raise RuntimeError(f"C-V15 model lost proposal-conditioning contract: {token}")

    for token in (
        'correction_row["reliability"]',
        "F.binary_cross_entropy(",
        "_base.acceptance_bce_loss = acceptance_bce_loss",
    ):
        if token not in training_source:
            raise RuntimeError(f"C-V15 training lost aligned acceptance contract: {token}")

    for token in (
        "C_V14_REQUIRED_EPOCH = 3",
        'architecture.get("aligned_reliability_revision", False)',
        "corrector.load_frozen_cv14_state_dict",
        "if parameter.requires_grad",
        '"rescue_ce": False',
        '"temporal_loss": False',
        '"raft_training": False',
    ):
        if token not in entry_source:
            raise RuntimeError(f"C-V15 entrypoint lost fixed experiment contract: {token}")


def main():
    _check_zero_residual_reproduces_cv14()
    _check_only_new_acceptance_modules_trainable()
    _check_proposal_and_host_conditioning()
    _check_soft_acceptance_composition()
    _check_acceptance_bce_trains_only_new_path()
    _check_labels_unchanged()
    _check_source_contract()

    _, model = _c_v15_from_synthetic_cv14()
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        {
            "passed": True,
            "architecture": "C-V15 Proposal-Conditioned Soft Acceptance",
            "source_proposal": "frozen aligned C-V14 E3",
            "acceptance_inputs": "temporal + semantic + normalized proposal + host c4",
            "zero_residual_start": "reproduces frozen C-V14 gate",
            "labels": "unchanged Host wrong/proposal correct vs Host correct/proposal wrong",
            "trainable_parameters": trainable,
            "fixed_protocol": True,
        }
    )


if __name__ == "__main__":
    main()
