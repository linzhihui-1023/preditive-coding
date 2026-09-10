"""CPU contracts for C-V16-A zero-initialization and L_seg-only training."""

import inspect

import torch
from torch.nn import functional as F

from predify2021.mce_scores import (
    check_c_v14_post_writeback_reliability_acceptance_contracts as _cv14_contract,
)
from predify2021.mce_scores import c_v16_a_fresh_gate_lseg_training as training
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_c_v16_a_fresh_gate import (
    CV16AFreshGateCorrector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_proposal_conditioned_soft_acceptance import (
    ProposalConditionedSoftAcceptanceCorrector,
)


def _source_model():
    torch.manual_seed(19)
    model = ProposalConditionedSoftAcceptanceCorrector(
        num_classes=_cv14_contract.NUM_CLASSES,
        history_length=_cv14_contract.K,
        semantic_channels=_cv14_contract.SEMANTIC,
        temporal_hidden_channels=_cv14_contract.TEMPORAL,
        host_channels=_cv14_contract.C4,
        residual_scale=0.10,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.03)
    return model


def _fresh_model(source):
    model = CV16AFreshGateCorrector(
        num_classes=_cv14_contract.NUM_CLASSES,
        history_length=_cv14_contract.K,
        semantic_channels=_cv14_contract.SEMANTIC,
        temporal_hidden_channels=_cv14_contract.TEMPORAL,
        host_channels=_cv14_contract.C4,
        residual_scale=0.10,
    )
    model.load_c_v15_state_dict(source.state_dict())
    return model


def main():
    source = _source_model()
    fresh = _fresh_model(source)
    state = fresh.contract_state()

    if not state["checkpoint_loaded_before_gate_reset"]:
        raise RuntimeError("C-V16-A did not record checkpoint-before-reset ordering")
    if not state["gate_initialized_after_checkpoint_loading"]:
        raise RuntimeError("C-V16-A Gate was not initialized after checkpoint load")
    if not state["proposal_generator_frozen"]:
        raise RuntimeError("C-V16-A Proposal generator is not fully frozen")
    if state["proposal_parameters_receive_gradient"]:
        raise RuntimeError("C-V16-A Proposal parameters remain trainable")

    for name in fresh.gate_module_names():
        before_state = getattr(source, name).state_dict()
        after_state = getattr(fresh, name).state_dict()
        if all(torch.equal(before_state[key], after_state[key]) for key in before_state):
            raise RuntimeError(f"C-V16-A stale C-V15 Gate weights survived: {name}")

    data = _cv14_contract._inputs(error_size=8, c4_size=4)
    fresh.train()
    with torch.no_grad():
        row = fresh(**data)
    if abs(float(row["reliability"].mean().item()) - 0.5) > 0.02:
        raise RuntimeError("Fresh Gate does not start in a neutral reliability state")

    # A frozen decoder probe is sufficient to verify the actual L_seg path.
    decoder_weight = torch.randn(
        _cv14_contract.NUM_CLASSES,
        _cv14_contract.C4,
        1,
        1,
    )
    fresh.zero_grad(set_to_none=True)
    row = fresh(**data)
    logits = F.conv2d(row["corrected_c4"], decoder_weight)
    target = torch.zeros((1, logits.shape[-2], logits.shape[-1]), dtype=torch.long)
    l_seg = F.cross_entropy(logits, target)
    l_seg.backward()

    for name in fresh.gate_module_names():
        gradients = [
            parameter.grad
            for parameter in getattr(fresh, name).parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError(f"L_seg did not reach Gate module: {name}")
        if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError(f"Non-finite Gate gradient: {name}")
        if sum(float(gradient.abs().sum().item()) for gradient in gradients) <= 0.0:
            raise RuntimeError(f"Zero Gate gradient: {name}")

    for name in fresh.proposal_module_names():
        if any(
            parameter.grad is not None
            for parameter in getattr(fresh, name).parameters()
        ):
            raise RuntimeError(f"Proposal received an L_seg gradient: {name}")

    training_source = inspect.getsource(training)
    entry_source = inspect.getsource(
        __import__(
            "predify2021.mce_scores.train_kitti_step_task_space_prior_c_v16_a_fresh_gate_lseg",
            fromlist=["main"],
        )
    )
    for token in (
        '"training_objective": "L_seg only"',
        '"l_accept_backprop": False',
        '"l_protect_backprop": False',
        '"gt_beneficial_harmful_backprop": False',
        "l_seg.backward()",
    ):
        if token not in training_source:
            raise RuntimeError(f"C-V16-A training contract missing: {token}")
    for token in (
        "load_c_v15_state_dict",
        "C-V16-A output must be independent of C-V15 output",
        "c_v15_files_modified",
        "c_v16_a_output_independent",
    ):
        if token not in entry_source:
            raise RuntimeError(f"C-V16-A entry contract missing: {token}")

    print(
        {
            "passed": True,
            "proposal_generator_frozen": True,
            "gate_initialized_after_checkpoint_loading": True,
            "bce_trained_gate_weights_inherited": False,
            "fresh_gate_initialization": state["gate_initialization"],
            "training_objective": "L_seg only",
            "l_seg_gate_gradient": True,
            "l_seg_gate_gradient_finite": True,
            "l_seg_gate_gradient_non_zero": True,
            "proposal_gradient": False,
            "c_v15_write_scope": "independent/read-only",
        }
    )


if __name__ == "__main__":
    main()
