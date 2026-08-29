import torch
from torch import nn

from predify2021.mce_scores.semantic_temporal_error_step import (
    DynamicTransitionState,
    detach_error_state,
    semantic_temporal_error_step,
    zero_dynamic_transition_state,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import (
    DPCSemanticTemporalErrorCorrection,
    DynamicErrorTransition,
    SemanticTemporalErrorCorrection,
)


def unified(value):
    return UnifiedFeatures(value, value, value, value)


def test_dynamic_error_controls_weighted_hidden_transition():
    torch.manual_seed(0)
    module = DynamicErrorTransition(channels=4)
    task_error = torch.randn(2, 4, 5, 6)
    hidden = torch.randn_like(task_error)
    previous_dynamic = torch.randn_like(task_error)

    transitioned, dynamic_error, weights = module(
        task_error, hidden, previous_dynamic
    )
    beta = module.beta
    expected_dynamic = beta * previous_dynamic + (1.0 - beta) * task_error
    expected_transition = sum(
        weights[:, index, None, None, None]
        * (hidden + module.residual_scale * basis(hidden))
        for index, basis in enumerate(module.bases)
    )

    assert torch.allclose(beta, torch.tensor(0.793), atol=1e-6)
    assert torch.allclose(dynamic_error, expected_dynamic)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert torch.allclose(transitioned, expected_transition)

    _, _, zero_weights = module(
        task_error, hidden, previous_dynamic, zero_dynamic=True
    )
    expected_zero_weights = torch.softmax(module.controller.bias, dim=0).expand(2, -1)
    assert torch.allclose(zero_weights, expected_zero_weights)


def test_dpc_state_shapes_detach_and_zero_response():
    torch.manual_seed(1)
    corrections = nn.ModuleList(
        [DPCSemanticTemporalErrorCorrection(), DPCSemanticTemporalErrorCorrection()]
    )
    observation_tensor = torch.randn(1, 128, 4, 5)
    observation = unified(observation_tensor)
    state = zero_dynamic_transition_state(observation)

    posterior, new_state, values = semantic_temporal_error_step(
        corrections,
        observation,
        observation,
        observation,
        state,
    )

    assert isinstance(new_state, DynamicTransitionState)
    assert all(value.shape == observation_tensor.shape for value in new_state.hidden)
    assert all(
        value.shape == observation_tensor.shape for value in new_state.dynamic_error
    )
    assert torch.count_nonzero(values["task_error_z1"]) == 0
    assert torch.count_nonzero(values["task_error_z4"]) == 0
    assert torch.count_nonzero(values["dynamic_error_z1"]) == 0
    assert torch.count_nonzero(values["dynamic_error_z4"]) == 0
    assert torch.allclose(posterior.z1, observation.z1)
    assert torch.allclose(posterior.z4, observation.z4)

    detached = detach_error_state(new_state)
    assert all(value.grad_fn is None for value in detached.hidden)
    assert all(value.grad_fn is None for value in detached.dynamic_error)


def test_dpc_adds_only_controller_beta_and_depthwise_bases():
    baseline = SemanticTemporalErrorCorrection()
    dpc = DPCSemanticTemporalErrorCorrection()
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    dpc_parameters = sum(parameter.numel() for parameter in dpc.parameters())

    expected_per_layer = 1 + (128 * 3 + 3) + 3 * (128 * 3 * 3)
    assert dpc_parameters - baseline_parameters == expected_per_layer
