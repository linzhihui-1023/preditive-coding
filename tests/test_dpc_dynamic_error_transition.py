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
    prediction_error = torch.randn(2, 4, 5, 6)
    hidden = torch.randn_like(prediction_error)
    previous_dynamic = torch.randn_like(prediction_error)

    transitioned, dynamic_error, weights = module(
        prediction_error, hidden, previous_dynamic
    )
    expected_dynamic = (
        (module.sample_time / module.time_constant) * prediction_error
        + (
            1.0
            - module.error_gain * module.sample_time / module.time_constant
        )
        * previous_dynamic
    )
    expected_transition = sum(
        weights[:, index, None, None, None]
        * (hidden + module.residual_scale * basis(hidden))
        for index, basis in enumerate(module.bases)
    )

    assert module.sample_time == 0.1035
    assert module.time_constant == 0.5
    assert module.error_gain == 1.0
    assert abs(module.integration_factor - 0.207) < 1e-12
    assert abs(module.memory_factor - 0.793) < 1e-12
    assert torch.allclose(dynamic_error, expected_dynamic)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert torch.allclose(transitioned, expected_transition)

    _, _, zero_weights = module(
        prediction_error, hidden, previous_dynamic, zero_dynamic=True
    )
    expected_zero_weights = torch.softmax(module.controller.bias, dim=0).expand(2, -1)
    assert torch.allclose(zero_weights, expected_zero_weights)


def test_dynamic_error_uses_observation_minus_prediction():
    torch.manual_seed(2)
    module = DPCSemanticTemporalErrorCorrection(channels=4)
    observation = torch.randn(1, 4, 5, 6)
    predicted = torch.randn_like(observation)
    semantic_context = torch.randn_like(observation)
    hidden = torch.zeros_like(observation)
    previous_dynamic = torch.zeros_like(observation)

    values = module(
        observation,
        predicted,
        semantic_context,
        hidden,
        previous_dynamic,
    )

    expected_prediction_error = observation - predicted
    expected_dynamic = (
        module.error_state.transition.sample_time
        / module.error_state.transition.time_constant
    ) * expected_prediction_error
    assert torch.allclose(values["error"], expected_prediction_error)
    assert torch.allclose(values["dynamic_error"], expected_dynamic)


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


def test_dpc_adds_only_controller_and_depthwise_bases():
    baseline = SemanticTemporalErrorCorrection()
    dpc = DPCSemanticTemporalErrorCorrection()
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    dpc_parameters = sum(parameter.numel() for parameter in dpc.parameters())

    expected_per_layer = (128 * 3 + 3) + 3 * (128 * 3 * 3)
    assert dpc_parameters - baseline_parameters == expected_per_layer
