import torch
from torch import nn

from predify2021.mce_scores.semantic_temporal_error_step import (
    DYNAMIC_ERROR_GAIN,
    DYNAMIC_ERROR_SAMPLE_TIME,
    DYNAMIC_ERROR_TIME_CONSTANT,
    SemanticTemporalState,
    detach_error_state,
    semantic_temporal_error_step,
    zero_semantic_temporal_state,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import (
    SemanticTemporalErrorCorrection,
)
from predify2021.model_factory.targetflow.core import (
    build_temporal_prediction_error_state,
)


def unified(value):
    return UnifiedFeatures(value, value, value, value)


def build_corrections(channels):
    return nn.ModuleList(
        [
            SemanticTemporalErrorCorrection(channels),
            SemanticTemporalErrorCorrection(channels),
        ]
    )


def test_dynamic_error_uses_observation_minus_prediction_and_full_formula():
    torch.manual_seed(0)
    corrections = build_corrections(4)
    observation_tensor = torch.randn(1, 4, 5, 6)
    prediction_tensor = torch.randn_like(observation_tensor)
    semantic_tensor = torch.randn_like(observation_tensor)
    observation = unified(observation_tensor)
    prediction = unified(prediction_tensor)
    semantic = unified(semantic_tensor)
    state = zero_semantic_temporal_state(observation)

    _, new_state, values = semantic_temporal_error_step(
        corrections,
        observation,
        prediction,
        semantic,
        state,
    )

    expected_error = observation_tensor - prediction_tensor
    expected_dynamic = build_temporal_prediction_error_state(
        expected_error,
        torch.zeros_like(expected_error),
        sample_time=DYNAMIC_ERROR_SAMPLE_TIME,
        time_constant=DYNAMIC_ERROR_TIME_CONSTANT,
        error_gain=DYNAMIC_ERROR_GAIN,
    )

    assert torch.allclose(values["error_z1"], expected_error)
    assert torch.allclose(values["dynamic_error_z1"], expected_dynamic)
    assert torch.allclose(new_state.dynamic_error[0], expected_dynamic)
    assert DYNAMIC_ERROR_SAMPLE_TIME / DYNAMIC_ERROR_TIME_CONSTANT == 0.207
    assert (
        1.0
        - DYNAMIC_ERROR_GAIN
        * DYNAMIC_ERROR_SAMPLE_TIME
        / DYNAMIC_ERROR_TIME_CONSTANT
        == 0.793
    )


def test_dynamic_error_is_tracked_but_does_not_change_correction():
    torch.manual_seed(1)
    corrections = build_corrections(4)
    observation_tensor = torch.randn(1, 4, 5, 6)
    prediction_tensor = torch.randn_like(observation_tensor)
    semantic_tensor = torch.randn_like(observation_tensor)
    observation = unified(observation_tensor)
    prediction = unified(prediction_tensor)
    semantic = unified(semantic_tensor)
    zero_state = zero_semantic_temporal_state(observation)
    different_dynamic_state = SemanticTemporalState(
        hidden=zero_state.hidden,
        dynamic_error=tuple(torch.randn_like(value) for value in zero_state.dynamic_error),
    )

    posterior_a, state_a, _ = semantic_temporal_error_step(
        corrections,
        observation,
        prediction,
        semantic,
        zero_state,
    )
    posterior_b, state_b, _ = semantic_temporal_error_step(
        corrections,
        observation,
        prediction,
        semantic,
        different_dynamic_state,
    )

    assert torch.allclose(posterior_a.z1, posterior_b.z1)
    assert torch.allclose(posterior_a.z4, posterior_b.z4)
    assert torch.allclose(state_a.hidden[0], state_b.hidden[0])
    assert torch.allclose(state_a.hidden[1], state_b.hidden[1])
    assert not torch.allclose(state_a.dynamic_error[0], state_b.dynamic_error[0])
    assert not hasattr(corrections[0].error_state, "transition")


def test_dynamic_error_state_detaches_with_correction_state():
    torch.manual_seed(2)
    observation_tensor = torch.randn(1, 4, 5, 6)
    observation = unified(observation_tensor)
    state = zero_semantic_temporal_state(observation)
    detached = detach_error_state(state)

    assert isinstance(detached, SemanticTemporalState)
    assert all(value.grad_fn is None for value in detached.hidden)
    assert all(value.grad_fn is None for value in detached.dynamic_error)
