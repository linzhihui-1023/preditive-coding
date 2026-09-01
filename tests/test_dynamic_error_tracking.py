import pytest
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
    apply_semantic_temporal_corrections,
    build_semantic_temporal_corrections,
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
    integration_factor = (
        DYNAMIC_ERROR_SAMPLE_TIME / DYNAMIC_ERROR_TIME_CONSTANT
    )
    expected_from_original_law = torch.zeros_like(expected_error) + (
        integration_factor
        * (
            expected_error
            - DYNAMIC_ERROR_GAIN * torch.zeros_like(expected_error)
        )
    )
    assert torch.allclose(expected_dynamic, expected_from_original_law)


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


def test_dynamic_branch_uses_previous_state_and_exposes_bounded_gate_evidence():
    torch.manual_seed(3)
    channels = 4
    corrections = nn.ModuleList(
        [
            SemanticTemporalErrorCorrection(
                channels, use_dynamic_error=True, dynamic_gate_limit=0.2
            ),
            SemanticTemporalErrorCorrection(
                channels, use_dynamic_error=True, dynamic_gate_limit=0.2
            ),
        ]
    )
    observation_tensor = torch.randn(1, channels, 5, 6)
    prediction_tensor = torch.randn_like(observation_tensor)
    semantic_tensor = torch.randn_like(observation_tensor)
    observation = unified(observation_tensor)
    prediction = unified(prediction_tensor)
    semantic = unified(semantic_tensor)
    previous_dynamic = tuple(torch.randn_like(observation_tensor) for _ in range(2))
    hidden = (torch.zeros_like(observation_tensor), torch.zeros_like(observation_tensor))

    _, _, values = apply_semantic_temporal_corrections(
        corrections,
        observation,
        prediction,
        semantic,
        hidden,
        previous_dynamic_error=previous_dynamic,
    )

    expected_historical = previous_dynamic[0]
    expected_residual = observation_tensor - prediction_tensor - expected_historical
    assert torch.allclose(values["historical_dynamic_component_z1"], expected_historical)
    assert torch.allclose(values["first_order_model_residual_z1"], expected_residual)
    assert values["gate_modulation_z1"].abs().max() <= 0.2
    assert "predicted_next_error_z1" in values
    assert "predicted_next_error_z4" in values


def test_zero_initialized_dynamic_branch_matches_original_transition_exactly():
    torch.manual_seed(4)
    channels = 4
    original = SemanticTemporalErrorCorrection(channels)
    dynamic = SemanticTemporalErrorCorrection(channels, use_dynamic_error=True)
    dynamic.load_state_dict(original.state_dict(), strict=False)
    observation = torch.randn(1, channels, 5, 6)
    prediction = torch.randn_like(observation)
    semantic = torch.randn_like(observation)
    hidden = torch.randn_like(observation)

    original_values = original(observation, prediction, semantic, hidden)
    dynamic_values = dynamic(
        observation,
        prediction,
        semantic,
        hidden,
        previous_dynamic_error=torch.randn_like(observation),
    )

    assert torch.equal(original_values[3], dynamic_values[3])
    assert torch.equal(original_values[4], dynamic_values[4])
    assert torch.count_nonzero(dynamic.last_dynamic_values["gate_modulation"]) == 0
    assert torch.count_nonzero(dynamic.last_dynamic_values["update_gate_delta"]) == 0
    assert torch.count_nonzero(dynamic.last_dynamic_values["reset_gate_delta"]) == 0


def test_temporal_prediction_baseline_does_not_enable_dynamic_modulation():
    torch.manual_seed(5)
    channels = 4
    correction = SemanticTemporalErrorCorrection(
        channels,
        use_dynamic_error=False,
        use_temporal_prediction=True,
    )
    observation = torch.randn(1, channels, 5, 6)

    correction(
        observation,
        torch.randn_like(observation),
        torch.randn_like(observation),
        torch.zeros_like(observation),
    )

    assert correction.dynamic_encoder is None
    assert "predicted_next_error" in correction.last_dynamic_values
    assert "gate_modulation" not in correction.last_dynamic_values


def test_dynamic_branch_rejects_non_monotonic_euler_q(monkeypatch):
    monkeypatch.setenv("PREDIFY_DYNAMIC_ERROR_SAMPLE_TIME", "0.6")
    monkeypatch.setenv("PREDIFY_DYNAMIC_ERROR_TIME_CONSTANT", "0.5")
    monkeypatch.setenv("PREDIFY_DYNAMIC_ERROR_GAIN", "1.0")

    with pytest.raises(ValueError, match="0 < Ts \\* K_e / tau_e <= 1"):
        build_semantic_temporal_corrections(use_dynamic_error=True)
