import pytest
import torch

from predify2021.mce_scores.diagnose_kitti_step_semantic_innovation_bottleneck import (
    oracle_projection_scale,
    residual_energy_fractions,
)
from predify2021.mce_scores.train_kitti_step_semantic_v3 import residual_aware_loss
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorGuidedSemanticRestorationPredictor,
    ErrorRegulatedSemanticRestorationPredictor,
    UnifiedFeatures,
)


def unified(value):
    return UnifiedFeatures(value, value, value, value)


def test_oracle_projection_scale_separates_direction_from_amplitude():
    oracle = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]])
    assert oracle_projection_scale(oracle * 0.25, oracle) == pytest.approx(0.25)
    assert oracle_projection_scale(-oracle * 0.5, oracle) == pytest.approx(-0.5)


def test_residual_energy_fraction_detects_sparse_energy():
    residual = torch.zeros(100)
    residual[:5] = 1.0
    fractions = residual_energy_fractions(residual)
    assert fractions["top_5pct_energy_fraction"] == pytest.approx(1.0)
    assert fractions["top_10pct_energy_fraction"] == pytest.approx(1.0)


def test_restoration_diagnostics_expose_every_innovation_stage():
    predictor = ErrorGuidedSemanticRestorationPredictor(hidden_channels=4)
    observation = unified(torch.randn(1, 128, 2, 3))
    prediction = unified(torch.randn(1, 128, 2, 3))

    restored, hidden, diagnostics = predictor.restore_current(
        observation, prediction, None
    )

    assert hidden.shape == (1, 4, 2, 3)
    assert restored.z4.shape == observation.z4.shape
    assert {
        "prediction_error_z4",
        "error_innovation",
        "update_gain",
        "error_drive",
        "context_modulation",
        "state_innovation",
        "restoration_delta_z4",
    } <= diagnostics.keys()
    assert diagnostics["update_gain"].shape == hidden.shape
    assert diagnostics["error_drive"].shape == hidden.shape
    assert diagnostics["context_modulation"].shape == hidden.shape
    assert diagnostics["state_innovation"].shape == hidden.shape


def test_v3_frame_zero_state_is_observation_and_error_only_changes_gain():
    predictor = ErrorRegulatedSemanticRestorationPredictor()
    observation = unified(torch.randn(1, 128, 2, 3))
    initial = predictor.initial_semantic_state(observation)
    assert torch.equal(initial, observation.z4)

    hidden = torch.randn_like(observation.z4)
    first = torch.randn_like(observation.z4)
    second = torch.randn_like(observation.z4)
    _, diagnostics_first = predictor.semantic_state_cell(observation.z4, first, hidden)
    _, diagnostics_second = predictor.semantic_state_cell(observation.z4, second, hidden)
    assert torch.equal(
        diagnostics_first["semantic_candidate"],
        diagnostics_second["semantic_candidate"],
    )
    assert not torch.equal(
        diagnostics_first["semantic_update_gain"],
        diagnostics_second["semantic_update_gain"],
    )

    output_zero, _ = predictor.semantic_state_cell(
        observation.z4,
        first,
        hidden,
        update_gain_override=torch.zeros_like(hidden),
    )
    output_one, diagnostics_one = predictor.semantic_state_cell(
        observation.z4,
        first,
        hidden,
        update_gain_override=torch.ones_like(hidden),
    )
    assert torch.allclose(output_zero, hidden)
    assert torch.allclose(output_one, diagnostics_one["semantic_candidate"])


def test_v3_restoration_is_discrepancy_only_and_dynamics_can_be_frozen():
    predictor = ErrorRegulatedSemanticRestorationPredictor()
    predictor.freeze_dynamics()
    observation = unified(torch.randn(1, 128, 2, 3))
    prediction = unified(torch.randn(1, 128, 2, 3))
    restored, _, diagnostics = predictor.restore_current(
        observation,
        prediction,
        predictor.initial_semantic_state(observation),
    )
    expected = predictor.semantic_restoration_head(
        diagnostics["semantic_discrepancy"]
    )
    assert torch.allclose(diagnostics["restoration_delta_z4"], expected)
    assert sum(
        p.numel()
        for name in predictor.DYNAMICS_MODULES
        for p in getattr(predictor, name).parameters()
        if p.requires_grad
    ) == 0
    assert restored.z4.shape == observation.z4.shape


def test_v3_residual_aware_loss_is_finite_and_single_scalar():
    prediction = torch.randn(1, 128, 2, 3, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[..., 0, 0] = 10.0
    loss = residual_aware_loss(prediction, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None


def test_v3_restoration_head_starts_at_pointwise_zero_point_one_identity():
    predictor = ErrorRegulatedSemanticRestorationPredictor()
    discrepancy = torch.randn(1, 128, 3, 4)
    restored_delta = predictor.semantic_restoration_head(discrepancy)
    assert torch.allclose(restored_delta, 0.1 * discrepancy, atol=1e-6, rtol=1e-5)


def test_v3_zero_error_preserves_raw_diagnostic_but_zeros_encoded_signal():
    predictor = ErrorRegulatedSemanticRestorationPredictor()
    observation = unified(torch.randn(1, 128, 2, 3))
    prediction = unified(torch.randn(1, 128, 2, 3))
    normal, _, normal_diagnostics = predictor.restore_current(
        observation,
        prediction,
        predictor.initial_semantic_state(observation),
    )
    zero, _, zero_diagnostics = predictor.restore_current(
        observation,
        prediction,
        predictor.initial_semantic_state(observation),
        zero_encoded_prediction_error=True,
    )
    assert torch.equal(
        normal_diagnostics["prediction_error_z4"],
        zero_diagnostics["prediction_error_z4"],
    )
    assert torch.count_nonzero(zero_diagnostics["encoded_prediction_error"]) == 0
    assert torch.equal(
        normal_diagnostics["semantic_candidate"],
        zero_diagnostics["semantic_candidate"],
    )
    assert normal.z4.shape == zero.z4.shape
