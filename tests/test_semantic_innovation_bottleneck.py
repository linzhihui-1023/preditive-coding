import pytest
import torch

from predify2021.mce_scores.diagnose_kitti_step_semantic_innovation_bottleneck import (
    oracle_projection_scale,
    residual_energy_fractions,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorGuidedSemanticRestorationPredictor,
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
