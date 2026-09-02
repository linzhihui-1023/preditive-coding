import math

import torch

from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorRegulatedSemanticStateCell,
    ErrorTemporalStatisticsState,
    update_error_temporal_statistics,
)
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_recurrent_predictor import (
    ERROR_STATS_BETA_LONG,
    ERROR_STATS_BETA_SHORT,
)


def test_exact_discrete_coefficients_and_initialization():
    assert math.isclose(ERROR_STATS_BETA_LONG, math.exp(-0.1035 / 1.5), rel_tol=0, abs_tol=1e-12)
    assert math.isclose(ERROR_STATS_BETA_SHORT, math.exp(-0.1035 / 0.5), rel_tol=0, abs_tol=1e-12)
    error = torch.tensor([[[[2.0]], [[-1.0]]]])
    state = update_error_temporal_statistics(None, error)
    assert torch.equal(state.mean, error)
    assert torch.equal(state.magnitude, error.abs())
    assert torch.equal(state.second_moment, error.square())
    assert torch.equal(state.change, torch.zeros_like(error))
    assert torch.equal(state.previous_error, error)


def test_batch_independent_o1_update_and_feature_shape():
    first = torch.tensor([[[[1.0]], [[-2.0]]], [[[3.0]], [[4.0]]]])
    second = torch.tensor([[[[2.0]], [[-1.0]]], [[[4.0]], [[1.0]]]])
    state = update_error_temporal_statistics(None, first)
    updated = update_error_temporal_statistics(state, second)
    assert updated.mean.shape == first.shape
    assert updated.as_features().shape == (2, 6, 1, 1)
    expected_change = (1.0 - ERROR_STATS_BETA_SHORT) * (second - first).abs()
    assert torch.allclose(updated.change, expected_change)
    assert torch.all(updated.magnitude[0] != updated.magnitude[1])


def test_history_only_changes_gain_not_candidate():
    torch.manual_seed(0)
    cell = ErrorRegulatedSemanticStateCell(channels=4, use_error_temporal_stats=True)
    observation = torch.randn(2, 4, 3, 3)
    hidden = torch.randn_like(observation)
    encoded = torch.randn_like(observation)
    stats_a = torch.randn(2, 12, 3, 3)
    stats_b = torch.randn(2, 12, 3, 3)
    _, diag_a = cell(observation, encoded, hidden, error_history_statistics=stats_a)
    _, diag_b = cell(observation, encoded, hidden, error_history_statistics=stats_b)
    assert torch.allclose(diag_a["semantic_candidate"], diag_b["semantic_candidate"])
    assert not torch.allclose(diag_a["semantic_update_gain"], diag_b["semantic_update_gain"])
    assert sum(p.numel() for p in cell.history_projection.parameters()) == 3 * 4 * 4
