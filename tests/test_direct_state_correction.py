import inspect

import torch

from predify2021.mce_scores.train_kitti_step_direct_state_correction import (
    direct_posterior,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    DirectStateCorrection,
    UnifiedFeatures,
)


def test_direct_correction_uses_three_state_inputs_and_preserves_shape():
    correction = DirectStateCorrection()
    observation = torch.randn(1, 128, 5, 7)
    error = torch.randn_like(observation)
    dynamic_error = torch.randn_like(observation)
    delta = correction(observation, error, dynamic_error)
    assert delta.shape == observation.shape
    assert tuple(inspect.signature(correction.forward).parameters) == (
        "observation",
        "error",
        "dynamic_error",
    )


def test_direct_posterior_only_changes_z1_and_z4():
    observation = UnifiedFeatures(
        torch.randn(1, 128, 2, 3),
        torch.randn(1, 128, 1, 2),
        torch.randn(1, 128, 1, 2),
        torch.randn(1, 128, 1, 2),
    )
    error = UnifiedFeatures(*(torch.randn_like(value) for value in observation.as_tuple()))
    dynamic_error = UnifiedFeatures(*(torch.randn_like(value) for value in observation.as_tuple()))
    corrections = torch.nn.ModuleList([DirectStateCorrection(), DirectStateCorrection()])
    posterior = direct_posterior(observation, error, dynamic_error, corrections)
    assert torch.equal(posterior.z2, observation.z2)
    assert torch.equal(posterior.z3, observation.z3)
    assert posterior.z1.shape == observation.z1.shape
    assert posterior.z4.shape == observation.z4.shape
