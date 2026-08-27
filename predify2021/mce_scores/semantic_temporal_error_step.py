import torch

from predify2021.model_factory.deeplabv3plus_resnet50 import UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import (
    apply_semantic_temporal_corrections,
)


def semantic_temporal_error_step(corrections, observation, dynamics, semantic, hidden):
    posterior, hidden, values = apply_semantic_temporal_corrections(
        corrections, observation, dynamics, semantic, hidden
    )
    return posterior, hidden, values


def detach_error_state(hidden):
    return tuple(value.detach() for value in hidden)


def zero_error_state(observation):
    return (
        torch.zeros_like(observation.z1),
        torch.zeros_like(observation.z4),
    )
