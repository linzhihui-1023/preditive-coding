from typing import NamedTuple

import torch

from predify2021.model_factory.deeplabv3plus_resnet50 import UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import (
    apply_dpc_semantic_temporal_corrections,
    apply_semantic_temporal_corrections,
)


class DynamicTransitionState(NamedTuple):
    hidden: tuple
    dynamic_error: tuple


def semantic_temporal_error_step(
    corrections,
    observation,
    dynamics,
    semantic,
    hidden,
    zero_dynamic=False,
):
    if getattr(corrections[0], "uses_dynamic_transition", False):
        return apply_dpc_semantic_temporal_corrections(
            corrections,
            observation,
            dynamics,
            semantic,
            hidden,
            zero_dynamic,
        )
    posterior, hidden, values = apply_semantic_temporal_corrections(
        corrections, observation, dynamics, semantic, hidden
    )
    return posterior, hidden, values


def detach_error_state(hidden):
    if isinstance(hidden, DynamicTransitionState):
        return DynamicTransitionState(
            hidden=tuple(value.detach() for value in hidden.hidden),
            dynamic_error=tuple(value.detach() for value in hidden.dynamic_error),
        )
    return tuple(value.detach() for value in hidden)


def zero_error_state(observation):
    return (
        torch.zeros_like(observation.z1),
        torch.zeros_like(observation.z4),
    )


def zero_dynamic_transition_state(observation):
    hidden = zero_error_state(observation)
    return DynamicTransitionState(
        hidden=hidden,
        dynamic_error=tuple(torch.zeros_like(value) for value in hidden),
    )
