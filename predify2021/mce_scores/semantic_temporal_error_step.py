import os
from typing import NamedTuple

import torch

from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import (
    apply_semantic_temporal_corrections,
)
from predify2021.model_factory.targetflow.core import (
    build_temporal_prediction_error_state,
)


# Provisional experiment defaults only. The dynamic-error definition is the
# parameterized first-order law in targetflow.core; these values are not fixed
# parts of that definition and can be replaced by later adaptation.
DYNAMIC_ERROR_SAMPLE_TIME = float(
    os.environ.get("PREDIFY_DYNAMIC_ERROR_SAMPLE_TIME", "0.1035")
)
DYNAMIC_ERROR_TIME_CONSTANT = float(
    os.environ.get("PREDIFY_DYNAMIC_ERROR_TIME_CONSTANT", "0.5")
)
DYNAMIC_ERROR_GAIN = float(
    os.environ.get("PREDIFY_DYNAMIC_ERROR_GAIN", "1.0")
)


class SemanticTemporalState(NamedTuple):
    hidden: tuple
    dynamic_error: tuple


def semantic_temporal_error_step(
    corrections,
    observation,
    dynamics,
    semantic,
    state,
):
    if not isinstance(state, SemanticTemporalState):
        return apply_semantic_temporal_corrections(
            corrections, observation, dynamics, semantic, state
        )

    posterior, hidden, values = apply_semantic_temporal_corrections(
        corrections,
        observation,
        dynamics,
        semantic,
        state.hidden,
        previous_dynamic_error=state.dynamic_error,
    )
    dynamic_error = (
        build_temporal_prediction_error_state(
            values["error_z1"],
            state.dynamic_error[0],
            sample_time=DYNAMIC_ERROR_SAMPLE_TIME,
            time_constant=DYNAMIC_ERROR_TIME_CONSTANT,
            error_gain=DYNAMIC_ERROR_GAIN,
        ).detach(),
        build_temporal_prediction_error_state(
            values["error_z4"],
            state.dynamic_error[1],
            sample_time=DYNAMIC_ERROR_SAMPLE_TIME,
            time_constant=DYNAMIC_ERROR_TIME_CONSTANT,
            error_gain=DYNAMIC_ERROR_GAIN,
        ).detach(),
    )
    values = {
        **values,
        "hidden_z1": hidden[0],
        "hidden_z4": hidden[1],
        "dynamic_error_z1": dynamic_error[0],
        "dynamic_error_z4": dynamic_error[1],
    }
    return posterior, SemanticTemporalState(hidden=hidden, dynamic_error=dynamic_error), values


def detach_error_state(state):
    if isinstance(state, SemanticTemporalState):
        return SemanticTemporalState(
            hidden=tuple(value.detach() for value in state.hidden),
            dynamic_error=tuple(value.detach() for value in state.dynamic_error),
        )
    return tuple(value.detach() for value in state)


def zero_error_state(observation):
    return (
        torch.zeros_like(observation.z1),
        torch.zeros_like(observation.z4),
    )


def zero_semantic_temporal_state(observation):
    hidden = zero_error_state(observation)
    return SemanticTemporalState(
        hidden=hidden,
        dynamic_error=tuple(torch.zeros_like(value) for value in hidden),
    )
