"""Shared VSPW FAST-B loading, host writeback, and temporal-state utilities.

This module contains no dataset corruption path.  It is intentionally shared by
the training and evaluation entry points so both use the same 124-class Host,
the same FAST-B checkpoint payload, and the same correction path.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorRegulatedSemanticRestorationPredictor,
    HostFeature,
    UnifiedFeatures,
    build_deeplabv3plus_resnet50_host,
)


VSPW_NUM_CLASSES = 124
VSPW_IGNORE_LABEL = 255
C4_ADAPTER_INDEX = 3
C4_WRITEBACK_KEY = "3"


@dataclass
class FastBTemporalState:
    pending_prediction: Optional[UnifiedFeatures] = None
    h4_dynamics: Optional[torch.Tensor] = None
    h1_dynamics: Optional[torch.Tensor] = None
    semantic_hidden: Optional[torch.Tensor] = None
    error_temporal_state: object = None


def _state_dict(payload):
    if not isinstance(payload, dict):
        raise RuntimeError("Checkpoint payload must be a dictionary.")
    for key in ("model_state_dict", "predictor_state_dict", "state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    if payload and all(isinstance(key, str) for key in payload):
        return payload
    raise RuntimeError("Checkpoint does not contain a state dictionary.")


def load_payload(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_vspw_host(checkpoint_path, device):
    payload = load_payload(checkpoint_path)
    host = build_deeplabv3plus_resnet50_host(
        num_classes=VSPW_NUM_CLASSES,
        load_cityscapes_checkpoint=False,
    ).to(device)
    host.load_state_dict(_state_dict(payload), strict=True)
    return host, payload


def load_dynamics_state(checkpoint_path):
    payload = load_payload(checkpoint_path)
    return _state_dict(payload), payload


def build_fast_b_predictor(dynamics_checkpoint, checkpoint_path=None, device="cpu"):
    dynamics_state, dynamics_payload = load_dynamics_state(dynamics_checkpoint)
    predictor = ErrorRegulatedSemanticRestorationPredictor(
        use_error_temporal_stats=True
    ).to(device)
    predictor.load_dynamics_from_role_separated_state_dict(dynamics_state)
    predictor.freeze_dynamics()
    fast_b_payload = None
    if checkpoint_path is not None:
        fast_b_payload = load_payload(checkpoint_path)
        predictor.load_state_dict(_state_dict(fast_b_payload), strict=True)
    return predictor, dynamics_payload, fast_b_payload


def configure_fast_b_host(host, trainable: bool):
    """Freeze the Host, optionally exposing only the validated Joint-C4 path."""
    host.requires_grad_(False)
    host.host_conditioned_writeback_enabled = True
    if trainable:
        host.multi_layer_adapter.output_adapters[C4_ADAPTER_INDEX].requires_grad_(True)
        host.host_conditioned_writebacks[C4_WRITEBACK_KEY].requires_grad_(True)
    host.eval()
    return host


def load_joint_c4_payload(host, payload):
    if payload is None or not payload.get("joint_c4_training", False):
        raise RuntimeError("FAST-B checkpoint must declare joint_c4_training.")
    host.multi_layer_adapter.output_adapters[C4_ADAPTER_INDEX].load_state_dict(
        payload["c4_output_adapter_state_dict"], strict=True
    )
    host.host_conditioned_writebacks[C4_WRITEBACK_KEY].load_state_dict(
        payload["c4_writeback_state_dict"], strict=True
    )


def encode_vspw_frame(host, image):
    """Extract a frozen Host frame and its 128-channel FAST-B features."""
    with torch.no_grad():
        raw = host.extract_backbone_features(image)
        latent = host.encode_backbone_features(raw)
    return raw, latent


def zero_state(state):
    return UnifiedFeatures(*(torch.zeros_like(value) for value in state.as_tuple()))


def error_state(observation, prediction):
    return UnifiedFeatures(
        *(current - predicted for current, predicted in zip(
            observation.as_tuple(), prediction.as_tuple()
        ))
    )


def corrected_host_feature_with_size(host, raw_features, observation, restored, output_size):
    """Size-explicit variant used because backbone tensors do not store input size."""
    zeros = zero_state(observation)
    delta = UnifiedFeatures(zeros.z1, zeros.z2, zeros.z3, restored.z4 - observation.z4)
    applied = host.decode_conditioned_adapter_deltas(raw_features, delta)
    baseline = host.decode_conditioned_adapter_deltas(raw_features, zeros)
    return HostFeature(
        tensor=raw_features.c4 + applied.c4 - baseline.c4,
        low_level=raw_features.c1 + applied.c1 - baseline.c1,
        output_size=tuple(output_size),
    )


def next_prediction(predictor, observation, prediction_error, state):
    with torch.no_grad():
        pending, h4, h1 = predictor.predict_next(
            observation,
            prediction_error,
            state.h4_dynamics,
            state.h1_dynamics,
        )
    state.pending_prediction = pending
    state.h4_dynamics = h4
    state.h1_dynamics = h1
    return state


def reset_temporal_state():
    return FastBTemporalState()


def restore_frame(predictor, observation, state, temporal_mode):
    """Restore one frame and return the updated state.

    ``full`` carries the causal prediction, dynamics state, semantic state and
    error statistics across frames.  ``reset`` starts every frame from zero;
    its current observation is used only to form the zero-error frame-local
    interface, so it cannot accidentally retain history.
    """
    if temporal_mode not in {"full", "reset"}:
        raise ValueError("temporal_mode must be 'full' or 'reset'")
    if temporal_mode == "reset":
        state = reset_temporal_state()
        state.pending_prediction = observation
        state.semantic_hidden = torch.zeros_like(observation.z4)
    elif state.pending_prediction is None:
        state.pending_prediction = observation
        state.semantic_hidden = torch.zeros_like(observation.z4)

    prediction_error = error_state(observation, state.pending_prediction)
    restored, semantic_hidden, diagnostics = predictor.restore_current(
        observation,
        state.pending_prediction,
        state.semantic_hidden,
        error_temporal_state=state.error_temporal_state,
    )
    state.semantic_hidden = semantic_hidden
    state.error_temporal_state = diagnostics["error_temporal_state"]
    if temporal_mode == "full":
        next_prediction(predictor, observation, prediction_error, state)
    else:
        state.pending_prediction = None
        state.h4_dynamics = None
        state.h1_dynamics = None
        state.semantic_hidden = None
        state.error_temporal_state = None
    return restored, state, diagnostics


def detach_temporal_state(state):
    state.pending_prediction = (
        UnifiedFeatures(*(value.detach() for value in state.pending_prediction.as_tuple()))
        if state.pending_prediction is not None else None
    )
    state.h4_dynamics = state.h4_dynamics.detach() if state.h4_dynamics is not None else None
    state.h1_dynamics = state.h1_dynamics.detach() if state.h1_dynamics is not None else None
    state.semantic_hidden = state.semantic_hidden.detach() if state.semantic_hidden is not None else None
    if state.error_temporal_state is not None:
        state.error_temporal_state = state.error_temporal_state.detach()
    return state


def parameter_report(host, predictor):
    host_params = sum(parameter.numel() for parameter in host.parameters())
    frozen_host_params = sum(
        parameter.numel() for parameter in host.parameters() if not parameter.requires_grad
    )
    fast_b_trainable = sum(
        parameter.numel() for parameter in predictor.semantic_parameters()
        if parameter.requires_grad
    )
    fast_b_trainable += sum(
        parameter.numel()
        for module in (
            host.multi_layer_adapter.output_adapters[C4_ADAPTER_INDEX],
            host.host_conditioned_writebacks[C4_WRITEBACK_KEY],
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    fast_b_host_interface = sum(
        parameter.numel()
        for module in (
            host.multi_layer_adapter.output_adapters[C4_ADAPTER_INDEX],
            host.host_conditioned_writebacks[C4_WRITEBACK_KEY],
        )
        for parameter in module.parameters()
    )
    additional_inference = sum(parameter.numel() for parameter in predictor.parameters())
    # C4 adapter/writeback are part of the loaded Host module and are therefore
    # already included in host_params. Keep their count explicit without
    # double-counting them in the unique total.
    total = host_params + additional_inference
    return {
        "host_params": host_params,
        "frozen_host_params": frozen_host_params,
        "fast_b_trainable_params": fast_b_trainable,
        "additional_inference_params": additional_inference,
        "fast_b_host_interface_params": fast_b_host_interface,
        "total_params": total,
        "trainable_ratio": fast_b_trainable / max(total, 1),
    }
