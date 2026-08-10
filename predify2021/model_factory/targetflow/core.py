from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class TargetFlowDynamicErrorConfig:
    sample_time: float
    time_constant: Sequence[float]
    error_gain: Sequence[float]
    enabled: bool = True


class TargetFlowFeedbackModule(nn.Module):
    def __init__(self, projector: nn.Module, activation: Optional[nn.Module] = None):
        super().__init__()
        self.projector = projector
        self.activation = activation

    def forward(self, target_source: torch.Tensor):
        target_preactivation = self.projector(target_source)
        if self.activation is None:
            target_output = target_preactivation
        else:
            target_output = self.activation(target_preactivation)
        return target_preactivation, target_output


@dataclass
class TargetFlowLayerState:
    layer_index: int
    forward_input: Optional[torch.Tensor] = None
    forward_output: Optional[torch.Tensor] = None
    target_source: Optional[torch.Tensor] = None
    target_preactivation: Optional[torch.Tensor] = None
    target_output: Optional[torch.Tensor] = None
    previous_prediction: Optional[torch.Tensor] = None
    instant_error: Optional[torch.Tensor] = None
    previous_error: Optional[torch.Tensor] = None
    previous_instant_error: Optional[torch.Tensor] = None
    error: Optional[torch.Tensor] = None
    loss_error: Optional[torch.Tensor] = None
    learn_signal: Optional[torch.Tensor] = None
    local_loss: Optional[torch.Tensor] = None
    parameter_grad_stats: Optional[dict] = None


def run_backward_target_flow(
    layer_states: Sequence[TargetFlowLayerState],
    feedback_modules,
    *,
    top_target: Optional[torch.Tensor] = None,
    mode: str = "recursive",
):
    """
    Populate target-flow state in-place.

    Modes:
    - quasi_steady: lower-layer targets are projected from the next layer's
      forward output, matching the "structured quasi-steady-state target flow"
      interpretation.
    - recursive: lower-layer targets are projected from the next layer's target
      output, which keeps a strictly recursive backward target chain.
    """
    if mode not in {"quasi_steady", "recursive"}:
        raise ValueError(f"Unsupported target flow mode: {mode}")
    if not layer_states:
        return []

    top_state = layer_states[-1]
    top_target_output = top_state.forward_output if top_target is None else top_target
    top_state.target_source = None
    top_state.target_preactivation = top_target_output
    top_state.target_output = top_target_output
    top_state.error = build_targetflow_error(top_state.target_output, top_state.forward_output)

    for idx in range(len(layer_states) - 2, -1, -1):
        next_state = layer_states[idx + 1]
        state = layer_states[idx]
        if mode == "quasi_steady":
            target_source = next_state.forward_output
        else:
            target_source = next_state.target_output

        target_preactivation, target_output = feedback_modules[idx](target_source)
        state.target_source = target_source
        state.target_preactivation = target_preactivation
        state.target_output = target_output
        state.error = build_targetflow_error(state.target_output, state.forward_output)

    return layer_states


def build_targetflow_instant_error(target_output: torch.Tensor, forward_output: torch.Tensor):
    if target_output is None or forward_output is None:
        return None
    return forward_output - target_output


def build_dynamic_targetflow_error(
    instant_error: Optional[torch.Tensor],
    previous_error: Optional[torch.Tensor],
    sample_time,
    time_constant,
    error_gain,
):
    if instant_error is None:
        return None

    if previous_error is None:
        previous_error = torch.zeros_like(instant_error)

    sample_time_tensor = instant_error.new_tensor(float(sample_time))
    time_constant_tensor = instant_error.new_tensor(float(time_constant))
    error_gain_tensor = instant_error.new_tensor(float(error_gain))

    if sample_time_tensor <= 0:
        raise ValueError(f"sample_time must be positive, but got {float(sample_time_tensor.item())}.")
    if time_constant_tensor <= 0:
        raise ValueError(f"time_constant must be positive, but got {float(time_constant_tensor.item())}.")

    integration_factor = sample_time_tensor / time_constant_tensor
    memory_factor = 1.0 - error_gain_tensor * integration_factor
    return integration_factor * instant_error + memory_factor * previous_error


def build_targetflow_error(
    target_output: torch.Tensor,
    forward_output: torch.Tensor,
    previous_error: Optional[torch.Tensor] = None,
    *,
    sample_time: float = 1.0,
    time_constant: float = 1.0,
    error_gain: float = 1.0,
    dynamic: bool = False,
    mode: Optional[str] = None,
    previous_instant_error: Optional[torch.Tensor] = None,
):
    """Build an instant, recursive-EMA, or one-step lagged error state."""
    instant_error = build_targetflow_instant_error(target_output, forward_output)
    if instant_error is None:
        return None
    resolved_mode = mode or ("ema" if dynamic else "instant")
    if resolved_mode == "instant":
        return instant_error
    if resolved_mode == "ema":
        memory_error = previous_error
    elif resolved_mode == "lag1":
        memory_error = previous_instant_error
    else:
        raise ValueError(f"Unsupported target-flow error state mode: {resolved_mode}")

    return build_dynamic_targetflow_error(
        instant_error,
        memory_error,
        sample_time=sample_time,
        time_constant=time_constant,
        error_gain=error_gain,
    )


def build_targetflow_learn_signal(target_output: torch.Tensor, forward_output: torch.Tensor):
    if target_output is None or forward_output is None:
        return None
    return target_output.detach() - forward_output


def build_targetflow_learn_signal_from_error(error: Optional[torch.Tensor]):
    if error is None:
        return None
    return -error


def build_targetflow_local_loss(target_output: torch.Tensor, forward_output: torch.Tensor):
    if target_output is None or forward_output is None:
        return None
    return torch.mean((target_output.detach() - forward_output) ** 2)


def build_targetflow_local_loss_from_error(error: Optional[torch.Tensor]):
    if error is None:
        return None
    return torch.mean(error ** 2)


def compute_module_grad_stats(module: nn.Module, local_loss: Optional[torch.Tensor]):
    if local_loss is None:
        return None

    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        return {
            "rms": 0.0,
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "num_parameters": 0,
        }

    gradients = torch.autograd.grad(local_loss, parameters, retain_graph=True, allow_unused=True)
    flat_components = []
    total_parameters = 0
    for gradient in gradients:
        if gradient is None:
            continue
        detached = gradient.detach().reshape(-1).float()
        total_parameters += detached.numel()
        flat_components.append(detached)

    if not flat_components:
        return {
            "rms": 0.0,
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "num_parameters": total_parameters,
        }

    flat = torch.cat(flat_components)
    return {
        "rms": float(torch.sqrt(torch.mean(flat.pow(2))).item()),
        "mean_abs": float(torch.mean(torch.abs(flat)).item()),
        "max_abs": float(torch.max(torch.abs(flat)).item()),
        "num_parameters": total_parameters,
    }
