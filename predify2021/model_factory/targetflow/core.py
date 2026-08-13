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


@dataclass
class TemporalPredictionErrorConfig:
    sample_time: float
    time_constant: float
    error_gain: float


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


@dataclass
class RealFramePCoderLayerState:
    """One layer's immutable audit record for a single real-frame update."""

    layer_index: int
    frame_index: int
    feedforward_drive: torch.Tensor
    previous_representation: Optional[torch.Tensor] = None
    previous_prediction: Optional[torch.Tensor] = None
    previous_feedback_prediction: Optional[torch.Tensor] = None
    previous_dynamic_error: Optional[torch.Tensor] = None
    error_correction: Optional[torch.Tensor] = None
    error_scale: Optional[torch.Tensor] = None
    c_sqrt: Optional[torch.Tensor] = None
    representation: Optional[torch.Tensor] = None
    prediction_target: Optional[torch.Tensor] = None
    prediction: Optional[torch.Tensor] = None
    instant_error: Optional[torch.Tensor] = None
    dynamic_error: Optional[torch.Tensor] = None


def _prediction_output(prediction_module: nn.Module, representation: torch.Tensor):
    module_output = prediction_module(representation)
    return module_output[-1] if isinstance(module_output, tuple) else module_output


def compute_pcoder_c_sqrt(
    prediction_module: nn.Module,
    representation: torch.Tensor,
    repeats: int = 10,
):
    """Reproduce the effective decoder-window calibration from ``PCoderN``."""
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    reference_representation = representation.detach()
    with torch.no_grad():
        original_prediction = _prediction_output(
            prediction_module,
            reference_representation,
        ).detach()
        affected_count = original_prediction.new_zeros(())
        for _ in range(repeats):
            perturbed = reference_representation.clone()
            perturbed[
                :,
                perturbed.shape[1] // 2,
                perturbed.shape[2] // 2,
                perturbed.shape[3] // 2,
            ] = torch.randint(
                -10000,
                10000,
                (perturbed.shape[0],),
                device=perturbed.device,
            ).to(dtype=perturbed.dtype)
            perturbed_prediction = _prediction_output(
                prediction_module,
                perturbed,
            )
            affected_count += (
                (original_prediction != perturbed_prediction).sum().float()
                / original_prediction.shape[0]
            )

    affected_count = affected_count / float(repeats)
    if not torch.isfinite(affected_count) or affected_count <= 0:
        raise RuntimeError(
            "PCoderN C calibration produced a non-positive affected-cell count."
        )
    return torch.sqrt(affected_count).detach()


def project_dynamic_error_to_representation(
    prediction_module: nn.Module,
    previous_representation: Optional[torch.Tensor],
    previous_prediction: Optional[torch.Tensor],
    previous_dynamic_error: Optional[torch.Tensor],
    c_sqrt: Optional[torch.Tensor],
):
    """Project a detached output-space error through a frozen PCoder decoder.

    The pseudo-target makes ``previous_dynamic_error`` the exact residual used
    by an ordinary MSE error-correction gradient. The returned gradient includes
    the original ``PCoderN`` factor ``K / C_sqrt`` while preventing gradients
    from reaching model parameters or earlier video frames.
    """
    if (
        previous_representation is None
        or previous_prediction is None
        or previous_dynamic_error is None
        or c_sqrt is None
    ):
        return None

    representation = previous_representation.detach().requires_grad_(True)
    with torch.enable_grad():
        prediction = _prediction_output(prediction_module, representation)
        dynamic_error = previous_dynamic_error.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        historical_prediction = previous_prediction.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        if (
            dynamic_error.shape != prediction.shape
            or historical_prediction.shape != prediction.shape
        ):
            return None
        pseudo_target = historical_prediction + dynamic_error
        correction_loss = nn.functional.mse_loss(prediction, pseudo_target)
        correction = torch.autograd.grad(
            correction_loss,
            representation,
            retain_graph=False,
            create_graph=False,
        )[0]
    resolved_c_sqrt = c_sqrt.to(device=prediction.device, dtype=prediction.dtype)
    if not torch.isfinite(resolved_c_sqrt) or resolved_c_sqrt <= 0:
        raise ValueError("PCoderN C_sqrt must be finite and positive.")
    error_scale = prediction.new_tensor(float(prediction.numel())) / resolved_c_sqrt
    return (error_scale * correction).detach()


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
    """Return the instantaneous Target Flow residual r_t = F_t - T_t."""
    if target_output is None or forward_output is None:
        return None
    return forward_output - target_output


def build_dynamic_targetflow_error(
    targetflow_residual: Optional[torch.Tensor],
    previous_error: Optional[torch.Tensor],
    sample_time,
    time_constant,
    error_gain,
):
    """Update epsilon_t = alpha*r_t + (1-K*alpha)*epsilon_(t-1)."""
    if targetflow_residual is None:
        return None

    if previous_error is None:
        previous_error = torch.zeros_like(targetflow_residual)

    sample_time_tensor = targetflow_residual.new_tensor(float(sample_time))
    time_constant_tensor = targetflow_residual.new_tensor(float(time_constant))
    error_gain_tensor = targetflow_residual.new_tensor(float(error_gain))

    if not torch.isfinite(sample_time_tensor) or sample_time_tensor <= 0:
        raise ValueError(f"sample_time must be positive, but got {float(sample_time_tensor.item())}.")
    if not torch.isfinite(time_constant_tensor) or time_constant_tensor <= 0:
        raise ValueError(f"time_constant must be positive, but got {float(time_constant_tensor.item())}.")
    if not torch.isfinite(error_gain_tensor):
        raise ValueError(f"error_gain must be finite, but got {float(error_gain_tensor.item())}.")

    integration_factor = sample_time_tensor / time_constant_tensor
    memory_factor = 1.0 - error_gain_tensor * integration_factor
    if torch.abs(memory_factor) >= 1.0:
        raise ValueError(
            "Unstable target-flow error dynamics: require "
            "abs(1 - error_gain * sample_time / time_constant) < 1, but got "
            f"memory_factor={float(memory_factor.item())}."
        )
    return integration_factor * targetflow_residual + memory_factor * previous_error


def build_temporal_prediction_error_state(
    prediction_error: Optional[torch.Tensor],
    previous_state: Optional[torch.Tensor],
    sample_time,
    time_constant,
    error_gain,
):
    """Update epsilon_t = alpha*e_t + (1-K*alpha)*epsilon_(t-1)."""
    if prediction_error is None:
        return None

    if previous_state is None:
        previous_state = torch.zeros_like(prediction_error)

    sample_time_tensor = prediction_error.new_tensor(float(sample_time))
    time_constant_tensor = prediction_error.new_tensor(float(time_constant))
    error_gain_tensor = prediction_error.new_tensor(float(error_gain))

    if not torch.isfinite(sample_time_tensor) or sample_time_tensor <= 0:
        raise ValueError(
            "temporal_error_sample_time must be positive, but got "
            f"{float(sample_time_tensor.item())}."
        )
    if not torch.isfinite(time_constant_tensor) or time_constant_tensor <= 0:
        raise ValueError(
            "temporal_error_time_constant must be positive, but got "
            f"{float(time_constant_tensor.item())}."
        )
    if not torch.isfinite(error_gain_tensor):
        raise ValueError(
            "temporal_error_gain must be finite, but got "
            f"{float(error_gain_tensor.item())}."
        )

    integration_factor = sample_time_tensor / time_constant_tensor
    memory_factor = 1.0 - error_gain_tensor * integration_factor
    if torch.abs(memory_factor) >= 1.0:
        raise ValueError(
            "Unstable temporal prediction error dynamics: require "
            "abs(1 - error_gain * sample_time / time_constant) < 1, but got "
            f"memory_factor={float(memory_factor.item())}."
        )
    return integration_factor * prediction_error + memory_factor * previous_state


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
    """Build an instant, recursive-EMA, or two-tap Target Flow residual state."""
    targetflow_residual = build_targetflow_instant_error(target_output, forward_output)
    if targetflow_residual is None:
        return None
    resolved_mode = mode or ("ema" if dynamic else "instant")
    if resolved_mode == "instant":
        return targetflow_residual
    if resolved_mode == "ema":
        memory_error = previous_error
    elif resolved_mode in {"lag1", "two_tap"}:
        memory_error = previous_instant_error
    else:
        raise ValueError(f"Unsupported target-flow error state mode: {resolved_mode}")

    return build_dynamic_targetflow_error(
        targetflow_residual,
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
