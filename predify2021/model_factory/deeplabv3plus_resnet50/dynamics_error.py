import torch
from torch import nn


class DynamicsGateEncoder(nn.Module):
    """Encode the analytical first-order error state as bounded gate logits."""

    VALID_MODES = {"error_only", "historical_only", "historical_residual"}

    def __init__(self, channels=128, scale=1.0, gate_limit=0.25, mode="historical_residual"):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown dynamics input mode {mode!r}; expected one of "
                f"{sorted(self.VALID_MODES)}"
            )
        if scale <= 0:
            raise ValueError("Dynamics normalization scale must be positive")
        if gate_limit <= 0:
            raise ValueError("Dynamics gate limit must be positive")
        self.register_buffer("normalization_scale", torch.tensor(float(scale)))
        self.register_buffer("gate_limit", torch.tensor(float(gate_limit)))
        self.mode = mode
        self.projection = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, 2 * channels, 3, padding=1),
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, error, previous_dynamic_error, error_gain=1.0):
        if error.shape != previous_dynamic_error.shape:
            raise ValueError(
                "Prediction error and previous dynamics state must have the "
                f"same shape, got {tuple(error.shape)} and "
                f"{tuple(previous_dynamic_error.shape)}"
            )
        error_gain = torch.as_tensor(error_gain, device=error.device, dtype=error.dtype)
        historical = error_gain * previous_dynamic_error
        residual = error - historical
        if self.mode == "error_only":
            encoder_left, encoder_right = error, torch.zeros_like(error)
        elif self.mode == "historical_only":
            encoder_left, encoder_right = historical, torch.zeros_like(historical)
        else:
            encoder_left, encoder_right = historical, residual
        features = torch.cat(
            (
                encoder_left / self.normalization_scale,
                encoder_right / self.normalization_scale,
            ),
            dim=1,
        )
        modulation = self.gate_limit * torch.tanh(self.projection(features))
        return modulation, historical, residual


class ErrorStateConvGRU(nn.Module):
    """ConvGRU state transition with optional residual gate-logit evidence."""

    def __init__(self, channels=128):
        super().__init__()
        self.hidden_channels = channels
        self.gates = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, bias=False)
        self.candidate = nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False)

    def forward(self, task_error, hidden, gate_modulation=None):
        if hidden is None:
            hidden = torch.zeros_like(task_error)
        gate_logits = self.gates(torch.cat((task_error, hidden), dim=1))
        base_gates = torch.sigmoid(gate_logits)
        if gate_modulation is not None:
            expected_shape = (
                task_error.shape[0],
                2 * self.hidden_channels,
                task_error.shape[2],
                task_error.shape[3],
            )
            if gate_modulation.shape != expected_shape:
                raise ValueError(
                    "Dynamics gate modulation has incompatible shape: "
                    f"expected {expected_shape}, got {tuple(gate_modulation.shape)}"
                )
            gate_logits = gate_logits + gate_modulation
        gates = torch.sigmoid(gate_logits)
        update, reset = gates.chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat((task_error, reset * hidden), dim=1)))
        self.last_gate_values = {
            "base_update": base_gates[:, : self.hidden_channels],
            "base_reset": base_gates[:, self.hidden_channels :],
            "update": update,
            "reset": reset,
        }
        return (1.0 - update) * hidden + update * candidate


class TemporalErrorPredictionHead(nn.Module):
    """Predict the next raw prediction error from the learned state only."""

    def __init__(self, channels=128):
        super().__init__()
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, hidden):
        return self.projection(hidden)
