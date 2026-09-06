"""Stateful semantic hysteresis for C-V4.

The controller is deliberately a temporal decision module, not a segmentation
head. It never receives C1 appearance features. It compares the frozen current
C-V3 semantic hypothesis against a motion-warped previous frozen C-V3
hypothesis, tracks an explicit Euler-discretized prediction-error state, and
uses a small ConvGRU to decide whether history should be kept at semantic
conflicts.

The controller output is NOT fed back as the next semantic-history candidate.
Only the frozen C-V3 output is propagated across time. This prevents an
incorrect Keep decision from becoming a self-reinforcing autoregressive
semantic history.
"""

import torch
from torch import nn

from .semantic_recurrent_predictor import ConvGRUCell


class EulerDynamicsError:
    """Explicit first-order prediction-error dynamics.

    Continuous definition:
        tau_e * d epsilon / dt = e - K_e * epsilon

    Forward Euler discretization:
        epsilon_t = (1 - dt*K_e/tau_e) * epsilon_{t-1}
                    + (dt/tau_e) * e_t

    This state is evidence for temporal decision making only. It does not
    directly generate semantic correction logits.
    """

    def __init__(self, tau_e=4.0, k_e=1.0, dt=1.0):
        self.tau_e = float(tau_e)
        self.k_e = float(k_e)
        self.dt = float(dt)
        if self.tau_e <= 0.0:
            raise ValueError("tau_e must be positive")
        if self.k_e < 0.0:
            raise ValueError("k_e must be non-negative")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        self.alpha = self.dt / self.tau_e
        self.decay = 1.0 - self.dt * self.k_e / self.tau_e
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(
                "Euler dynamics must have decay in [0,1]; adjust tau_e, k_e, dt"
            )

    def step(self, prediction_error, previous_state=None):
        if previous_state is None:
            previous_state = torch.zeros_like(prediction_error)
        if previous_state.shape != prediction_error.shape:
            raise ValueError("Dynamics state and prediction error must share shape")
        return self.decay * previous_state + self.alpha * prediction_error

    def config(self):
        return {
            "tau_e": self.tau_e,
            "k_e": self.k_e,
            "dt": self.dt,
            "alpha": self.alpha,
            "decay": self.decay,
        }


class StatefulSemanticHysteresisController(nn.Module):
    """Decide Keep-History vs Use-Current at Base/History semantic conflicts.

    Evidence contains only semantic/temporal decision signals:
      P_current, P_history, e_t, epsilon_t,
      confidence margins for both candidates,
      frozen T, frozen Q_mem, and predicted-motion history validity.

    No C1/current appearance feature is exposed to this module.

    The final head is exactly zero-initialized. With the strict inference rule
    `keep = keep_logit > 0`, step zero therefore reproduces frozen C-V3 exactly.
    """

    def __init__(self, num_classes=19, hidden_channels=32):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)

        # P_current + P_history + e_t + epsilon_t + two margins + T + Q + valid.
        self.input_channels = 4 * self.num_classes + 5
        groups = 8 if self.hidden_channels % 8 == 0 else 1
        self.pre = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_channels),
            nn.SiLU(),
        )
        self.recurrent = ConvGRUCell(self.hidden_channels, self.hidden_channels)
        self.keep_head = nn.Conv2d(self.hidden_channels, 1, 1, bias=True)
        nn.init.zeros_(self.keep_head.weight)
        nn.init.zeros_(self.keep_head.bias)

    @staticmethod
    def confidence_margin(probability):
        if probability.ndim != 4 or probability.shape[1] < 2:
            raise ValueError("Probability must be BCHW with at least two classes")
        top2 = probability.topk(k=2, dim=1).values
        return top2[:, :1] - top2[:, 1:2]

    def forward(
        self,
        current_probability,
        history_probability,
        prediction_error,
        dynamics_error,
        transportability_low,
        memory_reliability_low,
        history_valid_low,
        hidden=None,
    ):
        expected = current_probability.shape
        for name, tensor in (
            ("history_probability", history_probability),
            ("prediction_error", prediction_error),
            ("dynamics_error", dynamics_error),
        ):
            if tensor.shape != expected:
                raise ValueError(f"{name} must match current_probability shape")
        spatial = expected[-2:]
        for name, tensor in (
            ("transportability_low", transportability_low),
            ("memory_reliability_low", memory_reliability_low),
            ("history_valid_low", history_valid_low),
        ):
            if tensor.ndim != 4 or tensor.shape[1] != 1 or tensor.shape[-2:] != spatial:
                raise ValueError(f"{name} must be Bx1xHxW at controller resolution")

        current_margin = self.confidence_margin(current_probability)
        history_margin = self.confidence_margin(history_probability)
        evidence = torch.cat(
            (
                current_probability,
                history_probability,
                prediction_error,
                dynamics_error,
                current_margin,
                history_margin,
                transportability_low,
                memory_reliability_low,
                history_valid_low,
            ),
            dim=1,
        )
        encoded = self.pre(evidence)
        hidden = self.recurrent(encoded, hidden)
        keep_logit = self.keep_head(hidden)
        return {
            "keep_logit": keep_logit,
            "hidden": hidden,
            "current_margin": current_margin,
            "history_margin": history_margin,
        }
