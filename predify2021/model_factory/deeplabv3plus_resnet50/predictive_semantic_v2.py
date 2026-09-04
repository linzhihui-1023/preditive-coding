"""Minimal task-aware predictive semantic state used by Predify V2.

The module deliberately contains only the three trainable pieces introduced by
V2-Minimal: a Z4 encoder, a causal residual ConvGRU predictor, and a task
update head.  The host model and its C4 writeback/decoder remain outside this
module and are frozen by the training entry point.
"""

import torch
from torch import nn

from .adapters import UNIFIED_STATE_CHANNELS
from .semantic_recurrent_predictor import ConvGRUCell


class Z4PredictiveSemanticEncoder(nn.Module):
    """Map frozen Host Z4 features to a compact predictive state."""

    def __init__(self, input_channels=UNIFIED_STATE_CHANNELS, state_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, state_channels, 1),
            nn.SiLU(),
            nn.Conv2d(state_channels, state_channels, 3, padding=1),
        )

    def forward(self, z4):
        return self.net(z4)


class Z4PredictiveSemanticDecoder(nn.Module):
    """Small reconstruction decoder used only by V2-Staged Stage A."""

    def __init__(self, state_channels=64, output_channels=UNIFIED_STATE_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(state_channels, state_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(state_channels, output_channels, 1),
        )

    def forward(self, state):
        return self.net(state)


class Z4PredictiveTemporalPredictor(nn.Module):
    """Causal residual predictor for the encoded semantic state."""

    def __init__(self, state_channels=64, hidden_channels=64):
        super().__init__()
        self.recurrent = ConvGRUCell(2 * state_channels, hidden_channels)
        self.delta = nn.Conv2d(hidden_channels, state_channels, 3, padding=1)
        # Start exactly from persistence in the predictive representation.
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, state, error, hidden=None):
        hidden = self.recurrent(torch.cat((state, error), dim=1), hidden)
        return state + self.delta(hidden), hidden


class Z4TemporalUpdateHead(nn.Module):
    """Convert current/predicted/error states into a Z4 task update."""

    def __init__(self, state_channels=64, output_channels=UNIFIED_STATE_CHANNELS):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3 * state_channels, state_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(state_channels, output_channels, 1),
        )
        # Exact identity at initialization: Z4_post == Z4.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, state, predicted_state, error):
        command = self.body(torch.cat((state, predicted_state, error), dim=1))
        # The prediction error is a necessary carrier for every task update.
        # Broadcasting its mean magnitude keeps the head lightweight while
        # enforcing the causal contract e=0 => delta_z4=0.
        error_scale = error.abs().mean(dim=1, keepdim=True)
        return command * error_scale


class PredictiveSemanticV2(nn.Module):
    """The complete minimal V2 trainable plugin.

    ``z4_post`` is always formed from the frozen Host Z4 plus the update head;
    the predictor only receives the original encoded state and prediction
    error, never the corrected feature.
    """

    def __init__(self, state_channels=64, hidden_channels=64):
        super().__init__()
        self.encoder = Z4PredictiveSemanticEncoder(
            state_channels=state_channels
        )
        self.predictor = Z4PredictiveTemporalPredictor(
            state_channels=state_channels, hidden_channels=hidden_channels
        )
        self.update_head = Z4TemporalUpdateHead(
            state_channels=state_channels, output_channels=UNIFIED_STATE_CHANNELS
        )

    def encode(self, z4):
        return self.encoder(z4)

    def predict_next(self, state, error, hidden=None):
        return self.predictor(state, error, hidden)

    def update(self, z4, state, predicted_state):
        """Return the conceptual post-Z4, error and writeback command.

        The frozen host decoder does not consume ``z4 + delta`` directly: the
        caller sends ``delta`` through the validated C4 residual writeback.
        """
        error = state - predicted_state
        # Task losses may train the encoder/update head, but must not train the
        # Predictor through this task-update path.
        pred_for_update = predicted_state.detach()
        error_for_update = error.detach()
        delta_z4 = self.update_head(state, pred_for_update, error_for_update)
        return z4 + delta_z4, error, delta_z4

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]
