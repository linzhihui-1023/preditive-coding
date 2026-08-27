import torch
from torch import nn
from torch.nn import functional as F

from .adapters import UnifiedFeatures, UNIFIED_STATE_CHANNELS


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels=128):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, x, hidden):
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        gates = torch.sigmoid(self.gates(torch.cat((x, hidden), dim=1)))
        update, reset = gates.chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat((x, reset * hidden), dim=1)))
        return (1 - update) * hidden + update * candidate


class SemanticRecurrentPredictor(nn.Module):
    """Causal z4-to-z1 recurrent predictor; z2/z3 remain persistence."""

    def __init__(self, hidden_channels=128):
        super().__init__()
        self.z4_recurrent = ConvGRUCell(2 * UNIFIED_STATE_CHANNELS, hidden_channels)
        self.z1_recurrent = ConvGRUCell(3 * UNIFIED_STATE_CHANNELS, hidden_channels)
        self.z4_delta = nn.Conv2d(hidden_channels, UNIFIED_STATE_CHANNELS, 3, padding=1)
        self.z1_delta = nn.Conv2d(hidden_channels, UNIFIED_STATE_CHANNELS, 3, padding=1)

    def initial_state(self):
        return None, None

    def step(self, observation, error, hidden4=None, hidden1=None, persist_z4=False, detach_high_to_low=False):
        input4 = torch.cat((observation.z4, error.z4), dim=1)
        hidden4 = self.z4_recurrent(input4, hidden4)
        hidden4_for_z1 = hidden4.detach() if detach_high_to_low else hidden4
        hidden4_up = F.interpolate(hidden4_for_z1, size=observation.z1.shape[-2:], mode="bilinear", align_corners=False)
        input1 = torch.cat((observation.z1, error.z1, hidden4_up), dim=1)
        hidden1 = self.z1_recurrent(input1, hidden1)
        predicted_z4 = observation.z4 if persist_z4 else observation.z4 + self.z4_delta(hidden4)
        predicted = UnifiedFeatures(
            observation.z1 + self.z1_delta(hidden1),
            observation.z2,
            observation.z3,
            predicted_z4,
        )
        return predicted, hidden4, hidden1


class RoleSeparatedRecurrentPredictor(nn.Module):
    """Causal dynamics predictions and semantic diagnostic context use separate states."""

    def __init__(self, hidden_channels=128):
        super().__init__()
        self.z4_dyn_recurrent = ConvGRUCell(2 * UNIFIED_STATE_CHANNELS, hidden_channels)
        self.z4_sem_recurrent = ConvGRUCell(2 * UNIFIED_STATE_CHANNELS, hidden_channels)
        self.z1_dyn_recurrent = ConvGRUCell(3 * UNIFIED_STATE_CHANNELS, hidden_channels)
        self.z1_sem_recurrent = ConvGRUCell(3 * UNIFIED_STATE_CHANNELS, hidden_channels)
        self.z4_dyn_delta = nn.Conv2d(hidden_channels, UNIFIED_STATE_CHANNELS, 3, padding=1)
        self.z4_sem_delta = nn.Conv2d(hidden_channels, UNIFIED_STATE_CHANNELS, 3, padding=1)
        self.z1_dyn_delta = nn.Conv2d(hidden_channels, UNIFIED_STATE_CHANNELS, 3, padding=1)
        self.z1_sem_delta = nn.Conv2d(hidden_channels, UNIFIED_STATE_CHANNELS, 3, padding=1)

    def initial_state(self):
        return None, None, None, None

    def step(self, observation, dynamics_error, h4_dyn=None, h4_sem=None, h1_dyn=None, h1_sem=None):
        dyn4_input = torch.cat((observation.z4, dynamics_error.z4), dim=1)
        h4_dyn = self.z4_dyn_recurrent(dyn4_input, h4_dyn)
        sem4_input = torch.cat((observation.z4, dynamics_error.z4.detach()), dim=1)
        h4_sem = self.z4_sem_recurrent(sem4_input, h4_sem)
        h4_dyn_up = F.interpolate(h4_dyn, size=observation.z1.shape[-2:], mode="bilinear", align_corners=False)
        dyn1_input = torch.cat((observation.z1, dynamics_error.z1, h4_dyn_up), dim=1)
        h1_dyn = self.z1_dyn_recurrent(dyn1_input, h1_dyn)
        h4_sem_up = F.interpolate(h4_sem, size=observation.z1.shape[-2:], mode="bilinear", align_corners=False)
        sem1_input = torch.cat((observation.z1, dynamics_error.z1.detach(), h4_sem_up), dim=1)
        h1_sem = self.z1_sem_recurrent(sem1_input, h1_sem)
        dynamics_prediction = UnifiedFeatures(
            observation.z1 + self.z1_dyn_delta(h1_dyn),
            observation.z2,
            observation.z3,
            observation.z4 + self.z4_dyn_delta(h4_dyn),
        )
        semantic_diagnostic = UnifiedFeatures(
            dynamics_prediction.z1.detach() - self.z1_sem_delta(h1_sem),
            dynamics_prediction.z2.detach(),
            dynamics_prediction.z3.detach(),
            dynamics_prediction.z4.detach() - self.z4_sem_delta(h4_sem),
        )
        return dynamics_prediction, semantic_diagnostic, h4_dyn, h4_sem, h1_dyn, h1_sem
