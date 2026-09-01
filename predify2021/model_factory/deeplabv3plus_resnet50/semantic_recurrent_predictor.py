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

    def step(self, observation, prediction_error, hidden4=None, hidden1=None, persist_z4=False, detach_high_to_low=False):
        input4 = torch.cat((observation.z4, prediction_error.z4), dim=1)
        hidden4 = self.z4_recurrent(input4, hidden4)
        hidden4_for_z1 = hidden4.detach() if detach_high_to_low else hidden4
        hidden4_up = F.interpolate(hidden4_for_z1, size=observation.z1.shape[-2:], mode="bilinear", align_corners=False)
        input1 = torch.cat((observation.z1, prediction_error.z1, hidden4_up), dim=1)
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

    def step(self, observation, prediction_error, h4_dyn=None, h4_sem=None, h1_dyn=None, h1_sem=None):
        dyn4_input = torch.cat((observation.z4, prediction_error.z4), dim=1)
        h4_dyn = self.z4_dyn_recurrent(dyn4_input, h4_dyn)
        sem4_input = torch.cat((observation.z4, prediction_error.z4.detach()), dim=1)
        h4_sem = self.z4_sem_recurrent(sem4_input, h4_sem)
        h4_dyn_up = F.interpolate(h4_dyn, size=observation.z1.shape[-2:], mode="bilinear", align_corners=False)
        dyn1_input = torch.cat((observation.z1, prediction_error.z1, h4_dyn_up), dim=1)
        h1_dyn = self.z1_dyn_recurrent(dyn1_input, h1_dyn)
        h4_sem_up = F.interpolate(h4_sem, size=observation.z1.shape[-2:], mode="bilinear", align_corners=False)
        sem1_input = torch.cat((observation.z1, prediction_error.z1.detach(), h4_sem_up), dim=1)
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


class SemanticPredictionErrorEncoder(nn.Module):
    """Encode Z4 prediction error into a semantic innovation representation."""

    def __init__(self, channels=UNIFIED_STATE_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, prediction_error_z4):
        return self.net(prediction_error_z4)


class ErrorGuidedSemanticRecurrentCell(nn.Module):
    """Single error-driven recurrent update for the temporal semantic correction state.

    The direct carrier of the state change is encoded prediction error. Observation
    and the previous state modulate how that error changes the semantic state; there
    is no second external reliability gate.
    """

    def __init__(self, channels=UNIFIED_STATE_CHANNELS, hidden_channels=128):
        super().__init__()
        self.hidden_channels = hidden_channels
        context_channels = channels + channels + hidden_channels
        self.update_gain = nn.Conv2d(context_channels, hidden_channels, 3, padding=1)
        self.error_drive = nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False)
        self.context_modulation = nn.Conv2d(context_channels, hidden_channels, 3, padding=1)

    def forward(self, observation_z4, error_innovation, hidden):
        if hidden is None:
            hidden = torch.zeros(
                observation_z4.shape[0],
                self.hidden_channels,
                observation_z4.shape[2],
                observation_z4.shape[3],
                device=observation_z4.device,
                dtype=observation_z4.dtype,
            )
        context = torch.cat((hidden, observation_z4, error_innovation), dim=1)
        gain = torch.sigmoid(self.update_gain(context))
        error_drive = torch.tanh(self.error_drive(error_innovation))
        modulation = torch.tanh(self.context_modulation(context))
        state_innovation = gain * error_drive * (1.0 + modulation)
        return hidden + state_innovation, gain, state_innovation


class SemanticRestorationHead(nn.Module):
    """Read error-driven semantic state as a same-frame Z4 residual.

    Observation can modulate the state-to-residual mapping, but it cannot create a
    residual when the semantic correction state is zero. This prevents a direct
    single-frame observation bypass around the prediction-error-driven state.
    """

    def __init__(self, channels=UNIFIED_STATE_CHANNELS, hidden_channels=128):
        super().__init__()
        self.state_projection = nn.Conv2d(
            hidden_channels, hidden_channels, 3, padding=1, bias=False
        )
        self.observation_modulation = nn.Conv2d(
            channels, hidden_channels, 1, bias=False
        )
        self.output = nn.Conv2d(hidden_channels, channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, semantic_state, observation_z4):
        state_command = torch.tanh(self.state_projection(semantic_state))
        observation_gain = torch.tanh(self.observation_modulation(observation_z4))
        return self.output(state_command * (1.0 + observation_gain))


class ErrorGuidedSemanticRestorationPredictor(nn.Module):
    """Prediction-error-guided same-frame Z4 semantic restoration.

    Dynamics keeps the proven RoleSeparatedRecurrentPredictor architecture and is
    loaded from its checkpoint. Semantic V2 is Z4-only and follows the causal order:
      prior prediction for t -> observation t -> prediction error t -> restore t
      -> dynamics update -> prediction for t+1.
    """

    DYNAMICS_MODULES = (
        "z4_dyn_recurrent",
        "z4_dyn_delta",
        "z1_dyn_recurrent",
        "z1_dyn_delta",
    )

    def __init__(self, hidden_channels=128):
        super().__init__()
        self.z4_dyn_recurrent = ConvGRUCell(2 * UNIFIED_STATE_CHANNELS, hidden_channels)
        self.z1_dyn_recurrent = ConvGRUCell(3 * UNIFIED_STATE_CHANNELS, hidden_channels)
        self.z4_dyn_delta = nn.Conv2d(hidden_channels, UNIFIED_STATE_CHANNELS, 3, padding=1)
        self.z1_dyn_delta = nn.Conv2d(hidden_channels, UNIFIED_STATE_CHANNELS, 3, padding=1)

        self.semantic_error_encoder = SemanticPredictionErrorEncoder()
        self.semantic_recurrent = ErrorGuidedSemanticRecurrentCell(hidden_channels=hidden_channels)
        self.semantic_restoration_head = SemanticRestorationHead(hidden_channels=hidden_channels)

    def initial_dynamics_state(self):
        return None, None

    def initial_semantic_state(self):
        return None

    def load_dynamics_from_role_separated_state_dict(self, state_dict):
        for module_name in self.DYNAMICS_MODULES:
            prefix = module_name + "."
            module_state = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if not module_state:
                raise RuntimeError(f"Missing dynamics module in source checkpoint: {module_name}")
            getattr(self, module_name).load_state_dict(module_state, strict=True)

    def freeze_dynamics(self):
        for module_name in self.DYNAMICS_MODULES:
            getattr(self, module_name).requires_grad_(False)

    def semantic_parameters(self):
        modules = (
            self.semantic_error_encoder,
            self.semantic_recurrent,
            self.semantic_restoration_head,
        )
        return [parameter for module in modules for parameter in module.parameters()]

    def restore_current(self, observation, current_prediction, semantic_hidden=None):
        prediction_error_z4 = observation.z4 - current_prediction.z4
        error_innovation = self.semantic_error_encoder(prediction_error_z4)
        semantic_hidden, update_gain, state_innovation = self.semantic_recurrent(
            observation.z4,
            error_innovation,
            semantic_hidden,
        )
        restoration_delta_z4 = self.semantic_restoration_head(
            semantic_hidden,
            observation.z4,
        )
        restored = UnifiedFeatures(
            observation.z1,
            observation.z2,
            observation.z3,
            observation.z4 + restoration_delta_z4,
        )
        diagnostics = {
            "prediction_error_z4": prediction_error_z4,
            "error_innovation": error_innovation,
            "update_gain": update_gain,
            "state_innovation": state_innovation,
            "restoration_delta_z4": restoration_delta_z4,
        }
        return restored, semantic_hidden, diagnostics

    def predict_next(self, observation, prediction_error, h4_dyn=None, h1_dyn=None):
        dyn4_input = torch.cat((observation.z4, prediction_error.z4), dim=1)
        h4_dyn = self.z4_dyn_recurrent(dyn4_input, h4_dyn)
        h4_dyn_up = F.interpolate(
            h4_dyn,
            size=observation.z1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        dyn1_input = torch.cat((observation.z1, prediction_error.z1, h4_dyn_up), dim=1)
        h1_dyn = self.z1_dyn_recurrent(dyn1_input, h1_dyn)
        next_prediction = UnifiedFeatures(
            observation.z1 + self.z1_dyn_delta(h1_dyn),
            observation.z2,
            observation.z3,
            observation.z4 + self.z4_dyn_delta(h4_dyn),
        )
        return next_prediction, h4_dyn, h1_dyn
