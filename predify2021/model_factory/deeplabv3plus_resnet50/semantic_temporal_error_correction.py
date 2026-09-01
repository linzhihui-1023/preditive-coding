import os

import torch
from torch import nn
from torch.nn import functional as F

from .adapters import UnifiedFeatures
from .space_time_memory_reader import SpaceTimeMemoryReader


class LocalFeatureCorrelation(nn.Module):
    """Use a 3x3 local cross-attention to explain prediction error."""

    def __init__(self, channels=128, projection_channels=32):
        super().__init__()
        self.projection_channels = projection_channels
        self.scale = projection_channels ** -0.5
        self.query = nn.Conv2d(channels * 2, projection_channels, 1)
        self.key = nn.Conv2d(channels, projection_channels, 1)
        self.value = nn.Conv2d(channels, projection_channels, 1)

    def forward(self, observation, predicted, semantic_context):
        batch, _, height, width = predicted.shape
        query = self.query(torch.cat((observation, semantic_context), dim=1))
        key = self.key(predicted)
        value_observation = self.value(observation)
        value_predicted = self.value(predicted)
        key = F.unfold(key, kernel_size=3, padding=1).view(batch, self.projection_channels, 9, height, width)
        value_predicted = F.unfold(value_predicted, kernel_size=3, padding=1).view(batch, self.projection_channels, 9, height, width)
        scores = (query.unsqueeze(2) * key).sum(dim=1) * self.scale
        weights = torch.softmax(scores, dim=1)
        aligned = (weights.unsqueeze(1) * value_predicted).sum(dim=2)
        residual = value_observation - aligned
        raw_residual = value_observation - value_predicted[:, :, 4]
        return residual, raw_residual, weights


class SemanticTemporalErrorEncoder(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.error_backbone = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        self.alignment_modulation = nn.Conv2d(32, channels, 1)
        self.semantic_modulation = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, error, aligned_error, observation, predicted, semantic_context):
        error_backbone = self.error_backbone(error)
        alignment_gain = torch.tanh(self.alignment_modulation(aligned_error))
        base_error = error_backbone * (1.0 + alignment_gain)
        semantic_gain = torch.tanh(
            self.semantic_modulation(torch.cat((observation, predicted, semantic_context), dim=1))
        )
        return base_error * (1.0 + semantic_gain), error_backbone, base_error


class DynamicsGateEncoder(nn.Module):
    """Encode the analytical first-order error state as bounded gate logits.

    The encoder deliberately only modulates the GRU gates.  The analytical
    state is supplied by the caller so that this module cannot accidentally
    become another recurrent state or read a post-correction feature.
    """

    def __init__(self, channels=128, scale=1.0, gate_limit=0.25):
        super().__init__()
        if scale <= 0:
            raise ValueError("Dynamics normalization scale must be positive")
        if gate_limit <= 0:
            raise ValueError("Dynamics gate limit must be positive")
        self.register_buffer("normalization_scale", torch.tensor(float(scale)))
        self.register_buffer("gate_limit", torch.tensor(float(gate_limit)))
        self.projection = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, 2 * channels, 3, padding=1),
        )
        # Preserve the exact original ConvGRU at initialization.  The
        # preceding layers remain normally initialized so the final layer can
        # learn from the first update.
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, error, previous_dynamic_error, error_gain=1.0):
        if error.shape != previous_dynamic_error.shape:
            raise ValueError(
                "Prediction error and previous dynamics state must have the "
                f"same shape, got {tuple(error.shape)} and "
                f"{tuple(previous_dynamic_error.shape)}"
            )
        error_gain = torch.as_tensor(
            error_gain,
            device=error.device,
            dtype=error.dtype,
        )
        historical = error_gain * previous_dynamic_error
        residual = error - historical
        features = torch.cat(
            (
                historical / self.normalization_scale,
                residual / self.normalization_scale,
            ),
            dim=1,
        )
        modulation = self.gate_limit * torch.tanh(self.projection(features))
        return modulation, historical, residual


class ErrorStateConvGRU(nn.Module):
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
            if gate_modulation.shape != (task_error.shape[0], 2 * self.hidden_channels, task_error.shape[2], task_error.shape[3]):
                raise ValueError(
                    "Dynamics gate modulation has incompatible shape: "
                    f"expected {(task_error.shape[0], 2 * self.hidden_channels, task_error.shape[2], task_error.shape[3])}, "
                    f"got {tuple(gate_modulation.shape)}"
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


class SemanticTemporalDirectCorrection(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, observation, hidden):
        zero_hidden = torch.zeros_like(hidden)
        return self.network(torch.cat((observation, hidden), dim=1)) - self.network(
            torch.cat((observation, zero_hidden), dim=1)
        )


class TemporalErrorPredictionHead(nn.Module):
    """Predict the next raw prediction error from the learned state only."""

    def __init__(self, channels=128):
        super().__init__()
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, hidden):
        return self.projection(hidden)


class SemanticTemporalErrorCorrection(nn.Module):
    def __init__(
        self,
        channels=128,
        use_dynamic_error=False,
        dynamic_scale=1.0,
        dynamic_gate_limit=0.25,
        dynamic_error_gain=1.0,
        use_temporal_prediction=False,
        dynamic_sample_time=0.1035,
        dynamic_time_constant=0.5,
    ):
        super().__init__()
        self.correlation = LocalFeatureCorrelation(channels)
        self.encoder = SemanticTemporalErrorEncoder(channels)
        self.error_state = ErrorStateConvGRU(channels)
        self.direct = SemanticTemporalDirectCorrection(channels)
        self.use_dynamic_error = bool(use_dynamic_error)
        self.use_temporal_prediction = bool(
            use_temporal_prediction or use_dynamic_error
        )
        if self.use_dynamic_error:
            self.register_buffer(
                "dynamic_error_gain", torch.tensor(float(dynamic_error_gain))
            )
            self.register_buffer(
                "dynamic_sample_time", torch.tensor(float(dynamic_sample_time))
            )
            self.register_buffer(
                "dynamic_time_constant", torch.tensor(float(dynamic_time_constant))
            )
        else:
            self.dynamic_error_gain = None
            self.dynamic_sample_time = None
            self.dynamic_time_constant = None
        self.dynamic_encoder = (
            DynamicsGateEncoder(channels, dynamic_scale, dynamic_gate_limit)
            if self.use_dynamic_error
            else None
        )
        self.temporal_prediction = (
            TemporalErrorPredictionHead(channels)
            if self.use_temporal_prediction
            else None
        )
        self.last_dynamic_values = {}

    def forward(
        self,
        observation,
        predicted,
        semantic_context,
        hidden,
        previous_dynamic_error=None,
    ):
        error = observation - predicted
        aligned_error, raw_aligned_error, attention_weights = self.correlation(observation, predicted, semantic_context)
        task_error, error_backbone, base_error = self.encoder(
            error, aligned_error, observation, predicted, semantic_context
        )
        self.last_dynamic_values = {}
        gate_modulation = None
        if self.use_dynamic_error:
            if previous_dynamic_error is None:
                raise ValueError("Dynamic-aware correction requires previous_dynamic_error")
            gate_modulation, historical, residual = self.dynamic_encoder(
                error,
                previous_dynamic_error,
                error_gain=self.dynamic_error_gain,
            )
            self.last_dynamic_values = {
                "historical_dynamic_component": historical,
                "first_order_model_residual": residual,
                "gate_modulation": gate_modulation,
            }
        hidden = self.error_state(task_error, hidden, gate_modulation=gate_modulation)
        if gate_modulation is not None:
            gate_values = self.error_state.last_gate_values
            self.last_dynamic_values.update(
                {
                    "update_gate_delta": gate_values["update"] - gate_values["base_update"],
                    "reset_gate_delta": gate_values["reset"] - gate_values["base_reset"],
                    "update_gate": gate_values["update"],
                    "reset_gate": gate_values["reset"],
                }
            )
        if self.temporal_prediction is not None:
            self.last_dynamic_values["predicted_next_error"] = self.temporal_prediction(hidden)
        delta = self.direct(observation, hidden)
        entropy = -(attention_weights * (attention_weights.clamp_min(1e-12).log())).sum(dim=1)
        return error, aligned_error, task_error, hidden, delta, attention_weights.max(dim=1).values.mean(), entropy.mean(), error_backbone, base_error, raw_aligned_error


class SemanticPrototypeTargetCorrection(nn.Module):
    """Move the observation toward a frozen semantic prototype mixture."""

    def __init__(self, prototypes, channels=128, classes=19):
        super().__init__()
        self.error_correction = SemanticTemporalErrorCorrection(channels)
        self.register_buffer("prototypes", prototypes.float())
        self.target_transform = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, classes, 1),
        )
        self.gate = nn.Conv2d(channels, 1, 1, bias=False)

    def forward(self, observation, predicted, semantic_context, hidden):
        error = observation - predicted
        aligned_error, raw_aligned_error, attention_weights = self.error_correction.correlation(
            observation, predicted, semantic_context
        )
        task_error, error_backbone, base_error = self.error_correction.encoder(
            error, aligned_error, observation, predicted, semantic_context
        )
        hidden = self.error_correction.error_state(task_error, hidden)
        zero_hidden = torch.zeros_like(hidden)

        prototypes = F.normalize(self.prototypes, dim=1)
        observation_normalized = F.normalize(observation, dim=1)
        semantic_normalized = F.normalize(semantic_context, dim=1)
        observation_scores = torch.einsum("bchw,nc->bnhw", observation_normalized, prototypes)
        semantic_scores = torch.einsum("bchw,nc->bnhw", semantic_normalized, prototypes)
        prior_logits = 0.5 * (observation_scores + semantic_scores)
        target_delta = self.target_transform(
            torch.cat((observation, semantic_context, hidden), dim=1)
        ) - self.target_transform(
            torch.cat((observation, semantic_context, zero_hidden), dim=1)
        )
        target_logits = prior_logits + target_delta
        target_probability = torch.softmax(target_logits, dim=1)
        target_state = torch.einsum("bnhw,nc->bchw", target_probability, self.prototypes)
        direction = target_state - observation
        gain = torch.tanh(self.gate(hidden)).square()
        delta = gain * direction
        posterior = observation + delta
        entropy = -(attention_weights * attention_weights.clamp_min(1e-12).log()).sum(dim=1)
        return {
            "error": error,
            "aligned_error": aligned_error,
            "raw_aligned_error": raw_aligned_error,
            "task_error": task_error,
            "hidden": hidden,
            "delta": delta,
            "posterior": posterior,
            "target_logits": target_logits,
            "target_state": target_state,
            "gain": gain,
            "max_attention_weight": attention_weights.max(dim=1).values.mean(),
            "attention_entropy": entropy.mean(),
            "error_backbone": error_backbone,
            "base_error": base_error,
        }


class STCNMemoryCorrection(nn.Module):
    """Keep the existing error path and use STCN readout only for Z4."""

    def __init__(self, use_memory=False, channels=128):
        super().__init__()
        self.base = SemanticTemporalErrorCorrection(channels)
        self.use_memory = use_memory
        if use_memory:
            self.memory_reader = SpaceTimeMemoryReader(channels, 64, 4)
            self.gate = nn.Conv2d(channels, 1, 1, bias=True)

    def forward(self, observation, predicted, semantic_context, hidden, memory=()):
        base = self.base(observation, predicted, semantic_context, hidden)
        base_hidden = base[3]
        if not self.use_memory:
            return base[4], base[4], base_hidden, memory, {"delta": base[4], "gain": base_hidden.new_zeros((base_hidden.shape[0], 1, base_hidden.shape[2], base_hidden.shape[3])), "reference": observation, "time_ratios": base_hidden.new_zeros(4), "memory_entropy": base_hidden.new_zeros(()), "memory_normalized_entropy": base_hidden.new_zeros(())}
        reference, weights, memory_stats = self.memory_reader.read(observation, memory)
        zero_hidden = torch.zeros_like(base_hidden)
        gain = (torch.sigmoid(self.gate(base_hidden)) - torch.sigmoid(self.gate(zero_hidden))).abs()
        delta = gain * (reference - observation)
        return base[4], observation + delta, base_hidden, self.memory_reader.push(observation, memory), {"delta": delta, "gain": gain, "reference": reference, "weights": weights, "time_ratios": memory_stats["time_ratios"], "memory_entropy": memory_stats["entropy"], "memory_normalized_entropy": memory_stats["normalized_entropy"]}


def _env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


def build_semantic_temporal_corrections(
    use_dynamic_error=None,
    use_temporal_prediction=None,
):
    """Build the existing correction path, optionally with dynamics gates.

    The default remains the historical mainline.  Setting
    ``PREDIFY_ENABLE_DYNAMICS_ERROR=1`` (or passing ``True`` explicitly)
    creates the isolated dynamic-aware branch.
    """
    if use_dynamic_error is None:
        use_dynamic_error = os.environ.get("PREDIFY_ENABLE_DYNAMICS_ERROR", "0") == "1"
    if use_temporal_prediction is None:
        use_temporal_prediction = (
            os.environ.get("PREDIFY_ENABLE_TEMPORAL_PREDICTION", "0") == "1"
        )
    dynamic_scale_z1 = _env_float("PREDIFY_DYNAMIC_ERROR_SCALE_Z1", 1.0)
    dynamic_scale_z4 = _env_float("PREDIFY_DYNAMIC_ERROR_SCALE_Z4", 1.0)
    dynamic_gate_limit = _env_float("PREDIFY_DYNAMIC_ERROR_GATE_LIMIT", 0.25)
    dynamic_gain = _env_float("PREDIFY_DYNAMIC_ERROR_GAIN", 1.0)
    sample_time = _env_float("PREDIFY_DYNAMIC_ERROR_SAMPLE_TIME", 0.1035)
    time_constant = _env_float("PREDIFY_DYNAMIC_ERROR_TIME_CONSTANT", 0.5)
    if use_dynamic_error:
        if time_constant <= 0:
            raise ValueError("Dynamics time constant must be positive")
        effective_q = sample_time * dynamic_gain / time_constant
        if not 0.0 < effective_q <= 1.0:
            raise ValueError(
                "Dynamic-aware ConvGRU V1 requires "
                "0 < Ts * K_e / tau_e <= 1, but got "
                f"q={effective_q}."
            )
    return nn.ModuleList(
        [
            SemanticTemporalErrorCorrection(
                use_dynamic_error=use_dynamic_error,
                dynamic_scale=dynamic_scale_z1,
                dynamic_gate_limit=dynamic_gate_limit,
                dynamic_error_gain=dynamic_gain,
                use_temporal_prediction=use_temporal_prediction,
                dynamic_sample_time=sample_time,
                dynamic_time_constant=time_constant,
            ),
            SemanticTemporalErrorCorrection(
                use_dynamic_error=use_dynamic_error,
                dynamic_scale=dynamic_scale_z4,
                dynamic_gate_limit=dynamic_gate_limit,
                dynamic_error_gain=dynamic_gain,
                use_temporal_prediction=use_temporal_prediction,
                dynamic_sample_time=sample_time,
                dynamic_time_constant=time_constant,
            ),
        ]
    ).cuda()


def build_dynamic_semantic_temporal_corrections():
    """Explicit constructor for the opt-in dynamics experiment branch."""
    return build_semantic_temporal_corrections(use_dynamic_error=True)


def build_semantic_prototype_corrections(prototypes_z1=None, prototypes_z4=None):
    if prototypes_z1 is None:
        prototypes_z1 = torch.zeros(19, 128)
    if prototypes_z4 is None:
        prototypes_z4 = torch.zeros(19, 128)
    return nn.ModuleList(
        [
            SemanticPrototypeTargetCorrection(prototypes_z1),
            SemanticPrototypeTargetCorrection(prototypes_z4),
        ]
    ).cuda()


def build_stcn_memory_corrections():
    return nn.ModuleList([STCNMemoryCorrection(False), STCNMemoryCorrection(True)]).cuda()


def apply_semantic_temporal_corrections(
    corrections,
    observation,
    dynamics,
    semantic,
    hidden,
    previous_dynamic_error=None,
):
    error1, aligned_error1, task1, hidden1, delta1, max_weight1, entropy1, backbone1, base1, raw_aligned_error1 = corrections[0](
        observation.z1,
        dynamics.z1,
        semantic.z1,
        hidden[0],
        None if previous_dynamic_error is None else previous_dynamic_error[0],
    )
    error4, aligned_error4, task4, hidden4, delta4, max_weight4, entropy4, backbone4, base4, raw_aligned_error4 = corrections[1](
        observation.z4,
        dynamics.z4,
        semantic.z4,
        hidden[1],
        None if previous_dynamic_error is None else previous_dynamic_error[1],
    )
    posterior = UnifiedFeatures(
        observation.z1 + delta1,
        observation.z2,
        observation.z3,
        observation.z4 + delta4,
    )
    values = {
        "error_z1": error1,
        "error_z4": error4,
        "aligned_error_z1": aligned_error1,
        "aligned_error_z4": aligned_error4,
        "task_error_z1": task1,
        "task_error_z4": task4,
        "delta_z1": delta1,
        "delta_z4": delta4,
        "max_attention_weight_z1": max_weight1,
        "max_attention_weight_z4": max_weight4,
        "attention_entropy_z1": entropy1,
        "attention_entropy_z4": entropy4,
        "error_backbone_z1": backbone1,
        "error_backbone_z4": backbone4,
        "base_error_z1": base1,
        "base_error_z4": base4,
        "raw_aligned_error_z1": raw_aligned_error1,
        "raw_aligned_error_z4": raw_aligned_error4,
    }
    for level, correction in (("z1", corrections[0]), ("z4", corrections[1])):
        if correction.last_dynamic_values:
            for name, value in correction.last_dynamic_values.items():
                values[f"{name}_{level}"] = value
    return posterior, (hidden1, hidden4), values


def apply_semantic_prototype_corrections(corrections, observation, dynamics, semantic, hidden):
    values1 = corrections[0](observation.z1, dynamics.z1, semantic.z1, hidden[0])
    values4 = corrections[1](observation.z4, dynamics.z4, semantic.z4, hidden[1])
    posterior = UnifiedFeatures(
        values1["posterior"],
        observation.z2,
        observation.z3,
        values4["posterior"],
    )
    return posterior, (values1["hidden"], values4["hidden"]), {
        "error_z1": values1["error"],
        "error_z4": values4["error"],
        "aligned_error_z1": values1["aligned_error"],
        "aligned_error_z4": values4["aligned_error"],
        "raw_aligned_error_z1": values1["raw_aligned_error"],
        "raw_aligned_error_z4": values4["raw_aligned_error"],
        "task_error_z1": values1["task_error"],
        "task_error_z4": values4["task_error"],
        "delta_z1": values1["delta"],
        "delta_z4": values4["delta"],
        "target_logits_z1": values1["target_logits"],
        "target_logits_z4": values4["target_logits"],
        "target_state_z1": values1["target_state"],
        "target_state_z4": values4["target_state"],
        "gain_z1": values1["gain"],
        "gain_z4": values4["gain"],
        "max_attention_weight_z1": values1["max_attention_weight"],
        "max_attention_weight_z4": values4["max_attention_weight"],
        "attention_entropy_z1": values1["attention_entropy"],
        "attention_entropy_z4": values4["attention_entropy"],
    }
