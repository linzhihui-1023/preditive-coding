import torch
from torch import nn
from torch.nn import functional as F

from .adapters import UnifiedFeatures
from .space_time_memory_reader import SpaceTimeMemoryReader


DYNAMIC_ERROR_SAMPLE_TIME = 0.1035
DYNAMIC_ERROR_TIME_CONSTANT = 0.5
DYNAMIC_ERROR_GAIN = 1.0


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


class ErrorStateConvGRU(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.hidden_channels = channels
        self.gates = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, bias=False)
        self.candidate = nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False)

    def forward(self, task_error, hidden):
        if hidden is None:
            hidden = torch.zeros_like(task_error)
        gates = torch.sigmoid(self.gates(torch.cat((task_error, hidden), dim=1)))
        update, reset = gates.chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat((task_error, reset * hidden), dim=1)))
        return (1.0 - update) * hidden + update * candidate


class DynamicErrorTransition(nn.Module):
    """Condition the hidden transition on the full dynamic prediction-error state."""

    def __init__(
        self,
        channels=128,
        basis_count=3,
        sample_time=DYNAMIC_ERROR_SAMPLE_TIME,
        time_constant=DYNAMIC_ERROR_TIME_CONSTANT,
        error_gain=DYNAMIC_ERROR_GAIN,
    ):
        super().__init__()
        self.residual_scale = 0.1
        self.sample_time = float(sample_time)
        self.time_constant = float(time_constant)
        self.error_gain = float(error_gain)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.controller = nn.Linear(channels, basis_count)
        self.bases = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    groups=channels,
                    bias=False,
                )
                for _ in range(basis_count)
            ]
        )

    @property
    def integration_factor(self):
        return self.sample_time / self.time_constant

    @property
    def memory_factor(self):
        return 1.0 - self.error_gain * self.sample_time / self.time_constant

    def forward(self, prediction_error, hidden, dynamic_error, zero_dynamic=False):
        if hidden is None:
            hidden = torch.zeros_like(prediction_error)
        if dynamic_error is None:
            dynamic_error = torch.zeros_like(prediction_error)

        dynamic_error = (
            (self.sample_time / self.time_constant) * prediction_error
            + (
                1.0
                - self.error_gain * self.sample_time / self.time_constant
            )
            * dynamic_error
        )
        controller_state = torch.zeros_like(dynamic_error) if zero_dynamic else dynamic_error
        weights = torch.softmax(
            self.controller(self.pool(controller_state).flatten(1)), dim=1
        )

        transitioned = torch.zeros_like(hidden)
        for index, basis in enumerate(self.bases):
            candidate = hidden + self.residual_scale * basis(hidden)
            transitioned = transitioned + weights[:, index, None, None, None] * candidate
        return transitioned, dynamic_error, weights


class DynamicTransitionErrorStateConvGRU(ErrorStateConvGRU):
    def __init__(self, channels=128):
        super().__init__(channels)
        self.transition = DynamicErrorTransition(channels)

    def forward(
        self,
        task_error,
        prediction_error,
        hidden,
        dynamic_error,
        zero_dynamic=False,
    ):
        transitioned, dynamic_error, weights = self.transition(
            prediction_error, hidden, dynamic_error, zero_dynamic
        )
        gates = torch.sigmoid(
            self.gates(torch.cat((task_error, transitioned), dim=1))
        )
        update, reset = gates.chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat((task_error, reset * transitioned), dim=1))
        )
        hidden = (1.0 - update) * transitioned + update * candidate
        return hidden, dynamic_error, weights


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


class SemanticTemporalErrorCorrection(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.correlation = LocalFeatureCorrelation(channels)
        self.encoder = SemanticTemporalErrorEncoder(channels)
        self.error_state = ErrorStateConvGRU(channels)
        self.direct = SemanticTemporalDirectCorrection(channels)

    def forward(self, observation, predicted, semantic_context, hidden):
        error = observation - predicted
        aligned_error, raw_aligned_error, attention_weights = self.correlation(observation, predicted, semantic_context)
        task_error, error_backbone, base_error = self.encoder(
            error, aligned_error, observation, predicted, semantic_context
        )
        hidden = self.error_state(task_error, hidden)
        delta = self.direct(observation, hidden)
        entropy = -(attention_weights * (attention_weights.clamp_min(1e-12).log())).sum(dim=1)
        return error, aligned_error, task_error, hidden, delta, attention_weights.max(dim=1).values.mean(), entropy.mean(), error_backbone, base_error, raw_aligned_error


class DPCSemanticTemporalErrorCorrection(SemanticTemporalErrorCorrection):
    """Use explicit dynamic error to select the ConvGRU hidden transition."""

    uses_dynamic_transition = True

    def __init__(self, channels=128):
        super().__init__(channels)
        self.error_state = DynamicTransitionErrorStateConvGRU(channels)

    def forward(
        self,
        observation,
        predicted,
        semantic_context,
        hidden,
        dynamic_error,
        zero_dynamic=False,
    ):
        error = observation - predicted
        aligned_error, raw_aligned_error, attention_weights = self.correlation(
            observation, predicted, semantic_context
        )
        task_error, error_backbone, base_error = self.encoder(
            error, aligned_error, observation, predicted, semantic_context
        )
        hidden, dynamic_error, transition_weights = self.error_state(
            task_error, error, hidden, dynamic_error, zero_dynamic
        )
        delta = self.direct(observation, hidden)
        attention_entropy = -(
            attention_weights * attention_weights.clamp_min(1e-12).log()
        ).sum(dim=1)
        transition_entropy = -(
            transition_weights * transition_weights.clamp_min(1e-12).log()
        ).sum(dim=1)
        return {
            "error": error,
            "aligned_error": aligned_error,
            "raw_aligned_error": raw_aligned_error,
            "task_error": task_error,
            "hidden": hidden,
            "dynamic_error": dynamic_error,
            "delta": delta,
            "max_attention_weight": attention_weights.max(dim=1).values.mean(),
            "attention_entropy": attention_entropy.mean(),
            "error_backbone": error_backbone,
            "base_error": base_error,
            "transition_weights": transition_weights,
            "transition_entropy": transition_entropy.mean(),
            "dynamic_error_sample_time": self.error_state.transition.sample_time,
            "dynamic_error_time_constant": self.error_state.transition.time_constant,
            "dynamic_error_gain": self.error_state.transition.error_gain,
            "dynamic_error_integration_factor": self.error_state.transition.integration_factor,
            "dynamic_error_memory_factor": self.error_state.transition.memory_factor,
        }


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


def build_semantic_temporal_corrections():
    return nn.ModuleList(
        [SemanticTemporalErrorCorrection(), SemanticTemporalErrorCorrection()]
    ).cuda()


def build_dpc_semantic_temporal_corrections():
    return nn.ModuleList(
        [DPCSemanticTemporalErrorCorrection(), DPCSemanticTemporalErrorCorrection()]
    ).cuda()


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


def apply_semantic_temporal_corrections(corrections, observation, dynamics, semantic, hidden):
    error1, aligned_error1, task1, hidden1, delta1, max_weight1, entropy1, backbone1, base1, raw_aligned_error1 = corrections[0](
        observation.z1, dynamics.z1, semantic.z1, hidden[0]
    )
    error4, aligned_error4, task4, hidden4, delta4, max_weight4, entropy4, backbone4, base4, raw_aligned_error4 = corrections[1](
        observation.z4, dynamics.z4, semantic.z4, hidden[1]
    )
    posterior = UnifiedFeatures(
        observation.z1 + delta1,
        observation.z2,
        observation.z3,
        observation.z4 + delta4,
    )
    return posterior, (hidden1, hidden4), {
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


def apply_dpc_semantic_temporal_corrections(
    corrections,
    observation,
    dynamics,
    semantic,
    state,
    zero_dynamic=False,
):
    values1 = corrections[0](
        observation.z1,
        dynamics.z1,
        semantic.z1,
        state.hidden[0],
        state.dynamic_error[0],
        zero_dynamic,
    )
    values4 = corrections[1](
        observation.z4,
        dynamics.z4,
        semantic.z4,
        state.hidden[1],
        state.dynamic_error[1],
        zero_dynamic,
    )
    posterior = UnifiedFeatures(
        observation.z1 + values1["delta"],
        observation.z2,
        observation.z3,
        observation.z4 + values4["delta"],
    )
    new_state = state.__class__(
        hidden=(values1["hidden"], values4["hidden"]),
        dynamic_error=(values1["dynamic_error"], values4["dynamic_error"]),
    )
    return posterior, new_state, {
        "error_z1": values1["error"],
        "error_z4": values4["error"],
        "aligned_error_z1": values1["aligned_error"],
        "aligned_error_z4": values4["aligned_error"],
        "raw_aligned_error_z1": values1["raw_aligned_error"],
        "raw_aligned_error_z4": values4["raw_aligned_error"],
        "task_error_z1": values1["task_error"],
        "task_error_z4": values4["task_error"],
        "hidden_z1": values1["hidden"],
        "hidden_z4": values4["hidden"],
        "dynamic_error_z1": values1["dynamic_error"],
        "dynamic_error_z4": values4["dynamic_error"],
        "delta_z1": values1["delta"],
        "delta_z4": values4["delta"],
        "max_attention_weight_z1": values1["max_attention_weight"],
        "max_attention_weight_z4": values4["max_attention_weight"],
        "attention_entropy_z1": values1["attention_entropy"],
        "attention_entropy_z4": values4["attention_entropy"],
        "error_backbone_z1": values1["error_backbone"],
        "error_backbone_z4": values4["error_backbone"],
        "base_error_z1": values1["base_error"],
        "base_error_z4": values4["base_error"],
        "transition_weights_z1": values1["transition_weights"].mean(dim=0),
        "transition_weights_z4": values4["transition_weights"].mean(dim=0),
        "transition_entropy_z1": values1["transition_entropy"],
        "transition_entropy_z4": values4["transition_entropy"],
        "dynamic_error_sample_time": values1["dynamic_error_sample_time"],
        "dynamic_error_time_constant": values1["dynamic_error_time_constant"],
        "dynamic_error_gain": values1["dynamic_error_gain"],
        "dynamic_error_integration_factor": values1["dynamic_error_integration_factor"],
        "dynamic_error_memory_factor": values1["dynamic_error_memory_factor"],
    }


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
