import torch
from torch import nn
from torch.nn import functional as F

from .adapters import UnifiedFeatures


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


def build_semantic_temporal_corrections():
    return nn.ModuleList(
        [SemanticTemporalErrorCorrection(), SemanticTemporalErrorCorrection()]
    ).cuda()


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
