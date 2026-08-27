import torch
from torch import nn
from torch.nn import functional as F

from .adapters import UnifiedFeatures
from .semantic_recurrent_predictor import ConvGRUCell


class LocalFeatureCorrelation(nn.Module):
    """Use a 3x3 local correspondence to explain prediction error."""

    def __init__(self, channels=128, projection_channels=32):
        super().__init__()
        self.query = nn.Conv2d(channels * 2, projection_channels, 1)
        self.key = nn.Conv2d(channels, projection_channels, 1)

    def forward(self, observation, predicted, semantic_context):
        batch, _, height, width = predicted.shape
        query = F.normalize(self.query(torch.cat((observation, semantic_context), dim=1)), dim=1)
        key = F.normalize(self.key(predicted), dim=1)
        key = F.unfold(key, kernel_size=3, padding=1).view(batch, -1, 9, height, width)
        value = F.unfold(predicted, kernel_size=3, padding=1).view(batch, predicted.shape[1], 9, height, width)
        scores = (query.unsqueeze(2) * key).sum(dim=1)
        weights = torch.softmax(scores, dim=2)
        return (weights.unsqueeze(1) * value).sum(dim=2)


class SemanticTemporalErrorEncoder(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.error_backbone = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.context = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, error, aligned_error, observation, predicted, semantic_context):
        error_backbone = self.error_backbone(torch.cat((error, aligned_error), dim=1))
        return self.context(
            torch.cat((error_backbone, observation, predicted, semantic_context), dim=1)
        )


class ErrorStateConvGRU(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.cell = ConvGRUCell(channels, channels)

    def forward(self, task_error, hidden):
        return self.cell(task_error, hidden)


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
        aligned_predicted = self.correlation(observation, predicted, semantic_context)
        aligned_error = observation - aligned_predicted
        task_error = self.encoder(
            error, aligned_error, observation, predicted, semantic_context
        )
        hidden = self.error_state(task_error, hidden)
        delta = self.direct(observation, hidden)
        return error, aligned_error, task_error, hidden, delta


def build_semantic_temporal_corrections():
    return nn.ModuleList(
        [SemanticTemporalErrorCorrection(), SemanticTemporalErrorCorrection()]
    ).cuda()


def apply_semantic_temporal_corrections(corrections, observation, dynamics, semantic, hidden):
    error1, aligned_error1, task1, hidden1, delta1 = corrections[0](
        observation.z1, dynamics.z1, semantic.z1, hidden[0]
    )
    error4, aligned_error4, task4, hidden4, delta4 = corrections[1](
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
    }
