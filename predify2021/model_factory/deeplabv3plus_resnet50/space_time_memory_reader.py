import math

import torch
from torch import nn
from torch.nn import functional as F


class SpaceTimeMemoryReader(nn.Module):
    """Minimal STCN-style affinity/readout for observation feature memory."""

    def __init__(self, channels=128, key_channels=64, max_memory=4):
        super().__init__()
        self.key_projection = nn.Conv2d(channels, key_channels, 1, bias=False)
        self.value_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.key_channels = key_channels
        self.max_memory = max_memory

    def push(self, observation, memory):
        item = (self.key_projection(observation), self.value_projection(observation))
        return (*memory, item)[-self.max_memory:]

    def read(self, observation, memory):
        if not memory:
            return observation, None, {"time_ratios": torch.zeros(4, device=observation.device), "entropy": observation.new_zeros(()), "normalized_entropy": observation.new_zeros(())}
        batch, _, height, width = observation.shape
        keys = torch.stack([item[0] for item in memory], dim=2)
        values = torch.stack([item[1] for item in memory], dim=2)
        query = self.key_projection(observation)
        memory_keys = keys.flatten(2)
        query_key = query.flatten(2)
        affinity = 2.0 * torch.einsum("bcn,bcm->bnm", memory_keys, query_key)
        affinity = affinity - memory_keys.square().sum(dim=1, keepdim=False).unsqueeze(-1)
        affinity = affinity * (self.key_channels ** -0.5)
        weights = torch.softmax(affinity, dim=1)
        readout = torch.einsum("bcn,bnm->bcm", values.flatten(2), weights).view(batch, values.shape[1], height, width)
        time_count = len(memory)
        time_weights = weights.view(batch, time_count, height, width, height * width).sum(dim=(2, 3, 4)) / (height * width)
        ratios = torch.zeros(4, device=observation.device, dtype=observation.dtype)
        ratios[:time_count] = time_weights.mean(dim=0)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1).mean()
        normalized_entropy = entropy / math.log(weights.shape[1])
        return readout, weights, {"time_ratios": ratios, "entropy": entropy, "normalized_entropy": normalized_entropy}


def detach_memory(memory):
    return tuple((key.detach(), value.detach()) for key, value in memory)
