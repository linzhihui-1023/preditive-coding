from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    add_frame_noise,
    detach_state,
    encode_image,
    error_state,
    load_components,
    load_image,
    next_role_prediction,
    update_dynamic_error,
    zero_state,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import DirectStateCorrection, UnifiedFeatures


ALPHA = 0.207
BETA = 0.793
CORRECTION_INDICES = (0, 3)


def build_corrections():
    return torch.nn.ModuleList([DirectStateCorrection(), DirectStateCorrection()]).cuda()


def zero_initialize(corrections):
    for correction in corrections:
        torch.nn.init.zeros_(correction.output_projection.weight)
        torch.nn.init.zeros_(correction.output_projection.bias)


def direct_posterior(observation, error, dynamic_error, corrections):
    delta1 = corrections[0](observation.z1, error.z1, dynamic_error.z1)
    delta4 = corrections[1](observation.z4, error.z4, dynamic_error.z4)
    return UnifiedFeatures(
        observation.z1 + delta1,
        observation.z2,
        observation.z3,
        observation.z4 + delta4,
    ), (delta1, delta4)


def raw_state_mse(posterior, clean_state):
    z1 = F.mse_loss(posterior.z1, clean_state.z1)
    z4 = F.mse_loss(posterior.z4, clean_state.z4)
    return z1, z4, z1 + z4


def relative_state_loss(posterior, clean_state, observation):
    post_z1, post_z4, _ = raw_state_mse(posterior, clean_state)
    obs_z1 = F.mse_loss(observation.z1, clean_state.z1).detach()
    obs_z4 = F.mse_loss(observation.z4, clean_state.z4).detach()
    return 0.5 * (
        post_z1 / (obs_z1 + 1e-12) + post_z4 / (obs_z4 + 1e-12)
    )


def load_direct_corrections(path):
    corrections = build_corrections()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for position, index in enumerate(CORRECTION_INDICES):
        corrections[position].load_state_dict(payload["corrections"][str(index)], strict=True)
    corrections.requires_grad_(False)
    corrections.eval()
    return corrections, payload


def load_role_components(static_checkpoint, adapter_checkpoint, predictor_checkpoint, writeback_checkpoint):
    return load_components(
        str(static_checkpoint),
        str(adapter_checkpoint),
        str(predictor_checkpoint),
        str(writeback_checkpoint),
    )


def make_paths():
    return {
        "static": Path(STATIC_CHECKPOINT_DEFAULT),
        "adapter": Path(ADAPTER_CHECKPOINT_DEFAULT),
        "predictor": Path(ROLE_PREDICTOR_CHECKPOINT_DEFAULT),
        "writeback": Path(WRITEBACK_CHECKPOINT_DEFAULT),
    }
