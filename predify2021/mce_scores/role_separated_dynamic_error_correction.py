from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import load_writeback_checkpoint
from predify2021.mce_scores.train_kitti_step_fixed_adapter_predictor import ADAPTER_CHECKPOINT_DEFAULT, STATIC_CHECKPOINT_DEFAULT
from predify2021.mce_scores.train_kitti_step_state_predictor import load_static_kitti_checkpoint
from predify2021.model_factory.deeplabv3plus_resnet50 import RoleSeparatedRecurrentPredictor, UnifiedFeatures, build_deeplabv3plus_resnet50_host
from predify2021.model_factory.deeplabv3plus_resnet50.corrections import ErrorGainCorrection


ROLE_PREDICTOR_CHECKPOINT_DEFAULT = "/home/lin/predify/experiments/kitti_step_role_separated_predictor_3370f78/best_role_separated_predictor.pt"
WRITEBACK_CHECKPOINT_DEFAULT = "/home/lin/predify/experiments/kitti_step_host_conditioned_writeback/host_conditioned_writeback_epoch3.pt"
ALPHA = 0.207
BETA = 0.793


def load_components(static_checkpoint, adapter_checkpoint, predictor_checkpoint, writeback_checkpoint):
    model = build_deeplabv3plus_resnet50_host().cuda()
    load_static_kitti_checkpoint(model, Path(static_checkpoint))
    adapter_payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.load_state_dict(adapter_payload["adapter_state_dict"], strict=True)
    load_writeback_checkpoint(model, Path(writeback_checkpoint))
    predictor = RoleSeparatedRecurrentPredictor().cuda()
    predictor_payload = torch.load(predictor_checkpoint, map_location="cpu", weights_only=False)
    predictor.load_state_dict(predictor_payload["predictor_state_dict"], strict=True)
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    model.eval()
    predictor.eval()
    return model, predictor


def add_frame_noise(image, sigma=0.10):
    return torch.clamp(image + sigma * torch.randn_like(image), 0.0, 1.0)


def encode_image(model, image):
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        return model.encode_backbone_features(raw)


def zero_state(state):
    return UnifiedFeatures(*(torch.zeros_like(value) for value in state.as_tuple()))


def error_state(observation, prediction):
    return UnifiedFeatures(*(current - predicted for current, predicted in zip(observation.as_tuple(), prediction.as_tuple())))


def update_dynamic_error(error, previous):
    if previous is None:
        previous = zero_state(error)
    return UnifiedFeatures(*(ALPHA * current + BETA * old for current, old in zip(error.as_tuple(), previous.as_tuple())))


def detach_state(state):
    return UnifiedFeatures(*(value.detach() for value in state.as_tuple()))


def correction_posterior(semantic_prior, error, dynamic_error, corrections):
    gain_z1 = corrections[0](dynamic_error.z1)
    gain_z4 = corrections[1](dynamic_error.z4)
    return UnifiedFeatures(
        semantic_prior.z1 - gain_z1 * error.z1,
        semantic_prior.z2,
        semantic_prior.z3,
        semantic_prior.z4 - gain_z4 * error.z4,
    ), (gain_z1, gain_z4)


def correction_loss(posterior, clean_state):
    return F.mse_loss(posterior.z1, clean_state.z1) + F.mse_loss(posterior.z4, clean_state.z4)


def next_role_prediction(predictor, observation, error, hidden):
    return predictor.step(observation, error, *hidden)

