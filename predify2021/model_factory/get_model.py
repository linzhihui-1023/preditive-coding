from collections.abc import Sequence
from glob import glob
from os.path import join as opj

import toml
import torch
import predify


MODEL_ALIASES = {
    "pvgg": "pvgg",
    "pvgg_tf": "pvgg_tf",
    "pvggtargetflow": "pvgg_tf",
    "peffb0": "peffb0",
    "pefficientnetb0": "peffb0",
    "deeplabv3plus_resnet50": "deeplabv3plus_resnet50",
    "deeplabv3plus-resnet50": "deeplabv3plus_resnet50",
}

MODEL_PCODER_COUNTS = {
    "pvgg": 5,
    "pvgg_tf": 5,
    "peffb0": 8,
    "deeplabv3plus_resnet50": 0,
}


def canonicalize_model_name(name):
    key = name.lower()
    if key not in MODEL_ALIASES:
        raise ValueError("The model name is not supported yet.")
    return MODEL_ALIASES[key]


def get_num_pcoders(name):
    return MODEL_PCODER_COUNTS[canonicalize_model_name(name)]


def build_uniform_hyperparams(num_pcoders, ffm, fbm, erm):
    return [{"ffm": ffm, "fbm": fbm, "erm": erm} for _ in range(num_pcoders)]


def build_hyperparams(num_pcoders, ffm, fbm, erm_values):
    if isinstance(erm_values, Sequence) and not isinstance(erm_values, (str, bytes)):
        if len(erm_values) != num_pcoders:
            raise ValueError(f"Expected {num_pcoders} erm values, but got {len(erm_values)}.")
        return [
            {"ffm": ffm, "fbm": fbm, "erm": float(erm_values[idx])}
            for idx in range(num_pcoders)
        ]

    return build_uniform_hyperparams(num_pcoders, ffm, fbm, float(erm_values))


def disable_alpha(hyperparams):
    return [{**hp, "erm": 0.0} for hp in hyperparams]


def _resolve_weight_paths(pcoder_weights, num_pcoders):
    if pcoder_weights is None:
        raise ValueError(
            "pretrained=True requires `pcoder_weights`. Provide a checkpoint directory or a list of checkpoint files."
        )

    if isinstance(pcoder_weights, Sequence) and not isinstance(pcoder_weights, (str, bytes)):
        if len(pcoder_weights) != num_pcoders:
            raise ValueError(f"Expected {num_pcoders} checkpoint paths, but got {len(pcoder_weights)}.")
        return list(pcoder_weights)

    matches = []
    for pcoder_idx in range(1, num_pcoders + 1):
        pattern = opj(str(pcoder_weights), f"*pc{pcoder_idx}*.pth")
        found = sorted(glob(pattern))
        if len(found) != 1:
            raise FileNotFoundError(
                f"Expected exactly one checkpoint matching '{pattern}', but found {len(found)}."
            )
        matches.append(found[0])
    return matches


def _extract_pmodule_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "pcoderweights" in checkpoint:
        state_dict = checkpoint["pcoderweights"]
        return {
            key[len("pmodule."):] if key.startswith("pmodule.") else key: value
            for key, value in state_dict.items()
        }
    return checkpoint


def _normalize_state_dict_for_module(module, state_dict):
    target_keys = list(module.state_dict().keys())
    state_keys = list(state_dict.keys())

    if state_keys == target_keys:
        return state_dict

    if all(key.startswith("0.") for key in state_keys):
        stripped_state_dict = {key[2:]: value for key, value in state_dict.items()}
        if list(stripped_state_dict.keys()) == target_keys:
            return stripped_state_dict

    if all("." not in key for key in state_keys):
        prefixed_state_dict = {f"0.{key}": value for key, value in state_dict.items()}
        if list(prefixed_state_dict.keys()) == target_keys:
            return prefixed_state_dict

    return state_dict


def _load_pcoder_weights(net, checkpoint_paths):
    for pcoder_idx, checkpoint_path in enumerate(checkpoint_paths, 1):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        pmodule = getattr(net, f"pcoder{pcoder_idx}").pmodule
        state_dict = _normalize_state_dict_for_module(pmodule, _extract_pmodule_state_dict(checkpoint))
        pmodule.load_state_dict(state_dict)


def _load_targetflow_feedback_weights(net, checkpoint_paths):
    if net.input_prediction_module is not None:
        checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")
        projector = net.input_prediction_module.projector
        state_dict = _normalize_state_dict_for_module(
            projector,
            _extract_pmodule_state_dict(checkpoint),
        )
        projector.load_state_dict(state_dict)

    # Stage-to-stage decoders correspond to legacy PCoders 2..5.
    for module_idx, checkpoint_path in enumerate(checkpoint_paths[1:], 0):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        module = net.feedback_modules[module_idx]
        projector = module.projector
        state_dict = _normalize_state_dict_for_module(projector, _extract_pmodule_state_dict(checkpoint))
        projector.load_state_dict(state_dict)


def set_hyperparams(net, hps):
    num = net.number_of_pcoders

    assert len(hps) == num

    for n in range(1, num + 1):
        setattr(net, f"ffm{n}", torch.tensor(hps[n - 1]["ffm"], dtype=torch.float64))
        setattr(net, f"fbm{n}", torch.tensor(hps[n - 1]["fbm"], dtype=torch.float64))
        setattr(net, f"erm{n}", torch.tensor(hps[n - 1]["erm"], dtype=torch.float64))


def get_model(
    name,
    pretrained=False,
    deep_graph=False,
    timesteps=4,
    hyperparams=None,
    pcoder_weights=None,
    target_flow_mode="recursive",
    compute_local_param_grads=False,
    temporal_target_mode="next_top",
    temporal_horizons=(1,),
    dynamic_error=True,
    error_state_mode=None,
    local_loss_error_source="instant",
    error_sample_time=0.1035,
    error_time_constant=0.5,
    error_gain=1.0,
    temporal_error_sample_time=1.0,
    temporal_error_time_constant=1.0,
    temporal_error_gain=1.0,
    task="real_frame_pc",
    pc_ff_multiplier=(0.2, 0.4, 0.4, 0.5, 0.6),
    pc_fb_multiplier=(0.05, 0.1, 0.1, 0.1, 0.0),
    pc_error_multiplier=(0.01, 0.01, 0.01, 0.01, 0.01),
    real_frame_transition_mode="predify",
    real_frame_recurrent_error_input="dynamic",
    future_feature_stage=5,
    future_feature_history_mode="none",
    future_feature_temporal_fusion_mode="none",
    future_feature_predictor_kernel_size=1,
    future_feature_prediction_form="current_residual",
    future_motion_radius=1,
    future_motion_patch_size=3,
    segmentation_num_classes=19,
    segmentation_pretrained_backbone=None,
    segmentation_freeze_backbone=False,
    segmentation_checkpoint_path=None,
):
    canonical_name = canonicalize_model_name(name)

    if canonical_name == "pvgg":
        import torchvision
        from .pvgg16_shared import DeepPVGG16SeparateHP, PVGG16SeparateHP

        backbone = torchvision.models.vgg16(pretrained=True)

        if deep_graph:
            pnet = DeepPVGG16SeparateHP(
                backbone=backbone,
                number_of_pcoders=5,
                number_of_timesteps=timesteps,
                build_graph=True,
                random_init=False,
                ff_multiplier=0.33,
                fb_multiplier=0.33,
                er_multiplier=0.01,
            )
        else:
            pnet = PVGG16SeparateHP(
                backbone=backbone,
                build_graph=False,
                random_init=False,
                ff_multiplier=0.33,
                fb_multiplier=0.33,
                er_multiplier=0.01,
            )

    elif canonical_name == "pvgg_tf":
        import torchvision
        from .pvgg16_targetflow import PVGG16TargetFlow

        weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.vgg16(weights=weights)
        pnet = PVGG16TargetFlow(
            backbone=backbone,
            target_flow_mode=target_flow_mode,
            compute_local_param_grads=compute_local_param_grads,
            temporal_target_mode=temporal_target_mode,
            temporal_horizons=temporal_horizons,
            dynamic_error=dynamic_error,
            error_state_mode=error_state_mode,
            local_loss_error_source=local_loss_error_source,
            error_sample_time=error_sample_time,
            error_time_constant=error_time_constant,
            error_gain=error_gain,
            temporal_error_sample_time=temporal_error_sample_time,
            temporal_error_time_constant=temporal_error_time_constant,
            temporal_error_gain=temporal_error_gain,
            task=task,
            pc_ff_multiplier=pc_ff_multiplier,
            pc_fb_multiplier=pc_fb_multiplier,
            pc_error_multiplier=pc_error_multiplier,
            real_frame_transition_mode=real_frame_transition_mode,
            real_frame_recurrent_error_input=real_frame_recurrent_error_input,
            future_feature_stage=future_feature_stage,
            future_feature_history_mode=future_feature_history_mode,
            future_feature_temporal_fusion_mode=(
                future_feature_temporal_fusion_mode
            ),
            future_feature_predictor_kernel_size=(
                future_feature_predictor_kernel_size
            ),
            future_feature_prediction_form=future_feature_prediction_form,
            future_motion_radius=future_motion_radius,
            future_motion_patch_size=future_motion_patch_size,
        )

    elif canonical_name == "peffb0":
        from timm.models import efficientnet_b0
        from .peffficientnetb0_shared import DeepPEfficientNetB0_SeparateHP, PEfficientNetB0_SeparateHP

        backbone = efficientnet_b0(pretrained=True)

        if deep_graph:
            pnet = DeepPEfficientNetB0_SeparateHP(
                backbone=backbone,
                number_of_pcoders=8,
                number_of_timesteps=timesteps,
                build_graph=True,
                random_init=False,
                ff_multiplier=0.33,
                fb_multiplier=0.33,
                er_multiplier=0.01,
            )
        else:
            pnet = PEfficientNetB0_SeparateHP(
                backbone=backbone,
                build_graph=False,
                random_init=False,
                ff_multiplier=0.33,
                fb_multiplier=0.33,
                er_multiplier=0.01,
            )

    elif canonical_name == "deeplabv3plus_resnet50":
        from .deeplabv3plus_resnet50 import build_deeplabv3plus_resnet50_host

        if segmentation_pretrained_backbone is None:
            segmentation_pretrained_backbone = pretrained
        pnet = build_deeplabv3plus_resnet50_host(
            num_classes=segmentation_num_classes,
            checkpoint_path=segmentation_checkpoint_path,
            load_cityscapes_checkpoint=segmentation_pretrained_backbone,
            freeze_backbone=segmentation_freeze_backbone,
        )

    else:
        raise ValueError("The model name is not supported yet.")

    if pretrained and canonical_name != "deeplabv3plus_resnet50":
        checkpoint_paths = _resolve_weight_paths(pcoder_weights, pnet.number_of_pcoders)
        print(f"Loading feedback weights from {checkpoint_paths}")
        if canonical_name == "pvgg_tf":
            _load_targetflow_feedback_weights(pnet, checkpoint_paths)
        else:
            _load_pcoder_weights(pnet, checkpoint_paths)

    if hyperparams is not None:
        if canonical_name == "deeplabv3plus_resnet50":
            raise ValueError("deeplabv3plus_resnet50 does not use PCoder hyperparameters.")
        if canonical_name == "pvgg_tf":
            raise ValueError("pvgg_tf does not use legacy PCoder hyperparameters.")
        set_hyperparams(pnet, hyperparams)

    return pnet.eval()
