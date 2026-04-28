from collections.abc import Sequence
from glob import glob
from os.path import join as opj

import toml
import torch
import predify


MODEL_ALIASES = {
    "pvgg": "pvgg",
    "peffb0": "peffb0",
    "pefficientnetb0": "peffb0",
}

MODEL_PCODER_COUNTS = {
    "pvgg": 5,
    "peffb0": 8,
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


def set_hyperparams(net, hps):
    num = net.number_of_pcoders

    assert len(hps) == num

    for n in range(1, num + 1):
        setattr(net, f"ffm{n}", torch.tensor(hps[n - 1]["ffm"], dtype=torch.float64))
        setattr(net, f"fbm{n}", torch.tensor(hps[n - 1]["fbm"], dtype=torch.float64))
        setattr(net, f"erm{n}", torch.tensor(hps[n - 1]["erm"], dtype=torch.float64))


def get_model(name, pretrained=False, deep_graph=False, timesteps=4, hyperparams=None, pcoder_weights=None):
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

    else:
        raise ValueError("The model name is not supported yet.")

    if pretrained:
        checkpoint_paths = _resolve_weight_paths(pcoder_weights, pnet.number_of_pcoders)
        print(f"Loading feedback weights from {checkpoint_paths}")
        _load_pcoder_weights(pnet, checkpoint_paths)

    if hyperparams is not None:
        set_hyperparams(pnet, hyperparams)

    return pnet.eval()
