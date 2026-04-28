############################
# Analyse the mCE scores for the predified networks
############################

gpu_to_use     = 0
batchsize      = 16
MAX_TIME_STEPS = 10
rseed          = 17
imagenetc_dir = '/home/lin/predify/ImageNet-C/'
MODEL_NAME = 'pvgg'
ALPHA_MODES = ('with_alpha', 'without_alpha')
FFM = 0.8
FBM = 0.1
# Use a float for a shared alpha across all PCoders, or a list/tuple for layer-wise values.
ERM_VALUES = 0.01
HP_PRESET = "pvgg_table_s2"
# Set this to False only if you intentionally want random feedback weights.
LOAD_PCODER_WEIGHTS = True
# Set this to the downloaded feedback-weight directory, or to a list of checkpoint files.
PCODER_WEIGHTS = "/home/lin/predify/weights_pvgg16_imagenet"
############################


import os
os.environ["TORCH_HOME"] = "/home/lin/predify/.torch"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

batchsize = int(os.environ.get("PREDIFY_BATCHSIZE", str(batchsize)))


import numpy as np
from datetime import datetime
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from ..model_factory import build_hyperparams, disable_alpha, get_model, get_num_pcoders

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_workers = int(os.environ.get("PREDIFY_NUM_WORKERS", str(16 if device.type == 'cuda' else min(8, os.cpu_count() or 1))))
np.random.seed(rseed)
torch.manual_seed(rseed)

selected_noises = os.environ.get("PREDIFY_NOISES")
selected_noise_levels = os.environ.get("PREDIFY_LEVELS")
selected_alpha_modes = os.environ.get("PREDIFY_ALPHA_MODES")
selected_hp_preset = os.environ.get("PREDIFY_HP_PRESET")
max_images = int(os.environ.get("PREDIFY_MAX_IMAGES", "0"))

PVGG_TABLE_S2_HYPERPARAMS = [
    {"ffm": 0.2, "fbm": 0.05, "erm": 0.01},
    {"ffm": 0.4, "fbm": 0.10, "erm": 0.01},
    {"ffm": 0.4, "fbm": 0.10, "erm": 0.01},
    {"ffm": 0.5, "fbm": 0.10, "erm": 0.01},
    {"ffm": 0.6, "fbm": 0.00, "erm": 0.01},
]


def parse_csv_env(value, cast=str):
    if not value:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def build_alpha_mode_hyperparams(model_name, alpha_mode):
    hp_preset = (selected_hp_preset or HP_PRESET).strip().lower()
    num_pcoders = get_num_pcoders(model_name)

    if hp_preset == 'fixed':
        hyperparams = build_hyperparams(num_pcoders, FFM, FBM, ERM_VALUES)
    elif hp_preset == 'pvgg_table_s2':
        if model_name != 'pvgg':
            raise ValueError("The 'pvgg_table_s2' preset is only defined for MODEL_NAME='pvgg'.")
        hyperparams = [dict(hp) for hp in PVGG_TABLE_S2_HYPERPARAMS]
    else:
        raise ValueError(f"Unsupported PREDIFY_HP_PRESET='{hp_preset}'.")

    if alpha_mode == 'without_alpha':
        return disable_alpha(hyperparams)
    if alpha_mode != 'with_alpha':
        raise ValueError("alpha_mode must be 'with_alpha' or 'without_alpha'.")
    return hyperparams


def build_model(alpha_mode):
    hyperparams = build_alpha_mode_hyperparams(MODEL_NAME, alpha_mode)
    if LOAD_PCODER_WEIGHTS and PCODER_WEIGHTS is None:
        raise ValueError("Set PCODER_WEIGHTS before running, or disable LOAD_PCODER_WEIGHTS explicitly.")
    print(f"Building model for {alpha_mode} on device={device} with batchsize={batchsize}, num_workers={num_workers}", flush=True)
    net = get_model(
        MODEL_NAME,
        pretrained=LOAD_PCODER_WEIGHTS,
        deep_graph=False,
        hyperparams=hyperparams,
        pcoder_weights=PCODER_WEIGHTS,
    )
    net.to(device)
    return net


def eval_training(net, dataloader, timesteps, tqdm_desc=''):
    corrects = np.zeros((timesteps + 1, 1))
    for (images, labels) in tqdm(dataloader, desc=tqdm_desc):
        net.reset()

        for tt in range(timesteps + 1):
            if tt == 0:
                with torch.no_grad():
                    outputs = net(images.to(device))
            else:
                with torch.no_grad():
                    outputs = net(None)

            _, preds = outputs.max(-1)
            corrects[tt, 0] += torch.sum(preds == labels.to(device)).item()

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    for tt in range(timesteps + 1):
        acc = 100.0 * corrects[tt, 0] / len(dataloader.dataset)
        print(f"Test set t = {tt:02d}: Accuracy: {acc:.4f}")
    print()
    return corrects


def get_imagenetc_dataloader(noise_level, noise_type):
    print(f"Preparing dataloader for noise={noise_type}, severity={noise_level}", flush=True)
    transform_val = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    distorted_dataset = ImageFolder(
        root=os.path.join(imagenetc_dir, noise_type, str(noise_level)),
        transform=transform_val,
    )
    if max_images > 0 and max_images < len(distorted_dataset):
        rng = np.random.default_rng(rseed)
        indices = rng.permutation(len(distorted_dataset))[:max_images].tolist()
        distorted_dataset = Subset(distorted_dataset, indices)
        print(f"Using subset of {len(distorted_dataset)} images for fast validation", flush=True)
    distorted_dataset_loader = torch.utils.data.DataLoader(
        distorted_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
    )

    return distorted_dataset_loader


def evaluate_noise_suite(net, alpha_mode):
    all_noises = [
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        "gaussian_blur",
        "gaussian_noise",
        "glass_blur",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "saturate",
        "shot_noise",
        "snow",
        "spatter",
        "speckle_noise",
        "zoom_blur",
    ]

    noise_levels = [1, 2, 3, 4, 5]
    noise_override = parse_csv_env(selected_noises, str)
    level_override = parse_csv_env(selected_noise_levels, int)
    if noise_override is not None:
        all_noises = noise_override
    if level_override is not None:
        noise_levels = level_override
    accuracy_dict = {
        timestep: {noise: [0.0 for _ in range(len(noise_levels))] for noise in all_noises}
        for timestep in range(MAX_TIME_STEPS + 1)
    }

    for noise in all_noises:
        for noise_level_idx, noise_level in enumerate(noise_levels):
            imagenetc_trainloader = get_imagenetc_dataloader(noise_level=noise_level, noise_type=noise)
            print(flush=True)
            print('-' * 30, flush=True)
            print(f"TESTING {MODEL_NAME} ({alpha_mode}) ON : NOISE {noise} SEVERITY {noise_level}", flush=True)
            print('-' * 30, flush=True)

            tstart = datetime.now()
            print(f"STARTING AT : {tstart}", flush=True)
            accuracy = eval_training(
                net,
                imagenetc_trainloader,
                timesteps=MAX_TIME_STEPS,
                tqdm_desc=f"{alpha_mode}_{noise}_{noise_level}",
            )
            tend = datetime.now()
            print(f"TOTAL TIME TAKEN : {tend-tstart}", flush=True)

            for t in range(MAX_TIME_STEPS + 1):
                accuracy_dict[t][noise][noise_level_idx] = accuracy[t, 0]

    return accuracy_dict


def main():
    alpha_modes = tuple(parse_csv_env(selected_alpha_modes, str) or ALPHA_MODES)
    hp_preset = (selected_hp_preset or HP_PRESET).strip().lower()
    comparison_dict = {}
    print(
        f"Starting mCE run with model={MODEL_NAME}, alpha_modes={alpha_modes}, batchsize={batchsize}, "
        f"timesteps={MAX_TIME_STEPS}, device={device}, num_workers={num_workers}, "
        f"max_images={max_images or 'full'}, hp_preset={hp_preset}",
        flush=True,
    )

    for alpha_mode in alpha_modes:
        net = build_model(alpha_mode)
        accuracy_dict = evaluate_noise_suite(net, alpha_mode)
        comparison_dict[alpha_mode] = accuracy_dict

        with open(f"mce_{MODEL_NAME}_{alpha_mode}.p", 'wb') as f:
            pickle.dump(accuracy_dict, f)

    with open(f"mce_{MODEL_NAME}_alpha_comparison.p", 'wb') as f:
        pickle.dump(
            {
                "model_name": MODEL_NAME,
                "alpha_modes": alpha_modes,
                "hp_preset": hp_preset,
                "ffm": FFM,
                "fbm": FBM,
                "erm_values": ERM_VALUES,
                "results": comparison_dict,
            },
            f,
        )


main()
