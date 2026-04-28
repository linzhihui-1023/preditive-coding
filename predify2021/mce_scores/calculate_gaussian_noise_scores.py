############################
# Evaluate PVGG under on-the-fly Gaussian noise.
############################

gpu_to_use = 0
batchsize = 32
MAX_TIME_STEPS = 10
MAX_IMAGES = 512
rseed = 17
MODEL_NAME = "pvgg"
ALPHA_MODES = ("with_alpha", "without_alpha")
SIGMAS = (0.5, 1.0, 1.5, 2.0)
FFM = 0.8
FBM = 0.1
ERM_VALUES = 0.01
HP_PRESET = "pvgg_table_s2"
LOAD_PCODER_WEIGHTS = True
PCODER_WEIGHTS = "/home/lin/predify/weights_pvgg16_imagenet"
############################


import os
import pickle
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder, ImageNet
from torchvision.transforms import transforms
from tqdm import tqdm

from ..model_factory import build_hyperparams, disable_alpha, get_model, get_num_pcoders


os.environ["TORCH_HOME"] = "/home/lin/predify/.torch"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batchsize = int(os.environ.get("PREDIFY_BATCHSIZE", str(batchsize)))
MAX_TIME_STEPS = int(os.environ.get("PREDIFY_TIMESTEPS", str(MAX_TIME_STEPS)))
MAX_IMAGES = int(os.environ.get("PREDIFY_MAX_IMAGES", str(MAX_IMAGES)))
num_workers = int(
    os.environ.get(
        "PREDIFY_NUM_WORKERS",
        str(16 if device.type == "cuda" else min(8, os.cpu_count() or 1)),
    )
)
imagenet_root = os.environ.get("PREDIFY_IMAGENET_ROOT", "").strip()
selected_sigmas = os.environ.get("PREDIFY_SIGMAS")
selected_alpha_modes = os.environ.get("PREDIFY_ALPHA_MODES")
selected_alpha_values = os.environ.get("PREDIFY_ALPHA_VALUES")
selected_hp_preset = os.environ.get("PREDIFY_HP_PRESET")
selected_zero_ff = os.environ.get("PREDIFY_ZERO_FF")
selected_zero_fb = os.environ.get("PREDIFY_ZERO_FB")

np.random.seed(rseed)
torch.manual_seed(rseed)

PVGG_TABLE_S2_HYPERPARAMS = [
    {"ffm": 0.2, "fbm": 0.05, "erm": 0.01},
    {"ffm": 0.4, "fbm": 0.10, "erm": 0.01},
    {"ffm": 0.4, "fbm": 0.10, "erm": 0.01},
    {"ffm": 0.5, "fbm": 0.10, "erm": 0.01},
    {"ffm": 0.6, "fbm": 0.00, "erm": 0.01},
]


def parse_csv_env(value, cast=float):
    if not value:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_bool_env(value):
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def override_alpha_values(hyperparams, alpha_value):
    return [{**hp, "erm": float(alpha_value)} for hp in hyperparams]


def disable_feedforward(hyperparams):
    return [{**hp, "ffm": 0.0} for hp in hyperparams]


def disable_feedback(hyperparams):
    return [{**hp, "fbm": 0.0} for hp in hyperparams]


def format_alpha_label(alpha_value):
    return f"alpha_{alpha_value:g}".replace("-", "neg").replace(".", "p")


def build_base_hyperparams(model_name):
    hp_preset = (selected_hp_preset or HP_PRESET).strip().lower()
    num_pcoders = get_num_pcoders(model_name)

    if hp_preset == "fixed":
        hyperparams = build_hyperparams(num_pcoders, FFM, FBM, ERM_VALUES)
    elif hp_preset == "pvgg_table_s2":
        if model_name != "pvgg":
            raise ValueError("The 'pvgg_table_s2' preset is only defined for MODEL_NAME='pvgg'.")
        hyperparams = [dict(hp) for hp in PVGG_TABLE_S2_HYPERPARAMS]
    else:
        raise ValueError(f"Unsupported PREDIFY_HP_PRESET='{hp_preset}'.")

    if parse_bool_env(selected_zero_ff):
        hyperparams = disable_feedforward(hyperparams)
    if parse_bool_env(selected_zero_fb):
        hyperparams = disable_feedback(hyperparams)

    return hyperparams


def build_alpha_mode_hyperparams(model_name, alpha_mode):
    hyperparams = build_base_hyperparams(model_name)

    if alpha_mode == "with_alpha":
        return hyperparams
    if alpha_mode == "without_alpha":
        return disable_alpha(hyperparams)

    if alpha_mode.startswith("alpha="):
        return override_alpha_values(hyperparams, float(alpha_mode.split("=", 1)[1]))

    if alpha_mode.startswith("alpha_"):
        return override_alpha_values(hyperparams, float(alpha_mode.split("_", 1)[1]))

    if alpha_mode.startswith("alpha"):
        return override_alpha_values(hyperparams, float(alpha_mode[len("alpha"):]))

    if alpha_mode != "with_alpha":
        raise ValueError("alpha_mode must be 'with_alpha' or 'without_alpha'.")
    return hyperparams


def resolve_experiments():
    suffix = ""
    if parse_bool_env(selected_zero_ff):
        suffix += "_noff"
    if parse_bool_env(selected_zero_fb):
        suffix += "_nofb"
    alpha_values = parse_csv_env(selected_alpha_values, float)
    if alpha_values is not None:
        return [
            {"name": f"{format_alpha_label(alpha_value)}{suffix}", "alpha_value": float(alpha_value)}
            for alpha_value in alpha_values
        ]

    alpha_modes = tuple(parse_csv_env(selected_alpha_modes, str) or ALPHA_MODES)
    return [{"name": f"{alpha_mode}{suffix}", "alpha_mode": alpha_mode} for alpha_mode in alpha_modes]


def build_experiment_hyperparams(model_name, experiment):
    if "alpha_value" in experiment:
        hyperparams = build_base_hyperparams(model_name)
        return override_alpha_values(hyperparams, experiment["alpha_value"])

    return build_alpha_mode_hyperparams(model_name, experiment["alpha_mode"])


def build_model(experiment):
    experiment_name = experiment["name"]
    hyperparams = build_experiment_hyperparams(MODEL_NAME, experiment)
    if LOAD_PCODER_WEIGHTS and PCODER_WEIGHTS is None:
        raise ValueError("Set PCODER_WEIGHTS before running, or disable LOAD_PCODER_WEIGHTS explicitly.")

    alpha_values = [hp["erm"] for hp in hyperparams]
    beta_values = [hp["ffm"] for hp in hyperparams]
    lambda_values = [hp["fbm"] for hp in hyperparams]
    print(
        f"Building model for {experiment_name} on device={device} with batchsize={batchsize}, "
        f"timesteps={MAX_TIME_STEPS}, num_workers={num_workers}, "
        f"beta_values={beta_values}, lambda_values={lambda_values}, alpha_values={alpha_values}",
        flush=True,
    )
    net = get_model(
        MODEL_NAME,
        pretrained=LOAD_PCODER_WEIGHTS,
        deep_graph=False,
        hyperparams=hyperparams,
        pcoder_weights=PCODER_WEIGHTS,
    )
    net.to(device)
    return net


def get_imagenet_val_dataset(root):
    if not root:
        raise ValueError(
            "Set PREDIFY_IMAGENET_ROOT to your clean ImageNet validation root. "
            "You can point it either to the ImageNet root directory or directly to the val directory."
        )

    transform_val = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    imagefolder_candidates = [root, os.path.join(root, "val"), os.path.join(root, "validation")]
    if os.path.isfile(os.path.join(root, "meta.bin")):
        dataset = ImageNet(root, split="val", transform=transform_val)
    else:
        dataset = None
        for candidate in imagefolder_candidates:
            if not os.path.isdir(candidate):
                continue
            try:
                maybe_dataset = ImageFolder(candidate, transform=transform_val)
            except Exception:
                continue
            if len(maybe_dataset) > 0:
                dataset = maybe_dataset
                break
        if dataset is None:
            raise FileNotFoundError(
                f"Could not find a clean ImageNet validation dataset under '{root}'. "
                "Expected either an ImageNet root with meta.bin or an ImageFolder-style val directory."
            )

    if MAX_IMAGES > 0 and MAX_IMAGES < len(dataset):
        rng = np.random.default_rng(rseed)
        indices = rng.permutation(len(dataset))[:MAX_IMAGES].tolist()
        dataset = Subset(dataset, indices)

    return dataset


def build_dataloader():
    dataset = get_imagenet_val_dataset(imagenet_root)
    print(f"Using {len(dataset)} validation images from {imagenet_root}", flush=True)
    return DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def eval_under_gaussian_noise(net, dataloader, timesteps, sigma, tqdm_desc=""):
    corrects = np.zeros((timesteps + 1, 1))
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(rseed)

    for images, labels in tqdm(dataloader, desc=tqdm_desc):
        net.reset()

        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        noisy_images = images + torch.randn(
            images.shape,
            generator=noise_generator,
            device=device,
            dtype=images.dtype,
        ) * sigma

        for tt in range(timesteps + 1):
            if tt == 0:
                with torch.no_grad():
                    outputs = net(noisy_images)
            else:
                with torch.no_grad():
                    outputs = net(None)

            _, preds = outputs.max(-1)
            corrects[tt, 0] += torch.sum(preds == labels).item()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    for tt in range(timesteps + 1):
        acc = 100.0 * corrects[tt, 0] / len(dataloader.dataset)
        print(f"Test set t = {tt:02d}: Accuracy: {acc:.4f}", flush=True)
    print(flush=True)
    return corrects


def evaluate_sigma_suite(net, experiment_name, dataloader, sigmas):
    accuracy_dict = {timestep: {sigma: 0.0 for sigma in sigmas} for timestep in range(MAX_TIME_STEPS + 1)}

    for sigma in sigmas:
        print(flush=True)
        print("-" * 30, flush=True)
        print(f"TESTING {MODEL_NAME} ({experiment_name}) ON : GAUSSIAN NOISE sigma={sigma}", flush=True)
        print("-" * 30, flush=True)

        tstart = datetime.now()
        print(f"STARTING AT : {tstart}", flush=True)
        accuracy = eval_under_gaussian_noise(
            net,
            dataloader,
            timesteps=MAX_TIME_STEPS,
            sigma=sigma,
            tqdm_desc=f"{experiment_name}_gaussian_{sigma:g}",
        )
        tend = datetime.now()
        print(f"TOTAL TIME TAKEN : {tend-tstart}", flush=True)

        for t in range(MAX_TIME_STEPS + 1):
            accuracy_dict[t][sigma] = accuracy[t, 0]

    return accuracy_dict


def main():
    experiments = resolve_experiments()
    hp_preset = (selected_hp_preset or HP_PRESET).strip().lower()
    sigmas = parse_csv_env(selected_sigmas, float) or list(SIGMAS)
    dataloader = build_dataloader()
    comparison_dict = {}

    print(
        f"Starting Gaussian-noise run with model={MODEL_NAME}, experiments={tuple(exp['name'] for exp in experiments)}, "
        f"sigmas={tuple(sigmas)}, batchsize={batchsize}, max_images={len(dataloader.dataset)}, "
        f"timesteps={MAX_TIME_STEPS}, device={device}, num_workers={num_workers}, hp_preset={hp_preset}, "
        f"zero_ff={parse_bool_env(selected_zero_ff)}, zero_fb={parse_bool_env(selected_zero_fb)}",
        flush=True,
    )

    for experiment in experiments:
        experiment_name = experiment["name"]
        net = build_model(experiment)
        accuracy_dict = evaluate_sigma_suite(net, experiment_name, dataloader, sigmas)
        comparison_dict[experiment_name] = accuracy_dict

        with open(f"gaussian_noise_{MODEL_NAME}_{experiment_name}.p", "wb") as f:
            pickle.dump(accuracy_dict, f)

    with open(f"gaussian_noise_{MODEL_NAME}_alpha_comparison.p", "wb") as f:
        pickle.dump(
            {
                "model_name": MODEL_NAME,
                "experiments": tuple(exp["name"] for exp in experiments),
                "alpha_values": tuple(
                    exp.get("alpha_value", None) for exp in experiments
                ),
                "hp_preset": hp_preset,
                "zero_ff": parse_bool_env(selected_zero_ff),
                "zero_fb": parse_bool_env(selected_zero_fb),
                "ffm": FFM,
                "fbm": FBM,
                "erm_values": ERM_VALUES,
                "sigmas": tuple(sigmas),
                "max_images": len(dataloader.dataset),
                "results": comparison_dict,
            },
            f,
        )

if __name__ == "__main__":
    main()
