import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.role_separated_direct_state_correction import (
    add_frame_noise,
    direct_posterior,
    error_state,
    load_direct_corrections,
    load_image,
    load_role_components,
    make_paths,
    next_role_prediction,
    update_dynamic_error,
    zero_state,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    DYNAMIC_ERROR_GAIN,
    DYNAMIC_ERROR_SAMPLE_TIME,
    DYNAMIC_ERROR_TIME_CONSTANT,
    detach_state,
    residual_writeback_host_feature,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature


SEED = 0
SIGMA = 0.10
EXPECTED_SEQUENCES = 9
EXPECTED_TOTAL_FRAMES = 2981
EXPECTED_EFFECTIVE_FRAMES = 2963
CURRENT_GAIN_MIOU = 0.2973953001
HISTORICAL_DIRECT_MIOU = 0.3347162547


def load_corrections(path):
    return load_direct_corrections(path)


def corrected_host(model, raw_features, observation, posterior, output_size):
    delta = type(posterior)(
        posterior.z1 - observation.z1,
        torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3),
        posterior.z4 - observation.z4,
    )
    return residual_writeback_host_feature(model, raw_features, delta, output_size)


def add_metrics(confusion):
    return {
        name: float(torch.nanmean(compute_iou(value)).item())
        for name, value in confusion.items()
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Role-separated direct correction evaluation requires CUDA")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(os.environ.get("PREDIFY_ROLE_SEPARATED_DIRECT_EVAL_OUTPUT_DIR", "results/kitti_step_role_separated_direct_state_correction"))
    paths = make_paths()
    correction_paths = {
        "experiment_a": Path(os.environ["PREDIFY_ROLE_SEPARATED_DIRECT_STATE_CHECKPOINT"]),
        "experiment_b": Path(os.environ["PREDIFY_ROLE_SEPARATED_SEMANTIC_STATE_CHECKPOINT"]),
    }
    model, predictor = load_role_components(paths["static"], paths["adapter"], paths["predictor"], paths["writeback"])
    corrections = {name: load_corrections(path)[0] for name, path in correction_paths.items()}
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    if len(groups) != EXPECTED_SEQUENCES or len(dataset.samples) != EXPECTED_TOTAL_FRAMES:
        raise RuntimeError("KITTI-STEP validation protocol mismatch")
    names = ("clean_host", "noisy_host", "no_correction", "experiment_a", "experiment_b")
    confusion = {name: torch.zeros((19, 19), dtype=torch.int64) for name in names}
    state_mse = {name: {"z1": 0.0, "z4": 0.0} for name in ("experiment_a", "experiment_b")}
    diagnostics = {
        "mean_abs_prediction_error": {"z1": 0.0, "z4": 0.0},
        "mean_abs_dynamic_error": {"z1": 0.0, "z4": 0.0},
        "mean_abs_delta_z": {
            "experiment_a": {"z1": 0.0, "z4": 0.0},
            "experiment_b": {"z1": 0.0, "z4": 0.0},
        },
    }
    frame_count = 0
    finite = True
    identity_max = 0.0
    with torch.inference_mode():
        for samples in groups.values():
            hidden = (None, None, None, None)
            pending_dynamics = None
            dynamic_error = None
            for frame_index, sample in enumerate(samples):
                clean_image = load_image(sample)
                noisy_image = add_frame_noise(clean_image, SIGMA)
                clean_raw = model.extract_backbone_features(clean_image)
                noisy_raw = model.extract_backbone_features(noisy_image)
                clean_state = model.encode_backbone_features(clean_raw)
                observation = model.encode_backbone_features(noisy_raw)
                output_size = tuple(clean_image.shape[-2:])
                if frame_index == 0:
                    pending_dynamics, _, *hidden = next_role_prediction(
                        predictor, observation, zero_state(observation), hidden
                    )
                    zero_delta = zero_state(observation)
                    identity_host = residual_writeback_host_feature(model, noisy_raw, zero_delta, output_size)
                    identity_max = max(identity_max, float((model.decode_from_host_feature(identity_host) - model.decode_from_host_feature(HostFeature(noisy_raw.c4, noisy_raw.c1, output_size))).abs().max().item()))
                    continue
                error = error_state(observation, pending_dynamics)
                if frame_index == 1:
                    pending_dynamics, _, *hidden = next_role_prediction(
                        predictor, observation, error, hidden
                    )
                    continue
                dynamic_error = update_dynamic_error(error, dynamic_error)
                posteriors = {
                    name: direct_posterior(observation, error, dynamic_error, correction)[0]
                    for name, correction in corrections.items()
                }
                hosts = {
                    "clean_host": HostFeature(clean_raw.c4, clean_raw.c1, output_size),
                    "noisy_host": HostFeature(noisy_raw.c4, noisy_raw.c1, output_size),
                    "no_correction": HostFeature(noisy_raw.c4, noisy_raw.c1, output_size),
                    "experiment_a": corrected_host(model, noisy_raw, observation, posteriors["experiment_a"], output_size),
                    "experiment_b": corrected_host(model, noisy_raw, observation, posteriors["experiment_b"], output_size),
                }
                logits = {name: model.decode_from_host_feature(host) for name, host in hosts.items()}
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name in names:
                    prediction = logits[name].argmax(1).squeeze(0).cpu().to(torch.int64)
                    update_confusion_matrix(confusion[name], prediction, mask)
                for name, posterior in posteriors.items():
                    state_mse[name]["z1"] += F.mse_loss(posterior.z1, clean_state.z1).item()
                    state_mse[name]["z4"] += F.mse_loss(posterior.z4, clean_state.z4).item()
                    diagnostics["mean_abs_delta_z"][name]["z1"] += (posterior.z1 - observation.z1).abs().mean().item()
                    diagnostics["mean_abs_delta_z"][name]["z4"] += (posterior.z4 - observation.z4).abs().mean().item()
                diagnostics["mean_abs_prediction_error"]["z1"] += error.z1.abs().mean().item()
                diagnostics["mean_abs_prediction_error"]["z4"] += error.z4.abs().mean().item()
                diagnostics["mean_abs_dynamic_error"]["z1"] += dynamic_error.z1.abs().mean().item()
                diagnostics["mean_abs_dynamic_error"]["z4"] += dynamic_error.z4.abs().mean().item()
                finite = finite and all(torch.isfinite(value).all().item() for value in (error.z1, error.z4, dynamic_error.z1, dynamic_error.z4, logits["experiment_a"], logits["experiment_b"], posteriors["experiment_a"].z1, posteriors["experiment_b"].z4))
                frame_count += 1
                pending_dynamics, _, *hidden = next_role_prediction(predictor, observation, error, hidden)
                dynamic_error = detach_state(dynamic_error)
    metrics = add_metrics(confusion)
    for name in state_mse:
        state_mse[name]["z1"] /= frame_count
        state_mse[name]["z4"] /= frame_count
        state_mse[name]["mean"] = (state_mse[name]["z1"] + state_mse[name]["z4"]) / 2
    for name in ("mean_abs_prediction_error", "mean_abs_dynamic_error"):
        for layer in diagnostics[name]:
            diagnostics[name][layer] /= frame_count
    for name in diagnostics["mean_abs_delta_z"]:
        for layer in diagnostics["mean_abs_delta_z"][name]:
            diagnostics["mean_abs_delta_z"][name][layer] /= frame_count
    identity_pass = identity_max <= 1e-6 and metrics["noisy_host"] == metrics["no_correction"]
    summary = {
        "experiment": "kitti_step_role_separated_direct_state_correction_evaluation",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoints": {**{key: str(value) for key, value in paths.items()}, **{key: str(value) for key, value in correction_paths.items()}},
        "config": {"seed": SEED, "gaussian_noise_sigma": SIGMA, "dynamic_error_sample_time": DYNAMIC_ERROR_SAMPLE_TIME, "dynamic_error_time_constant": DYNAMIC_ERROR_TIME_CONSTANT, "dynamic_error_gain": DYNAMIC_ERROR_GAIN, "corrected_layers": [1, 4], "correction_feedback_to_predictor": False},
        "dataset": {"split": "val", "sequence_count": len(groups), "total_frame_count": len(dataset.samples), "effective_frame_count": frame_count},
        "metrics": {"mIoU": metrics, "references": {"current_gain_correction": CURRENT_GAIN_MIOU, "historical_direct_open_loop": HISTORICAL_DIRECT_MIOU}},
        "state_mse": state_mse,
        "diagnostics": diagnostics,
        "identity_gate": {"max_abs_logit_difference": identity_max, "noisy_host_equals_no_correction": metrics["noisy_host"] == metrics["no_correction"], "passed": identity_pass},
        "finite": finite,
        "trainable_parameter_count": 0,
        "decision": "ROLE_SEPARATED_DIRECT_STATE_CORRECTION: GO" if identity_pass and metrics["experiment_a"] > metrics["noisy_host"] else "ROLE_SEPARATED_DIRECT_STATE_CORRECTION: NO-GO",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
