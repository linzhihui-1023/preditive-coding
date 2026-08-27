import json
import os
import random
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    add_frame_noise,
    correction_posterior,
    encode_image,
    error_state,
    load_components,
    load_image,
    next_role_prediction,
    observation_centered_posterior,
    residual_writeback_host_feature,
    update_dynamic_error,
    zero_state,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.corrections import ErrorGainCorrection


CORRECTION_CHECKPOINT_DEFAULT = "/home/lin/predify/experiments/kitti_step_observation_centered_prediction_error_correction/best_observation_centered_correction.pt"


def load_corrections(path):
    corrections = torch.nn.ModuleList([ErrorGainCorrection(), ErrorGainCorrection()]).cuda()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    corrections.load_state_dict(payload["correction_state_dict"], strict=True)
    corrections.requires_grad_(False)
    corrections.eval()
    return corrections, payload


def correction_delta(observation, posterior):
    return UnifiedFeatures(
        posterior.z1 - observation.z1,
        torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3),
        posterior.z4 - observation.z4,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Observation-centered correction evaluation requires CUDA")
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(os.environ.get("PREDIFY_OBSERVATION_CENTERED_EVAL_OUTPUT_DIR", "results/kitti_step_observation_centered_prediction_error_correction"))
    correction_checkpoint = os.environ.get("PREDIFY_OBSERVATION_CENTERED_CORRECTION_CHECKPOINT", CORRECTION_CHECKPOINT_DEFAULT)
    static_checkpoint = os.environ.get("PREDIFY_KITTI_STEP_STATIC_CHECKPOINT", STATIC_CHECKPOINT_DEFAULT)
    adapter_checkpoint = os.environ.get("PREDIFY_KITTI_STEP_ADAPTER_CHECKPOINT", ADAPTER_CHECKPOINT_DEFAULT)
    predictor_checkpoint = os.environ.get("PREDIFY_ROLE_SEPARATED_PREDICTOR_CHECKPOINT", ROLE_PREDICTOR_CHECKPOINT_DEFAULT)
    writeback_checkpoint = os.environ.get("PREDIFY_KITTI_STEP_WRITEBACK_CHECKPOINT", WRITEBACK_CHECKPOINT_DEFAULT)
    model, predictor = load_components(static_checkpoint, adapter_checkpoint, predictor_checkpoint, writeback_checkpoint)
    corrections, correction_payload = load_corrections(correction_checkpoint)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    names = ("clean_host", "noisy_host", "no_correction", "correction")
    confusion = {name: torch.zeros((19, 19), dtype=torch.int64) for name in names}
    error_sum = {"z1": 0.0, "z4": 0.0}
    dynamic_sum = {"z1": 0.0, "z4": 0.0}
    gain_sum = {"z1": 0.0, "z4": 0.0}
    correction_sum = {"z1": 0.0, "z4": 0.0}
    identity_max_difference = 0.0
    frame_count = 0
    finite = True
    with torch.inference_mode():
        for samples in groups.values():
            hidden = (None, None, None, None)
            dynamic_error = None
            pending_dynamics = None
            pending_semantic = None
            for frame_index, sample in enumerate(samples):
                clean_image = load_image(sample)
                noisy_image = add_frame_noise(clean_image)
                observation = encode_image(model, noisy_image)
                raw_features = model.extract_backbone_features(noisy_image)
                if frame_index == 0:
                    pending_dynamics, pending_semantic, *hidden = next_role_prediction(
                        predictor, observation, zero_state(observation), hidden
                    )
                    identity = residual_writeback_host_feature(model, raw_features, zero_state(observation), tuple(noisy_image.shape[-2:]))
                    noisy_logits = model.decode_from_host_feature(HostFeature(raw_features.c4, raw_features.c1, tuple(noisy_image.shape[-2:])))
                    identity_logits = model.decode_from_host_feature(identity)
                    identity_max_difference = max(identity_max_difference, (identity_logits - noisy_logits).abs().max().item())
                    continue
                error = error_state(observation, pending_dynamics)
                if frame_index == 1:
                    pending_dynamics, pending_semantic, *hidden = next_role_prediction(
                        predictor, observation, error, hidden
                    )
                    continue
                dynamic_error = update_dynamic_error(error, dynamic_error)
                posterior, gains = observation_centered_posterior(observation, error, dynamic_error, corrections)
                delta = correction_delta(observation, posterior)
                corrected_host = residual_writeback_host_feature(model, raw_features, delta, tuple(noisy_image.shape[-2:]))
                clean_logits = model(clean_image)
                noisy_logits = model.decode_from_host_feature(HostFeature(raw_features.c4, raw_features.c1, tuple(noisy_image.shape[-2:])))
                correction_logits = model.decode_from_host_feature(corrected_host)
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name, logits in (("clean_host", clean_logits), ("noisy_host", noisy_logits), ("no_correction", noisy_logits), ("correction", correction_logits)):
                    update_confusion_matrix(confusion[name], logits.argmax(1).squeeze(0).cpu().to(torch.int64), mask)
                error_sum["z1"] += error.z1.abs().mean().item()
                error_sum["z4"] += error.z4.abs().mean().item()
                dynamic_sum["z1"] += dynamic_error.z1.abs().mean().item()
                dynamic_sum["z4"] += dynamic_error.z4.abs().mean().item()
                gain_sum["z1"] += gains[0].mean().item()
                gain_sum["z4"] += gains[1].mean().item()
                correction_sum["z1"] += (gains[0] * error.z1).abs().mean().item()
                correction_sum["z4"] += (gains[1] * error.z4).abs().mean().item()
                finite = finite and all(torch.isfinite(value).all().item() for value in (error.z1, error.z4, dynamic_error.z1, dynamic_error.z4, posterior.z1, posterior.z4, gains[0], gains[1], delta.z1, delta.z4, correction_logits))
                frame_count += 1
                pending_dynamics, pending_semantic, *hidden = next_role_prediction(predictor, observation, error, hidden)
                dynamic_error = UnifiedFeatures(*(value.detach() for value in dynamic_error.as_tuple()))
    metrics = {name: float(torch.nanmean(compute_iou(value)).item()) for name, value in confusion.items()}
    diagnostics = {
        "mean_abs_prediction_error": {key: value / frame_count for key, value in error_sum.items()},
        "mean_abs_dynamic_error": {key: value / frame_count for key, value in dynamic_sum.items()},
        "mean_gain": {key: value / frame_count for key, value in gain_sum.items()},
        "mean_abs_correction_delta": {key: value / frame_count for key, value in correction_sum.items()},
    }
    identity_passed = identity_max_difference <= 1e-6 and metrics["noisy_host"] == metrics["no_correction"]
    delta_no_correction = metrics["correction"] - metrics["no_correction"]
    delta_noisy = metrics["correction"] - metrics["noisy_host"]
    summary = {
        "experiment": "kitti_step_observation_centered_prediction_error_correction",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "static_host_checkpoint": static_checkpoint,
        "adapter_checkpoint": adapter_checkpoint,
        "predictor_checkpoint": predictor_checkpoint,
        "writeback_checkpoint": writeback_checkpoint,
        "correction_checkpoint": correction_checkpoint,
        "correction_epoch": correction_payload.get("epoch"),
        "trainable_parameter_count": 2 * sum(parameter.numel() for parameter in ErrorGainCorrection().parameters()),
        "config": {"seed": 0, "gaussian_noise_sigma": 0.10, "alpha": 0.207, "beta": 0.793, "corrected_layers": [1, 4], "labels_used_for_training": False},
        "dataset": {"split": "val", "sequence_count": len(groups), "total_frame_count": len(dataset.samples), "effective_frame_count": frame_count},
        "identity_gate": {"max_abs_logit_difference": identity_max_difference, "noisy_host_equals_no_correction": metrics["noisy_host"] == metrics["no_correction"], "passed": identity_passed},
        "metrics": {"mIoU_clean_host": metrics["clean_host"], "mIoU_noisy_host": metrics["noisy_host"], "mIoU_no_correction": metrics["no_correction"], "mIoU_correction": metrics["correction"], "delta_correction_minus_no_correction": delta_no_correction, "delta_correction_minus_noisy_host": delta_noisy},
        "validation_state_mse": correction_payload.get("val_correction_state_mse"),
        "diagnostics": diagnostics,
        "finite": finite,
        "decision": "CORRECTION_BASELINE_IDENTITY: NO-GO" if not identity_passed else ("OBSERVATION_CENTERED_PREDICTION_ERROR_CORRECTION: GO" if metrics["correction"] > metrics["noisy_host"] else "OBSERVATION_CENTERED_PREDICTION_ERROR_CORRECTION: NO-GO"),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
