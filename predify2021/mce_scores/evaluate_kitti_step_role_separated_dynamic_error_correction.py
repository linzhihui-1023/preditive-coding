import json
import os
import random
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.evaluate_kitti_step_error_correction import corrected_host_feature
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    add_frame_noise,
    correction_posterior,
    detach_state,
    encode_image,
    error_state,
    load_components,
    load_image,
    next_role_prediction,
    update_dynamic_error,
    zero_state,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature
from predify2021.model_factory.deeplabv3plus_resnet50.corrections import ErrorGainCorrection
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups


CORRECTION_CHECKPOINT_DEFAULT = "/home/lin/predify/experiments/kitti_step_role_separated_dynamic_error_correction/best_role_separated_dynamic_correction.pt"


def load_corrections(path):
    corrections = torch.nn.ModuleList([ErrorGainCorrection(), ErrorGainCorrection()]).cuda()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    corrections.load_state_dict(payload["correction_state_dict"], strict=True)
    corrections.requires_grad_(False)
    corrections.eval()
    return corrections, payload


def host_for_state(model, raw_features, observation, state, output_size):
    return corrected_host_feature(model, raw_features, observation, state, output_size)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Role-separated dynamic correction evaluation requires CUDA")
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(os.environ.get("PREDIFY_ROLE_SEPARATED_DYNAMIC_EVAL_OUTPUT_DIR", "results/kitti_step_role_separated_dynamic_error_correction"))
    correction_checkpoint = os.environ.get("PREDIFY_ROLE_SEPARATED_DYNAMIC_CORRECTION_CHECKPOINT", CORRECTION_CHECKPOINT_DEFAULT)
    static_checkpoint = os.environ.get("PREDIFY_KITTI_STEP_STATIC_CHECKPOINT", STATIC_CHECKPOINT_DEFAULT)
    adapter_checkpoint = os.environ.get("PREDIFY_KITTI_STEP_ADAPTER_CHECKPOINT", ADAPTER_CHECKPOINT_DEFAULT)
    predictor_checkpoint = os.environ.get("PREDIFY_ROLE_SEPARATED_PREDICTOR_CHECKPOINT", ROLE_PREDICTOR_CHECKPOINT_DEFAULT)
    writeback_checkpoint = os.environ.get("PREDIFY_KITTI_STEP_WRITEBACK_CHECKPOINT", WRITEBACK_CHECKPOINT_DEFAULT)
    model, predictor = load_components(static_checkpoint, adapter_checkpoint, predictor_checkpoint, writeback_checkpoint)
    corrections, correction_payload = load_corrections(correction_checkpoint)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    confusion = {name: torch.zeros((19, 19), dtype=torch.int64) for name in ("clean_host", "noisy_host", "no_correction", "correction")}
    error_sum = {"z1": 0.0, "z4": 0.0}
    dynamic_sum = {"z1": 0.0, "z4": 0.0}
    gain_sum = {"z1": 0.0, "z4": 0.0}
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
                clean_state = encode_image(model, clean_image)
                observation = encode_image(model, noisy_image)
                raw_features = model.extract_backbone_features(noisy_image)
                if frame_index == 0:
                    pending_dynamics, pending_semantic, *hidden = next_role_prediction(
                        predictor, observation, zero_state(observation), hidden
                    )
                    continue
                error = error_state(observation, pending_dynamics)
                if frame_index == 1:
                    pending_dynamics, pending_semantic, *hidden = next_role_prediction(
                        predictor, observation, error, hidden
                    )
                    continue
                dynamic_error = update_dynamic_error(error, dynamic_error)
                no_correction = pending_semantic
                correction, gains = correction_posterior(
                    pending_semantic, error, dynamic_error, corrections
                )
                output_size = tuple(clean_image.shape[-2:])
                clean_logits = model(clean_image)
                noisy_logits = model.decode_from_host_feature(HostFeature(raw_features.c4, raw_features.c1, output_size))
                no_correction_logits = model.decode_from_host_feature(host_for_state(model, raw_features, observation, no_correction, output_size))
                correction_logits = model.decode_from_host_feature(host_for_state(model, raw_features, observation, correction, output_size))
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name, logits in (("clean_host", clean_logits), ("noisy_host", noisy_logits), ("no_correction", no_correction_logits), ("correction", correction_logits)):
                    update_confusion_matrix(confusion[name], logits.argmax(1).squeeze(0).cpu().to(torch.int64), mask)
                error_sum["z1"] += error.z1.abs().mean().item()
                error_sum["z4"] += error.z4.abs().mean().item()
                dynamic_sum["z1"] += dynamic_error.z1.abs().mean().item()
                dynamic_sum["z4"] += dynamic_error.z4.abs().mean().item()
                gain_sum["z1"] += gains[0].mean().item()
                gain_sum["z4"] += gains[1].mean().item()
                finite = finite and all(torch.isfinite(value).all().item() for value in (error.z1, error.z4, dynamic_error.z1, dynamic_error.z4, correction.z1, correction.z4, gains[0], gains[1]))
                frame_count += 1
                pending_dynamics, pending_semantic, *hidden = next_role_prediction(
                    predictor, observation, error, hidden
                )
                dynamic_error = detach_state(dynamic_error)
    metrics = {name: float(torch.nanmean(compute_iou(value)).item()) for name, value in confusion.items()}
    diagnostics = {
        "mean_abs_prediction_error": {key: value / frame_count for key, value in error_sum.items()},
        "mean_abs_dynamic_error": {key: value / frame_count for key, value in dynamic_sum.items()},
        "mean_gain": {key: value / frame_count for key, value in gain_sum.items()},
    }
    delta = metrics["correction"] - metrics["no_correction"]
    summary = {
        "experiment": "kitti_step_role_separated_dynamic_error_correction",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "static_host_checkpoint": static_checkpoint,
        "adapter_checkpoint": adapter_checkpoint,
        "predictor_checkpoint": predictor_checkpoint,
        "writeback_checkpoint": writeback_checkpoint,
        "correction_checkpoint": correction_checkpoint,
        "correction_epoch": correction_payload.get("epoch"),
        "config": {"seed": 0, "gaussian_noise_sigma": 0.10, "alpha": 0.207, "beta": 0.793, "corrected_layers": [1, 4], "labels_used_for_training": False},
        "dataset": {"split": "val", "sequence_count": len(groups), "total_frame_count": len(dataset.samples), "effective_frame_count": frame_count},
        "metrics": {"mIoU_clean_host": metrics["clean_host"], "mIoU_noisy_host": metrics["noisy_host"], "mIoU_no_correction": metrics["no_correction"], "mIoU_correction": metrics["correction"], "delta_correction_minus_no_correction": delta, "delta_correction_minus_noisy_host": metrics["correction"] - metrics["noisy_host"]},
        "validation_state_mse": correction_payload.get("val_correction_state_mse"),
        "diagnostics": diagnostics,
        "trainable_parameter_count": 2 * sum(parameter.numel() for parameter in ErrorGainCorrection().parameters()),
        "finite": finite,
        "decision": "PREDICTION_ERROR_DRIVEN_CORRECTION: GO" if metrics["correction"] > metrics["no_correction"] else "PREDICTION_ERROR_DRIVEN_CORRECTION: NO-GO",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
