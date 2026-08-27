import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.kitti_step_persistent_blur import BLUR_KERNEL_SIZE, BLUR_SIGMA_LEVELS, BLUR_SIGMA_MAX, persistent_gaussian_blur
from predify2021.mce_scores.role_separated_direct_state_correction import (
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
from predify2021.mce_scores.role_separated_dynamic_error_correction import residual_writeback_host_feature
from predify2021.mce_scores.train_kitti_step_error_decomposition_correction import build_corrections, corruption_dynamic_error
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures


SEED = 0
EXPECTED_SEQUENCES = 9
EXPECTED_TOTAL_FRAMES = 2981
EXPECTED_EFFECTIVE_FRAMES = 2963


def load_decomposition(path):
    corrections = build_corrections()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    corrections.load_state_dict(payload["corrections"], strict=True)
    corrections.requires_grad_(False)
    corrections.eval()
    return corrections, payload


def corrected_host(model, raw_features, observation, posterior, output_size):
    delta = UnifiedFeatures(
        posterior.z1 - observation.z1,
        torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3),
        posterior.z4 - observation.z4,
    )
    return residual_writeback_host_feature(model, raw_features, delta, output_size)


def miou(confusion):
    return float(torch.nanmean(compute_iou(confusion)).item())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Persistent blur error decomposition evaluation requires CUDA")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(os.environ.get("PREDIFY_PERSISTENT_BLUR_EVAL_OUTPUT_DIR", "results/kitti_step_persistent_blur_error_decomposition"))
    paths = make_paths()
    corrected_b_path = Path(os.environ["PREDIFY_PERSISTENT_BLUR_CORRECTED_B_CHECKPOINT"])
    decomposition_path = Path(os.environ["PREDIFY_PERSISTENT_BLUR_DECOMPOSITION_CHECKPOINT"])
    model, predictor = load_role_components(paths["static"], paths["adapter"], paths["predictor"], paths["writeback"])
    corrected_b, corrected_b_payload = load_direct_corrections(corrected_b_path)
    decomposition, decomposition_payload = load_decomposition(decomposition_path)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    if len(groups) != EXPECTED_SEQUENCES or len(dataset.samples) != EXPECTED_TOTAL_FRAMES:
        raise RuntimeError("KITTI-STEP validation protocol mismatch")
    names = ("clean_host", "corrupted_host", "corrected_b", "error_decomposition_reliability")
    confusion = {name: torch.zeros((19, 19), dtype=torch.int64) for name in names}
    totals = {key: 0.0 for key in (
        "state_mse_z1", "state_mse_z4", "corruption_mse_z1", "corruption_mse_z4",
        "corruption_energy_z1", "corruption_energy_z4",
        "reliability_l1_z1", "reliability_l1_z4", "mean_reliability_z1", "mean_reliability_z4",
        "mean_abs_true_corruption_z1", "mean_abs_true_corruption_z4",
        "mean_abs_estimated_corruption_z1", "mean_abs_estimated_corruption_z4",
        "mean_abs_true_predictor_mismatch_z1", "mean_abs_true_predictor_mismatch_z4",
        "mean_abs_estimated_predictor_mismatch_z1", "mean_abs_estimated_predictor_mismatch_z4",
        "mean_abs_dynamic_corruption_z1", "mean_abs_dynamic_corruption_z4",
        "mean_abs_delta_z1", "mean_abs_delta_z4",
    )}
    identity_max = 0.0
    frame_count = 0
    finite = True
    with torch.inference_mode():
        for samples in groups.values():
            hidden = (None, None, None, None)
            pending_dynamics = None
            old_dynamic = None
            new_dynamic = None
            for frame_index, sample in enumerate(samples):
                clean_image = load_image(sample)
                corrupted_image = persistent_gaussian_blur(clean_image, frame_index, len(samples))
                clean_raw = model.extract_backbone_features(clean_image)
                corrupted_raw = model.extract_backbone_features(corrupted_image)
                clean_state = model.encode_backbone_features(clean_raw)
                observation = model.encode_backbone_features(corrupted_raw)
                output_size = tuple(clean_image.shape[-2:])
                if frame_index == 0:
                    pending_dynamics, _, *hidden = next_role_prediction(predictor, observation, zero_state(observation), hidden)
                    zero_host = residual_writeback_host_feature(model, corrupted_raw, zero_state(observation), output_size)
                    raw_host = HostFeature(corrupted_raw.c4, corrupted_raw.c1, output_size)
                    identity_max = max(identity_max, float((model.decode_from_host_feature(zero_host) - model.decode_from_host_feature(raw_host)).abs().max().item()))
                    continue
                error = error_state(observation, pending_dynamics)
                if frame_index == 1:
                    pending_dynamics, _, *hidden = next_role_prediction(predictor, observation, error, hidden)
                    continue
                old_dynamic = update_dynamic_error(error, old_dynamic)
                old_posterior, _ = direct_posterior(observation, error, old_dynamic, corrected_b)
                estimated_corruption = UnifiedFeatures(
                    decomposition[0].interpreter(observation.z1, error.z1),
                    torch.zeros_like(error.z2),
                    torch.zeros_like(error.z3),
                    decomposition[1].interpreter(observation.z4, error.z4),
                )
                new_dynamic = corruption_dynamic_error(
                    estimated_corruption, new_dynamic, decomposition_payload.get("disable_dynamic", False)
                )
                estimated_corruption_1, estimated_mismatch_1, reliability_1, delta_1 = decomposition[0](observation.z1, error.z1, new_dynamic.z1)
                estimated_corruption_4, estimated_mismatch_4, reliability_4, delta_4 = decomposition[1](observation.z4, error.z4, new_dynamic.z4)
                posterior = UnifiedFeatures(observation.z1 + delta_1, observation.z2, observation.z3, observation.z4 + delta_4)
                hosts = {
                    "clean_host": HostFeature(clean_raw.c4, clean_raw.c1, output_size),
                    "corrupted_host": HostFeature(corrupted_raw.c4, corrupted_raw.c1, output_size),
                    "corrected_b": corrected_host(model, corrupted_raw, observation, old_posterior, output_size),
                    "error_decomposition_reliability": corrected_host(model, corrupted_raw, observation, posterior, output_size),
                }
                logits = {name: model.decode_from_host_feature(host) for name, host in hosts.items()}
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name in names:
                    update_confusion_matrix(confusion[name], logits[name].argmax(1).squeeze(0).cpu().to(torch.int64), mask)
                c1 = observation.z1 - clean_state.z1
                c4 = observation.z4 - clean_state.z4
                p1 = clean_state.z1 - (observation.z1 - error.z1)
                p4 = clean_state.z4 - (observation.z4 - error.z4)
                target_rel_1 = c1.abs() / (c1.abs() + p1.abs() + 1e-12)
                target_rel_4 = c4.abs() / (c4.abs() + p4.abs() + 1e-12)
                values = {
                    "state_mse_z1": F.mse_loss(posterior.z1, clean_state.z1),
                    "state_mse_z4": F.mse_loss(posterior.z4, clean_state.z4),
                    "corruption_mse_z1": F.mse_loss(estimated_corruption_1, c1),
                    "corruption_mse_z4": F.mse_loss(estimated_corruption_4, c4),
                    "corruption_energy_z1": c1.square().mean(),
                    "corruption_energy_z4": c4.square().mean(),
                    "reliability_l1_z1": F.l1_loss(reliability_1, target_rel_1),
                    "reliability_l1_z4": F.l1_loss(reliability_4, target_rel_4),
                    "mean_reliability_z1": reliability_1.mean(), "mean_reliability_z4": reliability_4.mean(),
                    "mean_abs_true_corruption_z1": c1.abs().mean(), "mean_abs_true_corruption_z4": c4.abs().mean(),
                    "mean_abs_estimated_corruption_z1": estimated_corruption_1.abs().mean(), "mean_abs_estimated_corruption_z4": estimated_corruption_4.abs().mean(),
                    "mean_abs_true_predictor_mismatch_z1": p1.abs().mean(), "mean_abs_true_predictor_mismatch_z4": p4.abs().mean(),
                    "mean_abs_estimated_predictor_mismatch_z1": estimated_mismatch_1.abs().mean(), "mean_abs_estimated_predictor_mismatch_z4": estimated_mismatch_4.abs().mean(),
                    "mean_abs_dynamic_corruption_z1": new_dynamic.z1.abs().mean(), "mean_abs_dynamic_corruption_z4": new_dynamic.z4.abs().mean(),
                    "mean_abs_delta_z1": delta_1.abs().mean(), "mean_abs_delta_z4": delta_4.abs().mean(),
                }
                for key, value in values.items():
                    totals[key] += value.item()
                finite = finite and all(torch.isfinite(value).all().item() for value in (*logits.values(), posterior.z1, posterior.z4, new_dynamic.z1, new_dynamic.z4))
                frame_count += 1
                pending_dynamics, _, *hidden = next_role_prediction(predictor, observation, error, hidden)
                old_dynamic = type(old_dynamic)(*(value.detach() for value in old_dynamic.as_tuple()))
                new_dynamic = type(new_dynamic)(*(value.detach() for value in new_dynamic.as_tuple()))
    metrics = {name: miou(value) for name, value in confusion.items()}
    for key in totals:
        totals[key] /= frame_count
    for layer in ("z1", "z4"):
        totals[f"corruption_nmse_{layer}"] = totals[f"corruption_mse_{layer}"] / (totals[f"corruption_energy_{layer}"] + 1e-12)
        del totals[f"corruption_mse_{layer}"], totals[f"corruption_energy_{layer}"]
    totals["state_mse_mean"] = (totals["state_mse_z1"] + totals["state_mse_z4"]) / 2
    totals["reliability_l1_mean"] = (totals["reliability_l1_z1"] + totals["reliability_l1_z4"]) / 2
    result = {
        "experiment": "kitti_step_persistent_blur_error_decomposition_reliability",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoints": {**{key: str(value) for key, value in paths.items()}, "corrected_b": str(corrected_b_path), "error_decomposition": str(decomposition_path)},
        "config": {"seed": SEED, "blur_kernel_size": BLUR_KERNEL_SIZE, "blur_sigma_max": BLUR_SIGMA_MAX, "blur_sigma_levels": BLUR_SIGMA_LEVELS, "alpha": 0.207, "beta": 0.793, "correction_feedback_to_predictor": False},
        "dataset": {"split": "val", "sequence_count": len(groups), "total_frame_count": len(dataset.samples), "effective_frame_count": frame_count},
        "metrics": {"mIoU": metrics, "corrected_b_reference_mIoU": 0.3505486298, "new_minus_corrected_b": metrics["error_decomposition_reliability"] - metrics["corrected_b"], "new_minus_corrupted_host": metrics["error_decomposition_reliability"] - metrics["corrupted_host"]},
        "state_metrics": totals,
        "identity_gate": {"max_abs_logit_difference": identity_max, "passed": identity_max <= 1e-6},
        "finite": finite,
        "decision": "IMPROVED_OVER_CORRECTED_B" if finite and metrics["error_decomposition_reliability"] > metrics["corrected_b"] else "NOT_IMPROVED_OVER_CORRECTED_B",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
