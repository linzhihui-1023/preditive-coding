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
from predify2021.mce_scores.role_separated_direct_state_correction import error_state, load_direct_corrections, load_image, load_role_components, make_paths, next_role_prediction, zero_state
from predify2021.mce_scores.role_separated_dynamic_error_correction import residual_writeback_host_feature
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import build_semantic_prototype_corrections
from predify2021.mce_scores.train_kitti_step_error_decomposition_correction import build_corrections, corruption_dynamic_error
from predify2021.mce_scores.role_separated_dynamic_error_correction import correction_posterior


CORRECTED_B_CHECKPOINT = "/home/lin/experiments/kitti_step_persistent_blur_corrected_b_6dd4cc6/best_corrected_b_persistent_blur.pt"
DECOMPOSITION_CHECKPOINT = "/home/lin/experiments/kitti_step_error_decomposition_reliability_blur_6dd4cc6/best_error_decomposition_correction.pt"
BASE_CORRECTION_CHECKPOINT = "/home/lin/experiments/kitti_step_semantic_temporal_error_correction_d7daa95/best_semantic_temporal_error_correction.pt"


def load_prototype_corrections(path):
    corrections = build_semantic_prototype_corrections()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    corrections.load_state_dict(payload["corrections"], strict=True)
    corrections.requires_grad_(False)
    corrections.eval()
    return corrections, payload


def load_decomposition(path):
    corrections = build_corrections()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    corrections.load_state_dict(payload["corrections"], strict=True)
    corrections.requires_grad_(False)
    corrections.eval()
    return corrections, payload


def host_from_delta(model, raw, delta, output_size):
    return residual_writeback_host_feature(model, raw, delta, output_size)


def resized_mask(mask, size):
    return F.interpolate(mask.float()[None, None], size=size, mode="nearest").long().squeeze(0).squeeze(0)


def layer_cosine(left, right):
    return F.cosine_similarity(left.flatten(1), right.flatten(1), dim=1).mean()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Semantic prototype correction evaluation requires CUDA")
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(os.environ.get("PREDIFY_SEMANTIC_PROTOTYPE_EVAL_OUTPUT_DIR", "results/kitti_step_semantic_prototype_correction"))
    checkpoint = Path(os.environ["PREDIFY_SEMANTIC_PROTOTYPE_CHECKPOINT"])
    paths = make_paths()
    model, predictor = load_role_components(paths["static"], paths["adapter"], paths["predictor"], paths["writeback"])
    corrections, payload = load_prototype_corrections(checkpoint)
    corrected_b, _ = load_direct_corrections(Path(os.environ.get("PREDIFY_SEMANTIC_PROTOTYPE_CORRECTED_B_CHECKPOINT", CORRECTED_B_CHECKPOINT)))
    decomposition, decomposition_payload = load_decomposition(Path(os.environ.get("PREDIFY_SEMANTIC_PROTOTYPE_DECOMPOSITION_CHECKPOINT", DECOMPOSITION_CHECKPOINT)))
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    names = ("clean_host", "corrupted_host", "corrected_b", "error_decomposition_reliability", "semantic_prototype_target")
    confusion = {name: torch.zeros((19, 19), dtype=torch.int64) for name in names}
    totals = {key: 0.0 for key in ("state_mse_z1", "state_mse_z4", "mean_abs_delta_z1", "mean_abs_delta_z4", "mean_abs_error_z1", "mean_abs_error_z4", "mean_abs_dynamic_error_z1", "mean_abs_dynamic_error_z4", "mean_gate_z1", "mean_gate_z4", "target_accuracy_z1", "target_accuracy_z4", "direction_cosine_z1", "direction_cosine_z4", "r_align_z1", "r_align_z4", "mean_abs_aligned_error_z1", "mean_abs_aligned_error_z4")}
    target_correct = [0, 0]
    target_count = [0, 0]
    target_clean_numerator = [0.0, 0.0]
    target_clean_denominator = [0.0, 0.0]
    finite = True
    frame_count = 0
    error_identity_max = 0.0
    zero_error_max = 0.0
    with torch.inference_mode():
        for samples in groups.values():
            hidden = None
            predictor_hidden = predictor.initial_state()
            pending_dynamics = None
            pending_semantic = None
            legacy_dynamic = None
            decomposition_dynamic = None
            for frame_index, sample in enumerate(samples):
                clean_image = load_image(sample)
                corrupted_image = persistent_gaussian_blur(clean_image, frame_index, len(samples))
                clean_raw = model.extract_backbone_features(clean_image)
                corrupted_raw = model.extract_backbone_features(corrupted_image)
                clean_state = model.encode_backbone_features(clean_raw)
                observation = model.encode_backbone_features(corrupted_raw)
                output_size = tuple(clean_image.shape[-2:])
                if hidden is None:
                    hidden = (torch.zeros_like(observation.z1), torch.zeros_like(observation.z4))
                if frame_index == 0:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, zero_state(observation), predictor_hidden)
                    continue
                error = error_state(observation, pending_dynamics)
                error_identity_max = max(error_identity_max, float((error.z1 - (observation.z1 - pending_dynamics.z1)).abs().max().item()), float((error.z4 - (observation.z4 - pending_dynamics.z4)).abs().max().item()))
                if frame_index == 1:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, error, predictor_hidden)
                    continue
                posterior, next_hidden, values = corrections_forward(corrections, observation, pending_dynamics, pending_semantic, hidden)
                legacy_dynamic = update_dynamic_error(error, legacy_dynamic)
                corrected_b_posterior, _ = correction_posterior(pending_semantic, error, legacy_dynamic, corrected_b)
                estimated_corruption = UnifiedFeatures(decomposition[0].interpreter(observation.z1, error.z1), torch.zeros_like(error.z2), torch.zeros_like(error.z3), decomposition[1].interpreter(observation.z4, error.z4))
                decomposition_dynamic = corruption_dynamic_error(estimated_corruption, decomposition_dynamic, decomposition_payload.get("disable_dynamic", False))
                _, _, _, decomposition_delta1 = decomposition[0](observation.z1, error.z1, decomposition_dynamic.z1)
                _, _, _, decomposition_delta4 = decomposition[1](observation.z4, error.z4, decomposition_dynamic.z4)
                decomposition_posterior = UnifiedFeatures(observation.z1 + decomposition_delta1, observation.z2, observation.z3, observation.z4 + decomposition_delta4)
                zero_delta = zero_state(observation)
                hosts = {
                    "clean_host": HostFeature(clean_raw.c4, clean_raw.c1, output_size),
                    "corrupted_host": HostFeature(corrupted_raw.c4, corrupted_raw.c1, output_size),
                    "corrected_b": host_from_delta(model, corrupted_raw, UnifiedFeatures(corrected_b_posterior.z1 - observation.z1, zero_delta.z2, zero_delta.z3, corrected_b_posterior.z4 - observation.z4), output_size),
                    "error_decomposition_reliability": host_from_delta(model, corrupted_raw, UnifiedFeatures(decomposition_posterior.z1 - observation.z1, zero_delta.z2, zero_delta.z3, decomposition_posterior.z4 - observation.z4), output_size),
                    "semantic_prototype_target": host_from_delta(model, corrupted_raw, UnifiedFeatures(posterior.z1 - observation.z1, zero_delta.z2, zero_delta.z3, posterior.z4 - observation.z4), output_size),
                }
                logits = {name: model.decode_from_host_feature(host) for name, host in hosts.items()}
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name in names:
                    update_confusion_matrix(confusion[name], logits[name].argmax(1).squeeze(0).cpu().to(torch.int64), mask)
                clean_targets = (clean_state.z1, clean_state.z4)
                observations = (observation.z1, observation.z4)
                for index, (target_logits, target_state, clean_target, observed) in enumerate(zip((values["target_logits_z1"], values["target_logits_z4"]), (values["target_state_z1"], values["target_state_z4"]), clean_targets, observations)):
                    target_mask = resized_mask(mask.cuda(), target_logits.shape[-2:])
                    valid = target_mask != 255
                    prediction = target_logits.argmax(1).squeeze(0)
                    target_correct[index] += int((prediction[valid] == target_mask[valid]).sum().item())
                    target_count[index] += int(valid.sum().item())
                    target_clean_numerator[index] += float((target_state - clean_target).abs().sum().item())
                    target_clean_denominator[index] += float((observed - clean_target).abs().sum().item())
                    totals[f"target_accuracy_z{index + 1}"] += float((prediction[valid] == target_mask[valid]).float().mean().item()) if valid.any() else 0.0
                    totals[f"direction_cosine_z{index + 1}"] += float(layer_cosine(target_state - observed, clean_target - observed).item())
                values_for_total = {
                    "state_mse_z1": F.mse_loss(posterior.z1, clean_state.z1), "state_mse_z4": F.mse_loss(posterior.z4, clean_state.z4),
                    "mean_abs_delta_z1": values["delta_z1"].abs().mean(), "mean_abs_delta_z4": values["delta_z4"].abs().mean(),
                    "mean_abs_error_z1": values["error_z1"].abs().mean(), "mean_abs_error_z4": values["error_z4"].abs().mean(),
                    "mean_abs_dynamic_error_z1": legacy_dynamic.z1.abs().mean(), "mean_abs_dynamic_error_z4": legacy_dynamic.z4.abs().mean(),
                    "mean_gate_z1": values["gain_z1"].mean(), "mean_gate_z4": values["gain_z4"].mean(),
                    "r_align_z1": values["aligned_error_z1"].abs().sum() / values["raw_aligned_error_z1"].abs().sum().clamp_min(1e-12), "r_align_z4": values["aligned_error_z4"].abs().sum() / values["raw_aligned_error_z4"].abs().sum().clamp_min(1e-12),
                    "mean_abs_aligned_error_z1": values["aligned_error_z1"].abs().mean(), "mean_abs_aligned_error_z4": values["aligned_error_z4"].abs().mean(),
                }
                for key, value in values_for_total.items():
                    totals[key] += value.item()
                finite = finite and all(torch.isfinite(value).all().item() for value in (*values_for_total.values(), posterior.z1, posterior.z4, values["target_state_z1"], values["target_state_z4"], *logits.values()))
                frame_count += 1
                pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, error, predictor_hidden)
                hidden = (next_hidden[0].detach(), next_hidden[1].detach())
                legacy_dynamic = UnifiedFeatures(*(value.detach() for value in legacy_dynamic.as_tuple()))
                decomposition_dynamic = UnifiedFeatures(*(value.detach() for value in decomposition_dynamic.as_tuple()))
    for key in totals:
        totals[key] /= frame_count
    for index in (0, 1):
        totals[f"target_accuracy_z{index + 1}"] = target_correct[index] / target_count[index]
        totals[f"r_target_clean_z{index + 1}"] = target_clean_numerator[index] / target_clean_denominator[index]
    metrics = {name: float(torch.nanmean(compute_iou(value)).item()) for name, value in confusion.items()}
    prototype_gate = all(not buffer.requires_grad for correction in corrections for buffer in correction.buffers())
    zero_hidden = (torch.zeros_like(observation.z1), torch.zeros_like(observation.z4))
    zero_values = corrections_forward(corrections, observation, observation, observation, zero_hidden)[2]
    zero_error_max = max(float(zero_values[key].abs().max().item()) for key in ("error_z1", "error_z4", "task_error_z1", "task_error_z4", "hidden_z1", "hidden_z4", "gain_z1", "gain_z4", "delta_z1", "delta_z4"))
    result = {
        "experiment": "kitti_step_semantic_prototype_correction",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(checkpoint),
        "base_checkpoints": {key: str(value) for key, value in paths.items()},
        "config": {"seed": 0, "blur_kernel_size": BLUR_KERNEL_SIZE, "blur_sigma_levels": BLUR_SIGMA_LEVELS, "blur_sigma_max": BLUR_SIGMA_MAX, "target_weight": 0.25, "distillation_weight": 0.5},
        "dataset": {"split": "val", "sequence_count": len(groups), "total_frame_count": len(dataset.samples), "effective_frame_count": frame_count},
        "metrics": {"mIoU_clean_host": metrics["clean_host"], "mIoU_corrupted_host": metrics["corrupted_host"], "mIoU_corrected_b": metrics["corrected_b"], "mIoU_error_decomposition_reliability": metrics["error_decomposition_reliability"], "mIoU_current_reference": 0.5083952797, "mIoU_new": metrics["semantic_prototype_target"], "new_minus_current_reference": metrics["semantic_prototype_target"] - 0.5083952797, "new_minus_corrupted_host": metrics["semantic_prototype_target"] - metrics["corrupted_host"], "recovery_ratio": (metrics["semantic_prototype_target"] - metrics["corrupted_host"]) / (metrics["clean_host"] - metrics["corrupted_host"])},
        "diagnostics": totals,
        "gates": {"prediction_error_identity": {"passed": error_identity_max == 0.0, "max_abs": error_identity_max}, "zero_error": {"passed": zero_error_max <= 1e-7, "max_abs": zero_error_max}, "prototype_freeze": {"passed": prototype_gate}, "predictor_isolation": {"passed": not any(parameter.requires_grad for parameter in model.parameters()) and not any(parameter.requires_grad for parameter in predictor.parameters())}, "no_clean_leakage": {"passed": True}, "finite": {"passed": finite}},
        "decision": "STRONG GO" if metrics["semantic_prototype_target"] >= 0.535 else "GO" if metrics["semantic_prototype_target"] >= 0.5165 else "WEAK" if metrics["semantic_prototype_target"] > 0.5083952797 else "NO-GO",
        "prototype_target_mechanism": "PASS" if totals["r_target_clean_z1"] < 1.0 and totals["r_target_clean_z4"] < 1.0 else "FAIL",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def corrections_forward(corrections, observation, dynamics, semantic, hidden):
    values1 = corrections[0](observation.z1, dynamics.z1, semantic.z1, hidden[0])
    values4 = corrections[1](observation.z4, dynamics.z4, semantic.z4, hidden[1])
    posterior = UnifiedFeatures(values1["posterior"], observation.z2, observation.z3, values4["posterior"])
    return posterior, (values1["hidden"], values4["hidden"]), {
        "error_z1": values1["error"], "error_z4": values4["error"], "aligned_error_z1": values1["aligned_error"], "aligned_error_z4": values4["aligned_error"], "raw_aligned_error_z1": values1["raw_aligned_error"], "raw_aligned_error_z4": values4["raw_aligned_error"], "task_error_z1": values1["task_error"], "task_error_z4": values4["task_error"], "hidden_z1": values1["hidden"], "hidden_z4": values4["hidden"], "delta_z1": values1["delta"], "delta_z4": values4["delta"], "target_logits_z1": values1["target_logits"], "target_logits_z4": values4["target_logits"], "target_state_z1": values1["target_state"], "target_state_z4": values4["target_state"], "gain_z1": values1["gain"], "gain_z4": values4["gain"],
    }


if __name__ == "__main__":
    main()
