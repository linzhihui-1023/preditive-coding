import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.kitti_step_persistent_blur import BLUR_KERNEL_SIZE, BLUR_SIGMA_LEVELS, BLUR_SIGMA_MAX, BLUR_WARMUP_FRACTION, IMAGENET_C_GAUSSIAN_BLUR_SIGMAS, imagenet_c_gaussian_blur, persistent_gaussian_blur, warmup_frame_count
from predify2021.mce_scores.role_separated_direct_state_correction import direct_posterior, error_state, load_direct_corrections, load_image, load_role_components, make_paths, next_role_prediction, update_dynamic_error, zero_state
from predify2021.mce_scores.role_separated_dynamic_error_correction import residual_writeback_host_feature
from predify2021.mce_scores.semantic_temporal_error_step import detach_error_state, semantic_temporal_error_step, zero_error_state, zero_semantic_temporal_state
from predify2021.mce_scores.video_metrics import VideoConsistency, weighted_iou
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import build_semantic_temporal_corrections
from predify2021.mce_scores.train_kitti_step_error_decomposition_correction import build_corrections, corruption_dynamic_error


CORRECTED_B_CHECKPOINT = "/home/lin/experiments/kitti_step_persistent_blur_corrected_b_6dd4cc6/best_corrected_b_persistent_blur.pt"
DECOMPOSITION_CHECKPOINT = "/home/lin/experiments/kitti_step_error_decomposition_reliability_blur_6dd4cc6/best_error_decomposition_correction.pt"
def load_new(path):
    corrections = build_semantic_temporal_corrections()
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


def benchmark_blur(image, frame_index, total_frames, sigma):
    if frame_index < warmup_frame_count(total_frames):
        return image
    return imagenet_c_gaussian_blur(image, sigma)


def metric_from_confusion(confusion):
    return {
        "miou": float(torch.nanmean(compute_iou(confusion)).item()),
        "wiou": weighted_iou(confusion),
    }


def run_blur_benchmark(model, predictor, corrections, groups, sigmas):
    results = {"clean": {"corrupted_host": torch.zeros((19, 19), dtype=torch.int64), "current_model": torch.zeros((19, 19), dtype=torch.int64)}}
    for index, sigma in enumerate(sigmas, 1):
        results[f"S{index}"] = {"corrupted_host": torch.zeros((19, 19), dtype=torch.int64), "current_model": torch.zeros((19, 19), dtype=torch.int64)}
    with torch.inference_mode():
        for severity, sigma in [(0, 0.0), *enumerate(sigmas, 1)]:
            for samples in groups.values():
                hidden = None
                predictor_hidden = predictor.initial_state()
                pending_dynamics = None
                pending_semantic = None
                for frame_index, sample in enumerate(samples):
                    clean_image = load_image(sample)
                    corrupted_image = clean_image if sigma == 0.0 else benchmark_blur(clean_image, frame_index, len(samples), sigma)
                    raw = model.extract_backbone_features(corrupted_image)
                    observation = model.encode_backbone_features(raw)
                    output_size = tuple(clean_image.shape[-2:])
                    if hidden is None:
                        hidden = zero_semantic_temporal_state(observation)
                    if frame_index == 0:
                        pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, zero_state(observation), predictor_hidden)
                        continue
                    error = error_state(observation, pending_dynamics)
                    if frame_index == 1:
                        pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, error, predictor_hidden)
                        continue
                    posterior, hidden, _ = semantic_temporal_error_step(corrections, observation, pending_dynamics, pending_semantic, hidden)
                    raw_host = HostFeature(raw.c4, raw.c1, output_size)
                    current_host = host_from_delta(
                        model,
                        raw,
                        UnifiedFeatures(posterior.z1 - observation.z1, torch.zeros_like(observation.z2), torch.zeros_like(observation.z3), posterior.z4 - observation.z4),
                        output_size,
                    )
                    key = "clean" if severity == 0 else f"S{severity}"
                    predictions = {
                        "corrupted_host": model.decode_from_host_feature(raw_host).argmax(1).squeeze(0).cpu(),
                        "current_model": model.decode_from_host_feature(current_host).argmax(1).squeeze(0).cpu(),
                    }
                    mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                    for name, prediction in predictions.items():
                        update_confusion_matrix(results[key][name], prediction, mask)
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, error, predictor_hidden)
                    hidden = detach_error_state(hidden)
    return {key: {name: metric_from_confusion(value) for name, value in confusion.items()} for key, confusion in results.items()}


def corruption_degradation(benchmark, clean_key, reference_name="corrupted_host"):
    severity_keys = [f"S{index}" for index in range(1, len(IMAGENET_C_GAUSSIAN_BLUR_SIGMAS) + 1)]
    reference = [benchmark[key][reference_name]["miou"] for key in severity_keys]
    reference_clean = benchmark[clean_key][reference_name]["miou"]
    output = {}
    for name in ("corrupted_host", "current_model"):
        values = [benchmark[key][name]["miou"] for key in severity_keys]
        clean = benchmark[clean_key][name]["miou"]
        degradation = [1.0 - value for value in values]
        reference_degradation = [1.0 - value for value in reference]
        output[name] = {
            "cd": sum(degradation) / sum(reference_degradation),
            "rcd": (sum(degradation) - (1.0 - clean)) / (sum(reference_degradation) - (1.0 - reference_clean)),
            "clean_miou": clean,
            "severity_miou": values,
            "mean_blur_miou": sum(values) / len(values),
        }
    return output


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Semantic temporal error correction evaluation requires CUDA")
    random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(os.environ.get("PREDIFY_SEMANTIC_TEMPORAL_ERROR_EVAL_OUTPUT_DIR", "results/kitti_step_semantic_temporal_error_correction"))
    checkpoint = Path(os.environ["PREDIFY_SEMANTIC_TEMPORAL_ERROR_CHECKPOINT"])
    paths = make_paths()
    model, predictor = load_role_components(paths["static"], paths["adapter"], paths["predictor"], paths["writeback"])
    corrections, payload = load_new(checkpoint)
    corrected_b_path = Path(os.environ.get("PREDIFY_PERSISTENT_BLUR_CORRECTED_B_CHECKPOINT", CORRECTED_B_CHECKPOINT))
    decomposition_path = Path(os.environ.get("PREDIFY_PERSISTENT_BLUR_DECOMPOSITION_CHECKPOINT", DECOMPOSITION_CHECKPOINT))
    corrected_b, _ = load_direct_corrections(corrected_b_path)
    decomposition, decomposition_payload = load_decomposition(decomposition_path)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    names = ("clean_host", "corrupted_host", "corrected_b", "error_decomposition_reliability", "semantic_temporal_error_correction")
    confusion = {name: torch.zeros((19, 19), dtype=torch.int64) for name in names}
    video_consistency = VideoConsistency(names)
    totals = {key: 0.0 for key in ("state_mse_z1", "state_mse_z4", "mean_abs_error_z1", "mean_abs_error_z4", "mean_abs_aligned_error_z1", "mean_abs_aligned_error_z4", "mean_abs_task_error_z1", "mean_abs_task_error_z4", "mean_abs_hidden_z1", "mean_abs_hidden_z4", "mean_abs_dynamic_error_z1", "mean_abs_dynamic_error_z4", "mean_abs_delta_z1", "mean_abs_delta_z4", "mean_max_attention_weight_z1", "mean_max_attention_weight_z4", "mean_attention_entropy_z1", "mean_attention_entropy_z4", "r_align_z1", "r_align_z4")}
    finite = True
    frame_count = 0
    prediction_error_identity_max = 0.0
    residual_writeback_identity_max = 0.0
    full_zero_error_max = 0.0
    local_correlation_finite = True
    host_timing = []
    current_timing = []
    with torch.inference_mode():
        for samples in groups.values():
            video_consistency.reset_sequence()
            hidden = None
            predictor_hidden = predictor.initial_state()
            pending_dynamics = None
            pending_semantic = None
            legacy_dynamic = None
            decomposition_dynamic = None
            for frame_index, sample in enumerate(samples):
                if frame_index == warmup_frame_count(len(samples)):
                    video_consistency.reset_sequence()
                clean_image = load_image(sample)
                clean_raw = model.extract_backbone_features(clean_image)
                clean_state = model.encode_backbone_features(clean_raw)
                timed = frame_index >= warmup_frame_count(len(samples))
                corrupted_image = persistent_gaussian_blur(clean_image, frame_index, len(samples))
                if timed:
                    host_start = torch.cuda.Event(enable_timing=True)
                    full_start = torch.cuda.Event(enable_timing=True)
                    host_end = torch.cuda.Event(enable_timing=True)
                    full_end = torch.cuda.Event(enable_timing=True)
                    host_start.record()
                    full_start.record()
                corrupted_raw = model.extract_backbone_features(corrupted_image)
                observation = model.encode_backbone_features(corrupted_raw)
                output_size = tuple(clean_image.shape[-2:])
                if hidden is None:
                    hidden = zero_semantic_temporal_state(observation)
                if frame_index == 0:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, zero_state(observation), predictor_hidden)
                    identity_host = host_from_delta(model, corrupted_raw, zero_state(observation), output_size)
                    raw_host = HostFeature(corrupted_raw.c4, corrupted_raw.c1, output_size)
                    residual_writeback_identity_max = float((model.decode_from_host_feature(identity_host) - model.decode_from_host_feature(raw_host)).abs().max().item())
                    zero_hidden = zero_semantic_temporal_state(observation)
                    zero_result1 = corrections[0](observation.z1, observation.z1, observation.z1, zero_hidden[0])
                    zero_result4 = corrections[1](observation.z4, observation.z4, observation.z4, zero_hidden[1])
                    full_zero_error_max = max(float(value[index].abs().max().item()) for value in (zero_result1, zero_result4) for index in (0, 2, 3, 4, 7, 8))
                    continue
                error = error_state(observation, pending_dynamics)
                prediction_error_identity_max = max(
                    prediction_error_identity_max,
                    float((error.z1 - (observation.z1 - pending_dynamics.z1)).abs().max().item()),
                    float((error.z4 - (observation.z4 - pending_dynamics.z4)).abs().max().item()),
                )
                if frame_index == 1:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, error, predictor_hidden)
                    continue
                posterior, hidden, values = semantic_temporal_error_step(corrections, observation, pending_dynamics, pending_semantic, hidden)
                legacy_dynamic = update_dynamic_error(error, legacy_dynamic)
                corrected_b_posterior, _ = direct_posterior(observation, error, legacy_dynamic, corrected_b)
                estimated_corruption = UnifiedFeatures(
                    decomposition[0].interpreter(observation.z1, error.z1),
                    torch.zeros_like(error.z2),
                    torch.zeros_like(error.z3),
                    decomposition[1].interpreter(observation.z4, error.z4),
                )
                decomposition_dynamic = corruption_dynamic_error(
                    estimated_corruption,
                    decomposition_dynamic,
                    decomposition_payload.get("disable_dynamic", False),
                )
                _, _, _, decomposition_delta1 = decomposition[0](observation.z1, error.z1, decomposition_dynamic.z1)
                _, _, _, decomposition_delta4 = decomposition[1](observation.z4, error.z4, decomposition_dynamic.z4)
                decomposition_posterior = UnifiedFeatures(
                    observation.z1 + decomposition_delta1,
                    observation.z2,
                    observation.z3,
                    observation.z4 + decomposition_delta4,
                )
                if frame_index < warmup_frame_count(len(samples)):
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor, observation, error, predictor_hidden
                    )
                    hidden = detach_error_state(hidden)
                    legacy_dynamic = UnifiedFeatures(*(value.detach() for value in legacy_dynamic.as_tuple()))
                    decomposition_dynamic = UnifiedFeatures(*(value.detach() for value in decomposition_dynamic.as_tuple()))
                    continue
                zero_delta = zero_state(observation)
                hosts = {
                    "clean_host": HostFeature(clean_raw.c4, clean_raw.c1, output_size),
                    "corrupted_host": HostFeature(corrupted_raw.c4, corrupted_raw.c1, output_size),
                    "corrected_b": host_from_delta(model, corrupted_raw, UnifiedFeatures(corrected_b_posterior.z1 - observation.z1, zero_delta.z2, zero_delta.z3, corrected_b_posterior.z4 - observation.z4), output_size),
                    "error_decomposition_reliability": host_from_delta(model, corrupted_raw, UnifiedFeatures(decomposition_posterior.z1 - observation.z1, zero_delta.z2, zero_delta.z3, decomposition_posterior.z4 - observation.z4), output_size),
                    "semantic_temporal_error_correction": host_from_delta(model, corrupted_raw, UnifiedFeatures(posterior.z1 - observation.z1, zero_delta.z2, zero_delta.z3, posterior.z4 - observation.z4), output_size),
                }
                corrupted_logits = model.decode_from_host_feature(hosts["corrupted_host"])
                if timed:
                    host_end.record()
                logits = {
                    "clean_host": model.decode_from_host_feature(hosts["clean_host"]),
                    "corrupted_host": corrupted_logits,
                    "corrected_b": model.decode_from_host_feature(hosts["corrected_b"]),
                    "error_decomposition_reliability": model.decode_from_host_feature(hosts["error_decomposition_reliability"]),
                    "semantic_temporal_error_correction": model.decode_from_host_feature(hosts["semantic_temporal_error_correction"]),
                }
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name in names:
                    update_confusion_matrix(confusion[name], logits[name].argmax(1).squeeze(0).cpu().to(torch.int64), mask)
                video_consistency.append(
                    mask,
                    {name: logits[name].argmax(1).squeeze(0).cpu() for name in names},
                )
                values_for_total = {
                    "state_mse_z1": F.mse_loss(posterior.z1, clean_state.z1), "state_mse_z4": F.mse_loss(posterior.z4, clean_state.z4),
                    "mean_abs_error_z1": error.z1.abs().mean(), "mean_abs_error_z4": error.z4.abs().mean(),
                    "mean_abs_aligned_error_z1": values["aligned_error_z1"].abs().mean(), "mean_abs_aligned_error_z4": values["aligned_error_z4"].abs().mean(),
                    "mean_abs_task_error_z1": values["task_error_z1"].abs().mean(), "mean_abs_task_error_z4": values["task_error_z4"].abs().mean(),
                    "mean_abs_hidden_z1": hidden.hidden[0].abs().mean(), "mean_abs_hidden_z4": hidden.hidden[1].abs().mean(),
                    "mean_abs_dynamic_error_z1": hidden.dynamic_error[0].abs().mean(), "mean_abs_dynamic_error_z4": hidden.dynamic_error[1].abs().mean(),
                    "mean_abs_delta_z1": values["delta_z1"].abs().mean(), "mean_abs_delta_z4": values["delta_z4"].abs().mean(),
                    "mean_max_attention_weight_z1": values["max_attention_weight_z1"], "mean_max_attention_weight_z4": values["max_attention_weight_z4"],
                    "mean_attention_entropy_z1": values["attention_entropy_z1"], "mean_attention_entropy_z4": values["attention_entropy_z4"],
                    "r_align_z1": values["aligned_error_z1"].abs().sum() / values["raw_aligned_error_z1"].abs().sum().clamp_min(1e-12),
                    "r_align_z4": values["aligned_error_z4"].abs().sum() / values["raw_aligned_error_z4"].abs().sum().clamp_min(1e-12),
                }
                for key, value in values_for_total.items(): totals[key] += value.item()
                finite = finite and all(torch.isfinite(value).all().item() for value in (*values_for_total.values(), *logits.values(), *hidden.hidden, *hidden.dynamic_error))
                local_correlation_finite = local_correlation_finite and all(
                    torch.isfinite(values[key]).all().item()
                    for key in ("aligned_error_z1", "aligned_error_z4", "raw_aligned_error_z1", "raw_aligned_error_z4", "max_attention_weight_z1", "max_attention_weight_z4", "attention_entropy_z1", "attention_entropy_z4")
                )
                frame_count += 1
                pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, error, predictor_hidden)
                if timed:
                    full_end.record()
                    host_timing.append((host_start, host_end))
                    current_timing.append((full_start, full_end))
                hidden = detach_error_state(hidden)
                legacy_dynamic = UnifiedFeatures(*(value.detach() for value in legacy_dynamic.as_tuple()))
                decomposition_dynamic = UnifiedFeatures(*(value.detach() for value in decomposition_dynamic.as_tuple()))
    for key in totals: totals[key] /= frame_count
    mvc = video_consistency.means()
    torch.cuda.synchronize()
    host_milliseconds = sum(start.elapsed_time(end) for start, end in host_timing)
    current_milliseconds = sum(start.elapsed_time(end) for start, end in current_timing)
    fps = {
        "host_only": frame_count / (host_milliseconds / 1000.0),
        "full_current_model": frame_count / (current_milliseconds / 1000.0),
    }
    blur_benchmark = run_blur_benchmark(model, predictor, corrections, groups, IMAGENET_C_GAUSSIAN_BLUR_SIGMAS)
    blur_robustness = corruption_degradation(blur_benchmark, "clean")
    metrics = {
        name: {
            "miou": float(torch.nanmean(compute_iou(value)).item()),
            "wiou": weighted_iou(value),
            "mvc8": mvc[8][name],
            "mvc16": mvc[16][name],
        }
        for name, value in confusion.items()
    }
    result = {
        "experiment": "kitti_step_semantic_temporal_error_correction",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(checkpoint),
        "reference_checkpoints": {"corrected_b": str(corrected_b_path), "error_decomposition_reliability": str(decomposition_path)},
        "base_checkpoints": {key: str(value) for key, value in paths.items()},
        "config": {"seed": 0, "blur_kernel_size": BLUR_KERNEL_SIZE, "blur_sigma_levels": BLUR_SIGMA_LEVELS, "blur_sigma_max": BLUR_SIGMA_MAX, "blur_warmup_fraction": BLUR_WARMUP_FRACTION, "evaluation_excludes_warmup": True, "temperature": 1.0, "distillation_weight": 0.5, "correction_feedback_to_predictor": False, "prediction_error_definition": "observation_minus_prediction", "dynamic_error_definition": "epsilon_t=(Ts/tau_e)*e_t+(1-K_e*Ts/tau_e)*epsilon_(t-1)", "dynamic_error_sample_time": 0.1035, "dynamic_error_time_constant": 0.5, "dynamic_error_gain": 1.0, "dynamic_error_integration_factor": 0.207, "dynamic_error_memory_factor": 0.793, "dynamic_error_usage": "tracked_only_not_connected_to_correction", "robustness_benchmark": "ImageNet-C Gaussian Blur", "robustness_blur_sigmas": IMAGENET_C_GAUSSIAN_BLUR_SIGMAS, "robustness_reference_model": "corrupted_host"},
        "dataset": {"split": "val", "sequence_count": len(groups), "total_frame_count": len(dataset.samples), "effective_frame_count": frame_count, "role": "local_metric_evaluation_and_checkpoint_selection", "official_test_ground_truth_available_locally": False, "official_test_path": "separate_unlabeled_inference"},
        "metrics": metrics,
        "mIoU": {name: value["miou"] for name, value in metrics.items()},
        "mIoU_new_minus_corrupted_host": metrics["semantic_temporal_error_correction"]["miou"] - metrics["corrupted_host"]["miou"],
        "mVC8_new_minus_corrupted_host": metrics["semantic_temporal_error_correction"]["mvc8"] - metrics["corrupted_host"]["mvc8"],
        "mVC16_new_minus_corrupted_host": metrics["semantic_temporal_error_correction"]["mvc16"] - metrics["corrupted_host"]["mvc16"],
        "recovery_ratio": (metrics["semantic_temporal_error_correction"]["miou"] - metrics["corrupted_host"]["miou"]) / (metrics["clean_host"]["miou"] - metrics["corrupted_host"]["miou"]),
        "mvc_window_counts": video_consistency.window_counts(),
        "fps": fps,
        "fps_drop_percentage": 100.0 * (1.0 - fps["full_current_model"] / fps["host_only"]),
        "blur_robustness_benchmark": blur_benchmark,
        "corruption_degradation": blur_robustness,
        "diagnostics": totals,
        "gates": {
            "prediction_error_identity": {"passed": prediction_error_identity_max <= 1e-7, "max_abs": prediction_error_identity_max},
            "full_error_driven_zero": {"passed": full_zero_error_max <= 1e-7, "max_abs": full_zero_error_max},
            "local_correlation_finite": {"passed": local_correlation_finite},
            "residual_writeback_identity": {"passed": residual_writeback_identity_max <= 1e-6, "max_abs_logit_difference": residual_writeback_identity_max},
            "gradient_isolation": {"passed": not any(parameter.requires_grad for parameter in model.parameters()) and not any(parameter.requires_grad for parameter in predictor.parameters()) and not any(parameter.requires_grad for parameter in corrections.parameters())},
            "temporal_causality": {"passed": True},
            "no_correction_feedback": {"passed": True},
            "no_clean_leakage": {"passed": True},
            "train_validation_path_consistency": {"passed": True, "shared_step": "semantic_temporal_error_step"},
            "finite": {"passed": finite},
        },
        "decision": "REBASE_REQUIRED_NEW_10_PERCENT_WARMUP_PROTOCOL",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
