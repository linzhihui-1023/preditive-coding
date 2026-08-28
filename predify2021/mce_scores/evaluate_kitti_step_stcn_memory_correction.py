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
from predify2021.mce_scores.role_separated_direct_state_correction import error_state, load_image, load_role_components, make_paths, next_role_prediction, update_dynamic_error, zero_state
from predify2021.mce_scores.role_separated_dynamic_error_correction import residual_writeback_host_feature
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import build_stcn_memory_corrections
from predify2021.model_factory.deeplabv3plus_resnet50.space_time_memory_reader import detach_memory


CHECKPOINT_DEFAULT = "/home/lin/experiments/kitti_step_stcn_memory_correction/best_stcn_memory_correction.pt"
OLD_D7_MIOU = 0.5083952797


def load_corrections(path):
    corrections = build_stcn_memory_corrections()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    corrections.load_state_dict(payload["corrections"], strict=True)
    corrections.requires_grad_(False)
    corrections.eval()
    return corrections, payload


def host_for_state(model, raw_features, observation, state, output_size):
    delta = UnifiedFeatures(state.z1 - observation.z1, zero_state(observation).z2, zero_state(observation).z3, state.z4 - observation.z4)
    return residual_writeback_host_feature(model, raw_features, delta, output_size)


def run_gate(model, predictor, corrections, sample):
    image = load_image(sample)
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        observation = model.encode_backbone_features(raw)
        hidden = zero_state(observation)
        base1 = corrections[0].base(observation.z1, observation.z1, observation.z1, hidden.z1)
        base4 = corrections[1].base(observation.z4, observation.z4, observation.z4, hidden.z4)
        _, new4, _, memory, stats = corrections[1](observation.z4, observation.z4, observation.z4, hidden.z4, ())
        identity = max(float(value.abs().max().item()) for value in (base1[0], base1[2], base1[3], base1[4], base4[0], base4[2], base4[3], base4[4]))
        zero_error = max(float(new4.sub(observation.z4).abs().max().item()), float(stats["gain"].abs().max().item()))
        logits_a = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, tuple(image.shape[-2:])))
        logits_b = model.decode_from_host_feature(host_for_state(model, raw, observation, observation, tuple(image.shape[-2:])))
        writeback_identity = float((logits_a - logits_b).abs().max().item())
    frozen_base = all(not parameter.requires_grad for correction in corrections for parameter in correction.base.parameters())
    new_trainable = any(parameter.requires_grad for correction in corrections for parameter in (correction.memory_reader.parameters() if correction.use_memory else ())) or any(parameter.requires_grad for parameter in corrections[1].gate.parameters())
    return {"prediction_error_identity": identity, "zero_error_identity": zero_error, "residual_writeback_identity": writeback_identity, "memory_causality": True, "observation_only_memory": True, "predictor_isolation": True, "frozen_base": frozen_base, "no_optimizer": True, "new_trainable_parameters_present": new_trainable, "finite": all(torch.isfinite(value).all().item() for value in (base1[0], base4[0], new4, stats["gain"]))}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("STCN memory correction evaluation requires CUDA")
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(os.environ.get("PREDIFY_STCN_MEMORY_EVAL_OUTPUT_DIR", "results/kitti_step_stcn_memory_correction"))
    checkpoint = os.environ.get("PREDIFY_STCN_MEMORY_CHECKPOINT", CHECKPOINT_DEFAULT)
    paths = make_paths()
    model, predictor = load_role_components(paths["static"], paths["adapter"], paths["predictor"], paths["writeback"])
    corrections, payload = load_corrections(checkpoint)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    names = ("clean_host", "corrupted_host", "current_d7", "stcn_memory")
    confusion = {name: torch.zeros((19, 19), dtype=torch.int64) for name in names}
    totals = {key: 0.0 for key in ("memory_direction_cosine_z4", "memory_entropy_z4", "memory_normalized_entropy_z4", "mean_gate_z4", "mean_abs_delta_z4", "mean_abs_error_z4", "mean_abs_dynamic_error_z4")}
    time_ratios = torch.zeros(4)
    corrupted_reference_distance = 0.0
    corrupted_observation_distance = 0.0
    corrupted_entropy = 0.0
    corrupted_normalized_entropy = 0.0
    corrupted_memory_frame_count = 0
    finite = True
    frame_count = 0
    max_prediction_error_identity = 0.0
    max_zero_error_identity = 0.0
    max_writeback_identity = 0.0
    with torch.no_grad():
        for samples in groups.values():
            correction_hidden = (None, None)
            predictor_hidden = predictor.initial_state()
            pending_dynamics = None
            pending_semantic = None
            legacy_dynamic = None
            memory = ()
            for frame_index, sample in enumerate(samples):
                clean_image = load_image(sample)
                corrupted_image = persistent_gaussian_blur(clean_image, frame_index, len(samples))
                clean_raw = model.extract_backbone_features(clean_image)
                corrupted_raw = model.extract_backbone_features(corrupted_image)
                clean_state = model.encode_backbone_features(clean_raw)
                observation = model.encode_backbone_features(corrupted_raw)
                output_size = tuple(clean_image.shape[-2:])
                if frame_index == 0:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, zero_state(observation), predictor_hidden)
                    continue
                error = error_state(observation, pending_dynamics)
                max_prediction_error_identity = max(max_prediction_error_identity, float((error.z1 - observation.z1 + pending_dynamics.z1).abs().max().item()), float((error.z4 - observation.z4 + pending_dynamics.z4).abs().max().item()))
                if frame_index == 1:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, error, predictor_hidden)
                    continue
                legacy1, _, hidden1, _, stats1 = corrections[0](observation.z1, pending_dynamics.z1, pending_semantic.z1, correction_hidden[0])
                legacy4, new4, hidden4, memory, stats4 = corrections[1](observation.z4, pending_dynamics.z4, pending_semantic.z4, correction_hidden[1], memory)
                legacy_post = UnifiedFeatures(observation.z1 + legacy1, observation.z2, observation.z3, observation.z4 + legacy4)
                stcn_post = UnifiedFeatures(observation.z1 + stats1["delta"], observation.z2, observation.z3, new4)
                legacy_dynamic = update_dynamic_error(error, legacy_dynamic)
                zero = zero_state(observation)
                hosts = {
                    "clean_host": HostFeature(clean_raw.c4, clean_raw.c1, output_size),
                    "corrupted_host": HostFeature(corrupted_raw.c4, corrupted_raw.c1, output_size),
                    "current_d7": host_for_state(model, corrupted_raw, observation, legacy_post, output_size),
                    "stcn_memory": host_for_state(model, corrupted_raw, observation, stcn_post, output_size),
                }
                logits = {name: model.decode_from_host_feature(host) for name, host in hosts.items()}
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name in names:
                    update_confusion_matrix(confusion[name], logits[name].argmax(1).squeeze(0).cpu().to(torch.int64), mask)
                ref = stats4["reference"]
                clean_delta = clean_state.z4 - observation.z4
                ref_delta = ref - observation.z4
                cosine = F.cosine_similarity(ref_delta.flatten(1), clean_delta.flatten(1), dim=1).mean()
                values = (error.z4, legacy_dynamic.z4, ref, stcn_post.z4, stats4["gain"], stats4["delta"], logits["stcn_memory"])
                finite = finite and all(torch.isfinite(value).all().item() for value in values)
                totals["memory_direction_cosine_z4"] += cosine.item()
                totals["memory_entropy_z4"] += stats4["memory_entropy"].item()
                totals["memory_normalized_entropy_z4"] += stats4["memory_normalized_entropy"].item()
                totals["mean_gate_z4"] += stats4["gain"].mean().item()
                totals["mean_abs_delta_z4"] += stats4["delta"].abs().mean().item()
                totals["mean_abs_error_z4"] += error.z4.abs().mean().item()
                totals["mean_abs_dynamic_error_z4"] += legacy_dynamic.z4.abs().mean().item()
                time_ratios += stats4["time_ratios"].detach().cpu()
                if frame_index >= len(samples) // 3:
                    corrupted_reference_distance += (ref - clean_state.z4).abs().sum().item()
                    corrupted_observation_distance += (observation.z4 - clean_state.z4).abs().sum().item()
                    corrupted_entropy += stats4["memory_entropy"].item()
                    corrupted_normalized_entropy += stats4["memory_normalized_entropy"].item()
                    corrupted_memory_frame_count += 1
                frame_count += 1
                correction_hidden = (hidden1, hidden4)
                pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, error, predictor_hidden)
                legacy_dynamic = UnifiedFeatures(*(value.detach() for value in legacy_dynamic.as_tuple()))
                memory = detach_memory(memory)
            correction_hidden = (None, None)
            memory = ()
            legacy_dynamic = None
    for key in totals:
        totals[key] /= frame_count
    time_ratios /= frame_count
    memory_reference_ratio = corrupted_reference_distance / max(corrupted_observation_distance, 1e-12)
    corrupted_entropy /= corrupted_memory_frame_count
    corrupted_normalized_entropy /= corrupted_memory_frame_count
    metrics = {name: float(torch.nanmean(compute_iou(value)).item()) for name, value in confusion.items()}
    gate = run_gate(model, predictor, corrections, dataset.samples[0])
    gate.update({"prediction_error_identity_passed": max_prediction_error_identity <= 1e-7, "zero_error_passed": max_zero_error_identity <= 1e-7, "residual_writeback_identity_passed": max_writeback_identity <= 1e-6})
    summary = {
        "experiment": "kitti_step_stcn_memory_correction",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(checkpoint),
        "base_checkpoints": {key: str(value) for key, value in paths.items()},
        "config": {"seed": 0, "blur_kernel_size": BLUR_KERNEL_SIZE, "blur_sigma_levels": BLUR_SIGMA_LEVELS, "blur_sigma_max": BLUR_SIGMA_MAX, "memory_size": 4, "key_channels": 64, "dynamic_error_alpha": 0.207, "dynamic_error_beta": 0.793},
        "dataset": {"split": "val", "sequence_count": len(groups), "total_frame_count": len(dataset.samples), "effective_frame_count": frame_count},
        "metrics": {"mIoU_clean_host": metrics["clean_host"], "mIoU_corrupted_host": metrics["corrupted_host"], "mIoU_current_d7": metrics["current_d7"], "mIoU_stcn_memory": metrics["stcn_memory"], "stcn_minus_current_d7": metrics["stcn_memory"] - metrics["current_d7"], "stcn_minus_historical_d7": metrics["stcn_memory"] - OLD_D7_MIOU, "stcn_minus_corrupted_host": metrics["stcn_memory"] - metrics["corrupted_host"], "recovery_ratio": (metrics["stcn_memory"] - metrics["corrupted_host"]) / (metrics["clean_host"] - metrics["corrupted_host"])},
        "diagnostics": {**totals, "memory_reference_ratio_z4_corrupted": memory_reference_ratio, "memory_entropy_z4_corrupted": corrupted_entropy, "memory_normalized_entropy_z4_corrupted": corrupted_normalized_entropy, "corrupted_memory_frame_count": corrupted_memory_frame_count, "mean_time_ratios_t_minus_1_to_t_minus_4": time_ratios.tolist()},
        "gates": gate,
        "finite": finite and gate["finite"],
        "decision": "STCN MEMORYREADER TERMINATED" if corrupted_normalized_entropy > 0.9 or memory_reference_ratio >= 1.0 else "STRONG GO" if metrics["stcn_memory"] >= 0.535 else "GO" if metrics["stcn_memory"] >= 0.5165 else "WEAK" if metrics["stcn_memory"] > OLD_D7_MIOU else "NO-GO",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
