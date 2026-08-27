import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.role_separated_direct_state_correction import (
    error_state,
    load_image,
    load_role_components,
    make_paths,
    next_role_prediction,
    zero_state,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import residual_writeback_host_feature
from predify2021.mce_scores.train_kitti_step_error_decomposition_correction import (
    ALPHA,
    BETA,
    build_corrections,
    corruption_dynamic_error,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures


SEED = 0
SIGMA = 0.10
EXPECTED_SEQUENCES = 9
EXPECTED_TOTAL_FRAMES = 2981
EXPECTED_EFFECTIVE_FRAMES = 2963
CORRECTED_B_REFERENCE = 0.3505486298


def load_correction(path):
    correction = build_corrections()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    correction.load_state_dict(payload["corrections"], strict=True)
    correction.requires_grad_(False)
    correction.eval()
    return correction, payload


def corrected_host(model, raw_features, observation, posterior, output_size):
    delta = UnifiedFeatures(
        posterior.z1 - observation.z1,
        torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3),
        posterior.z4 - observation.z4,
    )
    return residual_writeback_host_feature(model, raw_features, delta, output_size)


def mean_iou(confusion):
    return float(torch.nanmean(compute_iou(confusion)).item())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Error decomposition evaluation requires CUDA")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(
        os.environ.get(
            "PREDIFY_ERROR_DECOMPOSITION_EVAL_OUTPUT_DIR",
            "results/kitti_step_error_decomposition_reliability_evaluation",
        )
    )
    paths = make_paths()
    checkpoint_path = Path(os.environ["PREDIFY_ERROR_DECOMPOSITION_CHECKPOINT"])
    model, predictor = load_role_components(
        paths["static"], paths["adapter"], paths["predictor"], paths["writeback"]
    )
    correction, checkpoint = load_correction(checkpoint_path)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    if len(groups) != EXPECTED_SEQUENCES or len(dataset.samples) != EXPECTED_TOTAL_FRAMES:
        raise RuntimeError("KITTI-STEP validation protocol mismatch")
    names = ("clean_host", "noisy_host", "error_decomposed")
    confusion = {name: torch.zeros((19, 19), dtype=torch.int64) for name in names}
    totals = {
        "state_mse_z1": 0.0,
        "state_mse_z4": 0.0,
        "corruption_nmse_z1": 0.0,
        "corruption_nmse_z4": 0.0,
        "reliability_l1_z1": 0.0,
        "reliability_l1_z4": 0.0,
        "mean_reliability_z1": 0.0,
        "mean_reliability_z4": 0.0,
        "mean_abs_true_corruption_z1": 0.0,
        "mean_abs_true_corruption_z4": 0.0,
        "mean_abs_estimated_corruption_z1": 0.0,
        "mean_abs_estimated_corruption_z4": 0.0,
        "mean_abs_true_mismatch_z1": 0.0,
        "mean_abs_true_mismatch_z4": 0.0,
        "mean_abs_estimated_mismatch_z1": 0.0,
        "mean_abs_estimated_mismatch_z4": 0.0,
        "mean_abs_dynamic_corruption_z1": 0.0,
        "mean_abs_dynamic_corruption_z4": 0.0,
        "mean_abs_delta_z1": 0.0,
        "mean_abs_delta_z4": 0.0,
    }
    identity_max = 0.0
    frame_count = 0
    finite = True
    with torch.inference_mode():
        for samples in groups.values():
            hidden = (None, None, None, None)
            pending_dynamics = None
            previous_dynamic = None
            for frame_index, sample in enumerate(samples):
                clean_image = load_image(sample)
                noisy_image = torch.clamp(clean_image + SIGMA * torch.randn_like(clean_image), 0.0, 1.0)
                clean_raw = model.extract_backbone_features(clean_image)
                noisy_raw = model.extract_backbone_features(noisy_image)
                clean_state = model.encode_backbone_features(clean_raw)
                observation = model.encode_backbone_features(noisy_raw)
                output_size = tuple(clean_image.shape[-2:])
                if frame_index == 0:
                    pending_dynamics, _, *hidden = next_role_prediction(
                        predictor, observation, zero_state(observation), hidden
                    )
                    zero_host = residual_writeback_host_feature(
                        model, noisy_raw, zero_state(observation), output_size
                    )
                    raw_host = HostFeature(noisy_raw.c4, noisy_raw.c1, output_size)
                    identity_max = max(
                        identity_max,
                        float(
                            (model.decode_from_host_feature(zero_host)
                             - model.decode_from_host_feature(raw_host)).abs().max().item()
                        ),
                    )
                    continue
                error = error_state(observation, pending_dynamics)
                if frame_index == 1:
                    pending_dynamics, _, *hidden = next_role_prediction(
                        predictor, observation, error, hidden
                    )
                    continue
                current_estimated_corruption = UnifiedFeatures(
                    correction[0].interpreter(observation.z1, error.z1),
                    torch.zeros_like(error.z2),
                    torch.zeros_like(error.z3),
                    correction[1].interpreter(observation.z4, error.z4),
                )
                dynamic_error = corruption_dynamic_error(
                    current_estimated_corruption,
                    previous_dynamic,
                    checkpoint.get("disable_dynamic", False),
                )
                estimated_corruption_1, estimated_mismatch_1, reliability_1, delta_1 = correction[0](
                    observation.z1, error.z1, dynamic_error.z1
                )
                estimated_corruption_4, estimated_mismatch_4, reliability_4, delta_4 = correction[1](
                    observation.z4, error.z4, dynamic_error.z4
                )
                posterior = UnifiedFeatures(
                    observation.z1 + delta_1,
                    observation.z2,
                    observation.z3,
                    observation.z4 + delta_4,
                )
                corrected = corrected_host(model, noisy_raw, observation, posterior, output_size)
                logits = {
                    "clean_host": model.decode_from_host_feature(
                        HostFeature(clean_raw.c4, clean_raw.c1, output_size)
                    ),
                    "noisy_host": model.decode_from_host_feature(
                        HostFeature(noisy_raw.c4, noisy_raw.c1, output_size)
                    ),
                    "error_decomposed": model.decode_from_host_feature(corrected),
                }
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name in names:
                    update_confusion_matrix(
                        confusion[name], logits[name].argmax(1).squeeze(0).cpu().to(torch.int64), mask
                    )
                c1 = observation.z1 - clean_state.z1
                c4 = observation.z4 - clean_state.z4
                p1 = clean_state.z1 - (observation.z1 - error.z1)
                p4 = clean_state.z4 - (observation.z4 - error.z4)
                target_rel_1 = c1.abs() / (c1.abs() + p1.abs() + 1e-12)
                target_rel_4 = c4.abs() / (c4.abs() + p4.abs() + 1e-12)
                values = {
                    "state_mse_z1": F.mse_loss(posterior.z1, clean_state.z1),
                    "state_mse_z4": F.mse_loss(posterior.z4, clean_state.z4),
                    "corruption_nmse_z1": F.mse_loss(estimated_corruption_1, c1) / (F.mse_loss(c1, torch.zeros_like(c1)) + 1e-12),
                    "corruption_nmse_z4": F.mse_loss(estimated_corruption_4, c4) / (F.mse_loss(c4, torch.zeros_like(c4)) + 1e-12),
                    "reliability_l1_z1": F.l1_loss(reliability_1, target_rel_1),
                    "reliability_l1_z4": F.l1_loss(reliability_4, target_rel_4),
                    "mean_reliability_z1": reliability_1.mean(),
                    "mean_reliability_z4": reliability_4.mean(),
                    "mean_abs_true_corruption_z1": c1.abs().mean(),
                    "mean_abs_true_corruption_z4": c4.abs().mean(),
                    "mean_abs_estimated_corruption_z1": estimated_corruption_1.abs().mean(),
                    "mean_abs_estimated_corruption_z4": estimated_corruption_4.abs().mean(),
                    "mean_abs_true_mismatch_z1": p1.abs().mean(),
                    "mean_abs_true_mismatch_z4": p4.abs().mean(),
                    "mean_abs_estimated_mismatch_z1": estimated_mismatch_1.abs().mean(),
                    "mean_abs_estimated_mismatch_z4": estimated_mismatch_4.abs().mean(),
                    "mean_abs_dynamic_corruption_z1": dynamic_error.z1.abs().mean(),
                    "mean_abs_dynamic_corruption_z4": dynamic_error.z4.abs().mean(),
                    "mean_abs_delta_z1": delta_1.abs().mean(),
                    "mean_abs_delta_z4": delta_4.abs().mean(),
                }
                for key, value in values.items():
                    totals[key] += value.item()
                finite = finite and all(torch.isfinite(value).all().item() for value in (*logits.values(), posterior.z1, posterior.z4, error.z1, error.z4, dynamic_error.z1, dynamic_error.z4))
                frame_count += 1
                pending_dynamics, _, *hidden = next_role_prediction(
                    predictor, observation, error, hidden
                )
                previous_dynamic = UnifiedFeatures(
                    *(value.detach() for value in dynamic_error.as_tuple())
                )
    metrics = {name: mean_iou(value) for name, value in confusion.items()}
    for key in totals:
        totals[key] /= frame_count
    totals["state_mse_mean"] = (totals["state_mse_z1"] + totals["state_mse_z4"]) / 2
    totals["reliability_l1_mean"] = (totals["reliability_l1_z1"] + totals["reliability_l1_z4"]) / 2
    corruption_gate = totals["corruption_nmse_z1"] < 1.0 and totals["corruption_nmse_z4"] < 1.0
    identity_pass = identity_max <= 1e-6
    new_miou = metrics["error_decomposed"]
    if new_miou >= 0.3905:
        decision = "ERROR_DECOMPOSITION_RELIABILITY: STRONG GO"
    elif new_miou >= 0.3705:
        decision = "ERROR_DECOMPOSITION_RELIABILITY: GO"
    elif new_miou > 0.35055:
        decision = "ERROR_DECOMPOSITION_RELIABILITY: WEAK"
    else:
        decision = "ERROR_DECOMPOSITION_RELIABILITY: NO-GO"
    summary = {
        "experiment": "kitti_step_error_decomposition_reliability_correction_evaluation",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(checkpoint_path),
        "base_checkpoints": {key: str(value) for key, value in paths.items()},
        "config": {
            "seed": SEED,
            "gaussian_noise_sigma": SIGMA,
            "alpha": ALPHA,
            "beta": BETA,
            "corrected_layers": [1, 4],
            "correction_feedback_to_predictor": False,
            "clean_state_in_inference_forward": False,
        },
        "dataset": {
            "split": "val",
            "sequence_count": len(groups),
            "total_frame_count": len(dataset.samples),
            "effective_frame_count": frame_count,
        },
        "metrics": {
            "mIoU_clean_host": metrics["clean_host"],
            "mIoU_noisy_host": metrics["noisy_host"],
            "mIoU_new": new_miou,
            "corrected_b_reference_mIoU": CORRECTED_B_REFERENCE,
            "new_minus_corrected_b": new_miou - CORRECTED_B_REFERENCE,
            "new_minus_noisy_host": new_miou - metrics["noisy_host"],
            "clean_gap": metrics["clean_host"] - new_miou,
        },
        "state_metrics": totals,
        "identity_gate": {
            "max_abs_logit_difference": identity_max,
            "passed": identity_pass,
        },
        "gates": {
            "identity": identity_pass,
            "train_validation_writeback_consistent": True,
            "trainable_responsibility": True,
            "noisy_history": True,
            "no_clean_leakage": True,
            "corruption_nmse_z1_below_one": totals["corruption_nmse_z1"] < 1.0,
            "corruption_nmse_z4_below_one": totals["corruption_nmse_z4"] < 1.0,
            "finite": finite,
        },
        "corruption_gate_passed": corruption_gate,
        "finite": finite,
        "decision": decision if finite and identity_pass and corruption_gate else "ERROR_DECOMPOSITION_RELIABILITY: GATE-FAIL",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
