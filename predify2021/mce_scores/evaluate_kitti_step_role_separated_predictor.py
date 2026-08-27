import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, predict_current, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_semantic_recurrent_predictor import decode_state
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import load_writeback_checkpoint
from predify2021.mce_scores.train_kitti_step_fixed_adapter_predictor import ADAPTER_CHECKPOINT_DEFAULT, STATIC_CHECKPOINT_DEFAULT
from predify2021.mce_scores.train_kitti_step_state_predictor import load_static_kitti_checkpoint
from predify2021.mce_scores.video_metrics import VideoConsistency, weighted_iou
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    MultiLayerPredictor,
    RoleSeparatedRecurrentPredictor,
    SemanticRecurrentPredictor,
    UnifiedFeatures,
    build_deeplabv3plus_resnet50_host,
)

OLD_CHECKPOINT = "/home/lin/predify/experiments/kitti_step_fixed_adapter_predictor_41aa5cc/best_predictor.pt"
BALANCED_CHECKPOINT = "/home/lin/predify/experiments/kitti_step_balanced_semantic_recurrent_predictor_b0dcb11/best_semantic_recurrent_predictor.pt"
ISOLATED_CHECKPOINT = "/home/lin/predify/experiments/kitti_step_gradient_isolated_predictor_a2b428a/best_semantic_recurrent_predictor.pt"
WRITEBACK_CHECKPOINT = "/home/lin/predify/experiments/kitti_step_host_conditioned_writeback/host_conditioned_writeback_epoch3.pt"
NAMES = ("persistence", "old_cnn", "balanced_new", "gradient_isolated", "role_separated")


def load_predictor(cls, path):
    predictor = cls().cuda()
    predictor.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["predictor_state_dict"], strict=True)
    predictor.requires_grad_(False)
    predictor.eval()
    return predictor


def error_state(target, prediction):
    return UnifiedFeatures(*(value.detach() for value in (
        target.z1 - prediction.z1,
        target.z2 - prediction.z2,
        target.z3 - prediction.z3,
        target.z4 - prediction.z4,
    )))


def zero_state(state):
    return UnifiedFeatures(*(torch.zeros_like(value) for value in state.as_tuple()))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Role-separated predictor evaluation requires CUDA")
    random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    checkpoint = Path(os.environ["PREDIFY_ROLE_SEPARATED_CHECKPOINT"])
    output = Path(os.environ.get("PREDIFY_ROLE_SEPARATED_EVAL_OUTPUT_DIR", "results/kitti_step_role_separated_predictor"))
    sequence_limit = int(os.environ.get("PREDIFY_ROLE_SEPARATED_SEQUENCE_LIMIT", "2"))
    model = build_deeplabv3plus_resnet50_host().cuda()
    load_static_kitti_checkpoint(model, Path(STATIC_CHECKPOINT_DEFAULT))
    adapter = torch.load(ADAPTER_CHECKPOINT_DEFAULT, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.load_state_dict(adapter["adapter_state_dict"], strict=True)
    load_writeback_checkpoint(model, Path(WRITEBACK_CHECKPOINT))
    model.requires_grad_(False); model.eval()
    old = load_predictor(MultiLayerPredictor, OLD_CHECKPOINT)
    balanced = load_predictor(SemanticRecurrentPredictor, BALANCED_CHECKPOINT)
    isolated = load_predictor(SemanticRecurrentPredictor, ISOLATED_CHECKPOINT)
    role_separated = load_predictor(RoleSeparatedRecurrentPredictor, checkpoint)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    if sequence_limit:
        groups = dict(list(groups.items())[:sequence_limit])
    confusion = {name: torch.zeros((19, 19), dtype=torch.int64) for name in NAMES}
    state_mse = {name: {"z1": 0.0, "z4": 0.0} for name in NAMES}
    vc = VideoConsistency(NAMES)
    count = 0
    with torch.inference_mode():
        for samples in groups.values():
            vc.reset_sequence()
            previous_previous = previous = None
            balanced_h4 = balanced_h1 = None
            isolated_h4 = isolated_h1 = None
            role_h4_dyn = role_h4_sem = role_h1_dyn = role_h1_sem = None
            balanced_pending = isolated_pending = None
            role_dynamics_pending = role_diagnostic_pending = None
            for sample in samples:
                image = load_image(sample)
                raw = model.extract_backbone_features(image)
                current = model.encode_backbone_features(raw)
                size = tuple(image.shape[-2:])
                if previous is None:
                    previous = current
                    zero = zero_state(current)
                    balanced_pending, balanced_h4, balanced_h1 = balanced.step(current, zero, balanced_h4, balanced_h1)
                    isolated_pending, isolated_h4, isolated_h1 = isolated.step(current, zero, isolated_h4, isolated_h1, detach_high_to_low=True)
                    role_dynamics_pending, role_diagnostic_pending, role_h4_dyn, role_h4_sem, role_h1_dyn, role_h1_sem = role_separated.step(current, zero, role_h4_dyn, role_h4_sem, role_h1_dyn, role_h1_sem)
                    continue
                if previous_previous is None:
                    previous_previous, previous = previous, current
                    balanced_pending, balanced_h4, balanced_h1 = balanced.step(current, error_state(current, balanced_pending), balanced_h4, balanced_h1)
                    isolated_pending, isolated_h4, isolated_h1 = isolated.step(current, error_state(current, isolated_pending), isolated_h4, isolated_h1, detach_high_to_low=True)
                    role_dynamics_pending, role_diagnostic_pending, role_h4_dyn, role_h4_sem, role_h1_dyn, role_h1_sem = role_separated.step(current, error_state(current, role_dynamics_pending), role_h4_dyn, role_h4_sem, role_h1_dyn, role_h1_sem)
                    continue
                targets = {
                    "persistence": previous,
                    "old_cnn": predict_current(old, previous_previous, previous, current)[0],
                    "balanced_new": balanced_pending,
                    "gradient_isolated": isolated_pending,
                    "role_separated": role_dynamics_pending,
                }
                semantic_states = dict(targets)
                semantic_states["role_separated"] = role_diagnostic_pending
                predictions = {}
                for name in NAMES:
                    predictions[name] = decode_state(model, semantic_states[name], size).argmax(1).squeeze(0).cpu()
                    state_mse[name]["z1"] += F.mse_loss(targets[name].z1, current.z1).item()
                    state_mse[name]["z4"] += F.mse_loss(targets[name].z4, current.z4).item()
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name in NAMES:
                    update_confusion_matrix(confusion[name], predictions[name], mask)
                vc.append(mask, predictions)
                balanced_pending, balanced_h4, balanced_h1 = balanced.step(current, error_state(current, balanced_pending), balanced_h4, balanced_h1)
                isolated_pending, isolated_h4, isolated_h1 = isolated.step(current, error_state(current, isolated_pending), isolated_h4, isolated_h1, detach_high_to_low=True)
                role_dynamics_pending, role_diagnostic_pending, role_h4_dyn, role_h4_sem, role_h1_dyn, role_h1_sem = role_separated.step(current, error_state(current, role_dynamics_pending), role_h4_dyn, role_h4_sem, role_h1_dyn, role_h1_sem)
                previous_previous, previous = previous, current
                count += 1
    for name in NAMES:
        state_mse[name]["z1"] /= count
        state_mse[name]["z4"] /= count
        state_mse[name]["mean"] = (state_mse[name]["z1"] + state_mse[name]["z4"]) / 2
    persistence = state_mse["persistence"]
    for name in NAMES:
        state_mse[name]["mse_ratio_vs_persistence"] = {
            "z1": state_mse[name]["z1"] / persistence["z1"],
            "z4": state_mse[name]["z4"] / persistence["z4"],
            "mean": state_mse[name]["mean"] / persistence["mean"],
        }
    mvc = vc.means()
    metrics = {name: {"predicted_state_miou": float(torch.nanmean(compute_iou(confusion[name])).item()), "wiou": weighted_iou(confusion[name]), "mvc8": mvc[8][name], "mvc16": mvc[16][name]} for name in NAMES}
    role = state_mse["role_separated"]
    dynamics_go = role["mse_ratio_vs_persistence"]["mean"] < 1.0 and role["z1"] < persistence["z1"] and role["z4"] <= persistence["z4"]
    semantic_go = metrics["role_separated"]["predicted_state_miou"] >= 0.24
    if dynamics_go and semantic_go:
        decision = "DYNAMICS_SEMANTIC_ROLE_SEPARATION: GO"
    elif dynamics_go:
        decision = "DYNAMICS_ROLE: GO; SEMANTIC_CONTEXT_ROLE: NO-GO"
    else:
        decision = "DYNAMICS_SEMANTIC_ROLE_SEPARATION: NO-GO"
    summary = {
        "checkpoint": str(checkpoint),
        "parameters_updated": False,
        "protocol": {"split": "val", "sequence_names": list(groups), "sequence_count": len(groups), "total_frame_count": len(dataset.samples), "evaluated_frame_count": count},
        "state_mse": state_mse,
        "semantic_diagnostics": metrics,
        "mvc_window_counts": vc.window_counts(),
        "decision": decision,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
