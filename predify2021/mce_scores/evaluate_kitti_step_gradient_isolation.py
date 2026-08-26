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
from predify2021.mce_scores.train_kitti_step_fixed_adapter_predictor import ADAPTER_CHECKPOINT_DEFAULT, STATIC_CHECKPOINT_DEFAULT
from predify2021.mce_scores.train_kitti_step_state_predictor import load_static_kitti_checkpoint
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import load_writeback_checkpoint
from predify2021.mce_scores.video_metrics import VideoConsistency, weighted_iou
from predify2021.model_factory.deeplabv3plus_resnet50 import SemanticRecurrentPredictor, UnifiedFeatures, build_deeplabv3plus_resnet50_host

OLD_CHECKPOINT = "/home/lin/predify/experiments/kitti_step_fixed_adapter_predictor_41aa5cc/best_predictor.pt"
BALANCED_CHECKPOINT = "/home/lin/predify/experiments/kitti_step_balanced_semantic_recurrent_predictor_b0dcb11/best_semantic_recurrent_predictor.pt"
NAMES = ("persistence", "old_cnn", "balanced_new", "gradient_isolated")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Gradient-isolated predictor evaluation requires CUDA")
    random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    isolated_checkpoint = Path(os.environ["PREDIFY_GRADIENT_ISOLATED_CHECKPOINT"])
    output = Path(os.environ.get("PREDIFY_GRADIENT_ISOLATED_EVAL_OUTPUT_DIR", "results/kitti_step_gradient_isolation"))
    model = build_deeplabv3plus_resnet50_host().cuda(); load_static_kitti_checkpoint(model, Path(STATIC_CHECKPOINT_DEFAULT))
    adapter = torch.load(ADAPTER_CHECKPOINT_DEFAULT, map_location="cpu", weights_only=False); model.multi_layer_adapter.load_state_dict(adapter["adapter_state_dict"], strict=True); load_writeback_checkpoint(model, Path("/home/lin/predify/experiments/kitti_step_host_conditioned_writeback/host_conditioned_writeback_epoch3.pt")); model.requires_grad_(False); model.eval()
    old = torch.load(OLD_CHECKPOINT, map_location="cpu", weights_only=False)
    balanced = SemanticRecurrentPredictor().cuda(); balanced.load_state_dict(torch.load(BALANCED_CHECKPOINT, map_location="cpu", weights_only=False)["predictor_state_dict"], strict=True); balanced.requires_grad_(False); balanced.eval()
    isolated = SemanticRecurrentPredictor().cuda(); isolated.load_state_dict(torch.load(isolated_checkpoint, map_location="cpu", weights_only=False)["predictor_state_dict"], strict=True); isolated.requires_grad_(False); isolated.eval()
    from predify2021.model_factory.deeplabv3plus_resnet50 import MultiLayerPredictor
    old_model = MultiLayerPredictor().cuda(); old_model.load_state_dict(old["predictor_state_dict"], strict=True); old_model.requires_grad_(False); old_model.eval()
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val"); groups = sequence_groups(dataset)
    limit = int(os.environ.get("PREDIFY_GRADIENT_ISOLATED_SEQUENCE_LIMIT", "2")); groups = dict(list(groups.items())[:limit])
    confusion = {name: torch.zeros((19, 19), dtype=torch.int64) for name in NAMES}; vc = VideoConsistency(NAMES); state_mse = {name: {"z1": 0.0, "z4": 0.0} for name in NAMES}; count = 0
    with torch.inference_mode():
        for samples in groups.values():
            vc.reset_sequence(); previous_previous = previous = None; balanced_h4 = balanced_h1 = None; isolated_h4 = isolated_h1 = None; balanced_pending = isolated_pending = None
            for sample_index, sample in enumerate(samples):
                image = load_image(sample); raw = model.extract_backbone_features(image); current = model.encode_backbone_features(raw); size = tuple(image.shape[-2:])
                if previous is None:
                    previous = current; zero = UnifiedFeatures(*(torch.zeros_like(v) for v in current.as_tuple())); balanced_pending, balanced_h4, balanced_h1 = balanced.step(current, zero, balanced_h4, balanced_h1); isolated_pending, isolated_h4, isolated_h1 = isolated.step(current, zero, isolated_h4, isolated_h1, detach_high_to_low=True); continue
                if previous_previous is None:
                    previous_previous, previous = previous, current; balanced_error = UnifiedFeatures(*(a - b for a, b in zip(current.as_tuple(), balanced_pending.as_tuple()))); isolated_error = UnifiedFeatures(*(a - b for a, b in zip(current.as_tuple(), isolated_pending.as_tuple()))); balanced_pending, balanced_h4, balanced_h1 = balanced.step(current, balanced_error, balanced_h4, balanced_h1); isolated_pending, isolated_h4, isolated_h1 = isolated.step(current, isolated_error, isolated_h4, isolated_h1, detach_high_to_low=True); continue
                persistence = previous; old_prediction = predict_current(old_model, previous_previous, previous, current)[0]; balanced_prediction = balanced_pending; isolated_prediction = isolated_pending
                targets = {"persistence": persistence, "old_cnn": old_prediction, "balanced_new": balanced_prediction, "gradient_isolated": isolated_prediction}
                predictions = {}
                for name, target in targets.items():
                    predictions[name] = decode_state(model, target, size).argmax(1).squeeze(0).cpu(); state_mse[name]["z1"] += F.mse_loss(target.z1, current.z1).item(); state_mse[name]["z4"] += F.mse_loss(target.z4, current.z4).item()
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name in NAMES: update_confusion_matrix(confusion[name], predictions[name], mask)
                vc.append(mask, predictions)
                balanced_error = UnifiedFeatures(*(a - b for a, b in zip(current.as_tuple(), balanced_prediction.as_tuple()))); isolated_error = UnifiedFeatures(*(a - b for a, b in zip(current.as_tuple(), isolated_prediction.as_tuple()))); balanced_pending, balanced_h4, balanced_h1 = balanced.step(current, balanced_error, balanced_h4, balanced_h1); isolated_pending, isolated_h4, isolated_h1 = isolated.step(current, isolated_error, isolated_h4, isolated_h1, detach_high_to_low=True); previous_previous, previous = previous, current; count += 1
    mvc = vc.means(); metrics = {name: {"miou": float(torch.nanmean(compute_iou(confusion[name])).item()), "wiou": weighted_iou(confusion[name]), "mvc8": mvc[8][name], "mvc16": mvc[16][name]} for name in NAMES}
    for name in NAMES: state_mse[name]["z1"] /= count; state_mse[name]["z4"] /= count; state_mse[name]["mean"] = (state_mse[name]["z1"] + state_mse[name]["z4"]) / 2
    persistence_mean = state_mse["persistence"]["mean"]
    for name in NAMES: state_mse[name]["nmse"] = state_mse[name]["mean"] / persistence_mean
    summary = {"protocol": {"split": "val", "sequence_names": list(groups), "sequence_count": len(groups), "total_frame_count": len(dataset.samples), "evaluated_frame_count": count}, "checkpoint": str(isolated_checkpoint), "parameters_updated": False, "metrics": metrics, "state_mse": state_mse, "mvc_window_counts": vc.window_counts(), "gradient_isolation": {"gradient_high_to_low_detached": True}}
    output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__": main()
