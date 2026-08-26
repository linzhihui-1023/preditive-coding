import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, predict_current, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_error_correction import corrected_host_feature
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import load_writeback_checkpoint
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.train_kitti_step_fixed_adapter_predictor import ADAPTER_CHECKPOINT_DEFAULT, STATIC_CHECKPOINT_DEFAULT
from predify2021.mce_scores.train_kitti_step_state_predictor import load_static_kitti_checkpoint
from predify2021.mce_scores.video_metrics import VideoConsistency, weighted_iou
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, MultiLayerPredictor, SemanticRecurrentPredictor, UnifiedFeatures, build_deeplabv3plus_resnet50_host

NAMES = ("persistence", "constant_velocity", "old_cnn", "new_predictor", "oracle_current")
PREDICTOR = "/home/lin/predify/experiments/kitti_step_fixed_adapter_predictor_41aa5cc/best_predictor.pt"
WRITEBACK = "/home/lin/predify/experiments/kitti_step_host_conditioned_writeback/host_conditioned_writeback_epoch3.pt"


def decode_state(model, state, output_size):
    decoded = model.decode_adapter_deltas(state)
    return model.decode_from_host_feature(HostFeature(decoded.c4, decoded.c1, output_size))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Semantic recurrent predictor evaluation requires CUDA")
    random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    checkpoint = Path(os.environ["PREDIFY_SEMANTIC_RECURRENT_CHECKPOINT"])
    output = Path(os.environ.get("PREDIFY_SEMANTIC_RECURRENT_EVAL_OUTPUT_DIR", "results/kitti_step_semantic_recurrent_predictor"))
    model = build_deeplabv3plus_resnet50_host().cuda(); load_static_kitti_checkpoint(model, Path(STATIC_CHECKPOINT_DEFAULT))
    adapter = torch.load(ADAPTER_CHECKPOINT_DEFAULT, map_location="cpu", weights_only=False); model.multi_layer_adapter.load_state_dict(adapter["adapter_state_dict"], strict=True); load_writeback_checkpoint(model, Path(WRITEBACK)); model.requires_grad_(False); model.eval()
    old = MultiLayerPredictor().cuda(); old.load_state_dict(torch.load(PREDICTOR, map_location="cpu", weights_only=False)["predictor_state_dict"], strict=True); old.requires_grad_(False); old.eval()
    new = SemanticRecurrentPredictor().cuda(); new.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["predictor_state_dict"], strict=True); new.requires_grad_(False); new.eval()
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val"); groups = sequence_groups(dataset)
    sequence_limit = int(os.environ.get("PREDIFY_SEMANTIC_RECURRENT_EVAL_SEQUENCE_LIMIT", "0"))
    if sequence_limit:
        groups = dict(list(groups.items())[:sequence_limit])
    compare_context = os.environ.get("PREDIFY_SEMANTIC_RECURRENT_Z4_CONTEXT_ONLY") == "1"
    names = NAMES + ("z4_context_only",) if compare_context else NAMES
    confusion = {name: torch.zeros((19, 19), dtype=torch.int64) for name in names}; vc = VideoConsistency(names); state_mse = {name: {"z1": 0.0, "z4": 0.0} for name in names if name != "oracle_current"}; count = 0
    context = None
    if compare_context:
        context = SemanticRecurrentPredictor().cuda(); context.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["predictor_state_dict"], strict=True); context.requires_grad_(False); context.eval()
    with torch.inference_mode():
        for samples in groups.values():
            vc.reset_sequence(); previous_previous = previous = None; hidden4 = hidden1 = None; context_hidden4 = context_hidden1 = None; new_error = None; context_error = None; pending = None; context_pending = None
            for index, sample in enumerate(samples):
                image = load_image(sample); raw = model.extract_backbone_features(image); current = model.encode_backbone_features(raw); size = tuple(image.shape[-2:])
                if previous is None:
                    previous = current; zero = UnifiedFeatures(*(torch.zeros_like(v) for v in current.as_tuple())); pending, hidden4, hidden1 = new.step(current, zero, hidden4, hidden1)
                    if compare_context:
                        context_pending, context_hidden4, context_hidden1 = context.step(current, zero, context_hidden4, context_hidden1, persist_z4=True)
                    continue
                if previous_previous is None:
                    previous_previous, previous = previous, current; old_pred, _ = predict_current(old, previous_previous, previous, current); new_error = UnifiedFeatures(*(a - b for a, b in zip(current.as_tuple(), pending.as_tuple()))); pending, hidden4, hidden1 = new.step(current, new_error, hidden4, hidden1)
                    if compare_context:
                        context_error = UnifiedFeatures(*(a - b for a, b in zip(current.as_tuple(), context_pending.as_tuple()))); context_pending, context_hidden4, context_hidden1 = context.step(current, context_error, context_hidden4, context_hidden1, persist_z4=True)
                    continue
                old_pred, _ = predict_current(old, previous_previous, previous, current)
                new_pred = pending
                persistence = previous; velocity = UnifiedFeatures(*(2 * a - b for a, b in zip(previous.as_tuple(), previous_previous.as_tuple())))
                targets = {"persistence": persistence, "constant_velocity": velocity, "old_cnn": old_pred, "new_predictor": new_pred, "oracle_current": current}
                if compare_context:
                    targets["z4_context_only"] = context_pending
                predictions = {}
                for name, target in targets.items():
                    predictions[name] = decode_state(model, target, size).argmax(1).squeeze(0).cpu()
                    if name != "oracle_current":
                        state_mse[name]["z1"] += F.mse_loss(target.z1, current.z1).item(); state_mse[name]["z4"] += F.mse_loss(target.z4, current.z4).item()
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name in names: update_confusion_matrix(confusion[name], predictions[name], mask)
                vc.append(mask, predictions); new_error = UnifiedFeatures(*(a - b for a, b in zip(current.as_tuple(), new_pred.as_tuple()))); pending, hidden4, hidden1 = new.step(current, new_error, hidden4, hidden1)
                if compare_context:
                    context_error = UnifiedFeatures(*(a - b for a, b in zip(current.as_tuple(), context_pending.as_tuple()))); context_pending, context_hidden4, context_hidden1 = context.step(current, context_error, context_hidden4, context_hidden1, persist_z4=True)
                previous_previous, previous = previous, current; count += 1
    mvc = vc.means(); metrics = {name: {"miou": float(torch.nanmean(compute_iou(confusion[name])).item()), "wiou": weighted_iou(confusion[name]), "mvc8": mvc[8][name], "mvc16": mvc[16][name]} for name in NAMES}
    for name in state_mse: state_mse[name]["mean"] = (state_mse[name]["z1"] + state_mse[name]["z4"]) / (2 * count)
    persistence_mean = state_mse["persistence"]["mean"]; constant_mean = state_mse["constant_velocity"]["mean"]; new_mean = state_mse["new_predictor"]["mean"]
    for name in state_mse: state_mse[name]["z1"] /= count; state_mse[name]["z4"] /= count
    for name in state_mse: state_mse[name]["nmse"] = state_mse[name]["mean"] / persistence_mean
    strong = max(metrics["constant_velocity"]["miou"], metrics["old_cnn"]["miou"])
    dynamics = "GO" if new_mean < 0.9 * constant_mean else "NO-GO"; semantic = "GO" if metrics["new_predictor"]["miou"] >= strong + 0.01 else "NO-GO"
    summary = {"protocol":{"split":"val","sequence_count":len(groups),"total_frame_count":len(dataset.samples),"evaluated_frame_count":count},"metrics":metrics,"state_mse":state_mse,"mvc_window_counts":vc.window_counts(),"oracle_current_state_decode_miou":metrics["oracle_current"]["miou"],"new_vs_constant_velocity_mse_improvement":1-new_mean/constant_mean,"new_vs_strongest_miou_improvement":metrics["new_predictor"]["miou"]-strong,"DYNAMICS":dynamics,"SEMANTIC_PREDICTION":semantic,"NEW_PREDICTOR":"GO" if dynamics=="GO" and semantic=="GO" else "NO-GO","parameters_updated":False}
    output.mkdir(parents=True, exist_ok=True); (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__": main()
