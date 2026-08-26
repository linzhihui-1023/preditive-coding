import json
import os
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
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, MultiLayerPredictor, build_deeplabv3plus_resnet50_host

PREDICTOR = "/home/lin/predify/experiments/kitti_step_fixed_adapter_predictor_41aa5cc/best_predictor.pt"
WRITEBACK = "/home/lin/predify/experiments/kitti_step_host_conditioned_writeback/host_conditioned_writeback_epoch3.pt"
NAMES = ("static", "persistence", "predictor")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Line A requires CUDA")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(os.environ.get("PREDIFY_LINE_A_OUTPUT_DIR", "results/kitti_step_line_a"))
    model = build_deeplabv3plus_resnet50_host().cuda()
    load_static_kitti_checkpoint(model, Path(STATIC_CHECKPOINT_DEFAULT))
    adapter = torch.load(ADAPTER_CHECKPOINT_DEFAULT, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.load_state_dict(adapter["adapter_state_dict"], strict=True)
    load_writeback_checkpoint(model, Path(WRITEBACK))
    predictor = MultiLayerPredictor().cuda()
    predictor.load_state_dict(torch.load(PREDICTOR, map_location="cpu", weights_only=False)["predictor_state_dict"], strict=True)
    model.eval(); predictor.eval()
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    confusion = {name: torch.zeros((19, 19), dtype=torch.int64) for name in NAMES}
    vc = VideoConsistency(NAMES)
    sums = {name: {"z1": 0.0, "z4": 0.0} for name in ("persistence", "predictor")}
    cosine = {"z1": 0.0, "z4": 0.0}; evaluated = 0
    with torch.inference_mode():
        for samples in groups.values():
            vc.reset_sequence(); previous_previous = previous = None
            for sample in samples:
                image = load_image(sample)
                raw = model.extract_backbone_features(image)
                current = model.encode_backbone_features(raw)
                if previous is None:
                    previous = current; continue
                if previous_previous is None:
                    previous_previous, previous = previous, current; continue
                predicted, _ = predict_current(predictor, previous_previous, previous, current)
                targets = {"persistence": previous, "predictor": predicted}
                for name, target in targets.items():
                    sums[name]["z1"] += F.mse_loss(target.z1, current.z1).item()
                    sums[name]["z4"] += F.mse_loss(target.z4, current.z4).item()
                for layer in ("z1", "z4"):
                    actual = getattr(current, layer) - getattr(previous, layer)
                    forecast = getattr(predicted, layer) - getattr(previous, layer)
                    cosine[layer] += F.cosine_similarity(forecast.flatten(1), actual.flatten(1)).mean().item()
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                static_host = HostFeature(raw.c4, raw.c1, tuple(image.shape[-2:]))
                predictions = {"static": model.decode_from_host_feature(static_host).argmax(1).squeeze(0).cpu()}
                for name, target in targets.items():
                    host = corrected_host_feature(model, raw, current, target, tuple(image.shape[-2:]))
                    predictions[name] = model.decode_from_host_feature(host).argmax(1).squeeze(0).cpu()
                for name in NAMES:
                    update_confusion_matrix(confusion[name], predictions[name], mask)
                vc.append(mask, predictions)
                previous_previous, previous = previous, current
                evaluated += 1
    mvc = vc.means()
    metrics = {name: {"miou": float(torch.nanmean(compute_iou(confusion[name])).item()), "wiou": weighted_iou(confusion[name]), "mvc8": mvc[8][name], "mvc16": mvc[16][name]} for name in NAMES}
    state = {name: {layer: sums[name][layer] / evaluated for layer in ("z1", "z4")} for name in sums}
    for name in state: state[name]["mean"] = (state[name]["z1"] + state[name]["z4"]) / 2
    temporal_signal = "GO" if state["predictor"]["mean"] < state["persistence"]["mean"] and metrics["predictor"]["miou"] > metrics["persistence"]["miou"] else "NO-GO"
    semantic = "GO" if metrics["predictor"]["miou"] >= metrics["static"]["miou"] - 0.005 and metrics["predictor"]["mvc8"] > metrics["static"]["mvc8"] and metrics["predictor"]["mvc16"] > metrics["static"]["mvc16"] else "NO-GO"
    summary = {"protocol": {"sequence_count": len(groups), "total_frame_count": len(dataset.samples), "evaluated_frame_count": evaluated}, "historical_clean_static_miou": 0.6552125562, "metrics": metrics, "state_mse": state, "predictor_delta_cosine": {layer: cosine[layer] / evaluated for layer in cosine}, "mvc_window_counts": vc.window_counts(), "PREDICTOR_TEMPORAL_SIGNAL": temporal_signal, "TEMPORAL_SEMANTIC_CAPABILITY": semantic, "parameters_updated": False}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__": main()
