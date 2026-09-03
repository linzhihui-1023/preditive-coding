"""Decode Stage-P predicted Z4 features for several training epochs.

This is inference-only: each checkpoint is frozen and evaluated on the same
Full9 causal protocol.  It reports frame-weighted prediction MSE ratio and the
segmentation quality obtained by sending the predicted Z4 through the existing
frozen C4 writeback and decoder.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import VideoConsistency
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
    sequence_groups,
)
from predify2021.mce_scores.evaluate_kitti_step_raft_error_memory import FrozenRAFT
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    load_components,
    residual_writeback_host_feature,
)
from predify2021.mce_scores.train_kitti_step_z4_only import FAST_B_DEFAULT, z4_predict_next
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorRegulatedSemanticRestorationPredictor,
    HostFeature,
    UnifiedFeatures,
)

SEQUENCES = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
EPOCHS = (1, 5, 10, 15)
NUM_CLASSES = 19
RESULT_DEFAULT = "results/kitti_step_predictor_feature_decode"
CHECKPOINT_DIR_DEFAULT = "/home/lin/predify/experiments/kitti_step_z4_only_stage_p"


def load_components_once(args):
    model, _ = load_components(
        STATIC_CHECKPOINT_DEFAULT,
        ADAPTER_CHECKPOINT_DEFAULT,
        args.dynamics_checkpoint,
        WRITEBACK_CHECKPOINT_DEFAULT,
    )
    fast = torch.load(args.fast_b_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.output_adapters[3].load_state_dict(
        fast["c4_output_adapter_state_dict"], strict=True
    )
    model.host_conditioned_writebacks["3"].load_state_dict(
        fast["c4_writeback_state_dict"], strict=True
    )
    predictors = {}
    payloads = {}
    for epoch in EPOCHS:
        path = Path(args.checkpoint_dir) / f"epoch_{epoch:03d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        predictor = ErrorRegulatedSemanticRestorationPredictor(
            use_error_temporal_stats=True
        ).cuda()
        predictor.load_state_dict(payload["model_state_dict"], strict=True)
        predictor.requires_grad_(False).eval()
        predictors[epoch] = predictor
        payloads[epoch] = payload
    model.requires_grad_(False).eval().cuda()
    return model, predictors, payloads


def encode(model, sample):
    image = load_image(sample)
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        observation = model.encode_backbone_features(raw)
    return image, observation, raw, tuple(image.shape[-2:])


def decode_predicted_batch(model, raw, observation, pending_values, output_size):
    count = len(pending_values)
    delta = UnifiedFeatures(
        torch.zeros_like(observation.z1).expand(count, -1, -1, -1),
        torch.zeros_like(observation.z2).expand(count, -1, -1, -1),
        torch.zeros_like(observation.z3).expand(count, -1, -1, -1),
        torch.cat(tuple(pending - observation.z4 for pending in pending_values), dim=0),
    )
    feature = residual_writeback_host_feature(model, raw, delta, output_size)
    return model.decode_from_host_feature(feature)


def pair_mtc(previous_prediction, current_prediction, backward_flow):
    height, width = current_prediction.shape[-2:]
    flow = backward_flow
    source_h, source_w = flow.shape[-2:]
    if (source_h, source_w) != (height, width):
        flow = torch.nn.functional.interpolate(flow, size=(height, width), mode="bilinear", align_corners=True)
        flow = flow.clone(); flow[:, 0].mul_(width / source_w); flow[:, 1].mul_(height / source_h)
    y, x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype), indexing="ij"
    )
    source_x = x.unsqueeze(0) + flow[:, 0]; source_y = y.unsqueeze(0) + flow[:, 1]
    valid = (source_x >= 0) & (source_x <= width - 1) & (source_y >= 0) & (source_y <= height - 1)
    grid = torch.stack((2 * source_x / max(width - 1, 1) - 1, 2 * source_y / max(height - 1, 1) - 1), dim=-1)
    warped = torch.nn.functional.grid_sample(
        previous_prediction.float().unsqueeze(1), grid, mode="nearest", padding_mode="zeros", align_corners=True
    )[0, 0].long()
    keep = valid[0]; a, b = warped[keep].cpu(), current_prediction[0][keep].cpu()
    if not a.numel():
        return float("nan")
    confusion = torch.bincount(NUM_CLASSES * a + b, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    return float(torch.nanmean(compute_iou(confusion)).item())


@torch.inference_mode()
def evaluate(model, predictors, groups, raft):
    names = tuple(f"epoch_{epoch:03d}" for epoch in EPOCHS)
    all_names = ("host",) + names
    confusion = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in all_names}
    vc = {name: VideoConsistency() for name in all_names}
    mvc_sums = {name: {8: 0.0, 16: 0.0} for name in all_names}; mvc_counts = {name: {8: 0, 16: 0} for name in all_names}
    mtc_sum = {name: 0.0 for name in all_names}; mtc_count = {name: 0 for name in all_names}
    pred_sum = {name: 0.0 for name in names}; copy_sum = {name: 0.0 for name in names}; frame_count = {name: 0 for name in names}
    per_sequence = {}
    for sequence in SEQUENCES:
        samples = groups[sequence]
        seq_conf = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in all_names}
        seq_vc = {name: VideoConsistency() for name in all_names}
        seq_mtc_sum = {name: 0.0 for name in all_names}; seq_mtc_count = {name: 0 for name in all_names}
        seq_pred_sum = {name: 0.0 for name in names}; seq_copy_sum = {name: 0.0 for name in names}; seq_frames = {name: 0 for name in names}
        image, observation, raw, output_size = encode(model, samples[0])
        host_logits = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))
        mask = semantic_mask_from_panoptic_png(samples[0]["mask_path"])
        predictions = {"host": host_logits.argmax(1)}
        hidden = {}; pending = {}
        for epoch, predictor in predictors.items():
            name = f"epoch_{epoch:03d}"
            pending[epoch], hidden[epoch] = z4_predict_next(predictor, observation.z4, torch.zeros_like(observation.z4), None)
            predictions[name] = predictions["host"]
        previous_observation = observation.z4
        previous_image = image; previous_predictions = {name: pred.detach() for name, pred in predictions.items()}
        for name, prediction in predictions.items():
            update_confusion_matrix(confusion[name], prediction[0].cpu(), mask); update_confusion_matrix(seq_conf[name], prediction[0].cpu(), mask); vc[name].update(mask, prediction[0].cpu()); seq_vc[name].update(mask, prediction[0].cpu())
        for sample in samples[1:]:
            image, observation, raw, output_size = encode(model, sample)
            mask = semantic_mask_from_panoptic_png(sample["mask_path"])
            host_logits = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))
            host_prediction = host_logits.argmax(1)
            pending_values = tuple(pending[epoch] for epoch in EPOCHS)
            predicted_logits = decode_predicted_batch(model, raw, observation, pending_values, output_size)
            predictions = {"host": host_prediction}
            for index, epoch in enumerate(EPOCHS):
                predictions[f"epoch_{epoch:03d}"] = predicted_logits[index:index + 1].argmax(1)
                target = observation.z4
                pred_mse = torch.nn.functional.mse_loss(pending[epoch], target).item()
                copy_mse = torch.nn.functional.mse_loss(previous_observation, target).item()
                pred_sum[f"epoch_{epoch:03d}"] += pred_mse; copy_sum[f"epoch_{epoch:03d}"] += copy_mse; frame_count[f"epoch_{epoch:03d}"] += 1
                seq_pred_sum[f"epoch_{epoch:03d}"] += pred_mse; seq_copy_sum[f"epoch_{epoch:03d}"] += copy_mse; seq_frames[f"epoch_{epoch:03d}"] += 1
            for name, prediction in predictions.items():
                update_confusion_matrix(confusion[name], prediction[0].cpu(), mask); update_confusion_matrix(seq_conf[name], prediction[0].cpu(), mask); vc[name].update(mask, prediction[0].cpu()); seq_vc[name].update(mask, prediction[0].cpu())
            backward_flow = raft.backward_flow(image, previous_image)
            for name, prediction in predictions.items():
                score = pair_mtc(previous_predictions[name], prediction, backward_flow)
                if math.isfinite(score):
                    mtc_sum[name] += score; mtc_count[name] += 1; seq_mtc_sum[name] += score; seq_mtc_count[name] += 1
            for epoch, predictor in predictors.items():
                error = observation.z4 - pending[epoch]
                pending[epoch], hidden[epoch] = z4_predict_next(predictor, observation.z4, error, hidden[epoch])
            previous_observation = observation.z4; previous_image = image; previous_predictions = {name: pred.detach() for name, pred in predictions.items()}
        for name in all_names:
            stats = seq_vc[name].stats()
            for length in (8, 16): mvc_sums[name][length] += stats[length]["sum"]; mvc_counts[name][length] += stats[length]["count"]
        per_sequence[sequence] = {}
        for name in all_names:
            per_sequence[sequence][name] = {
                "mIoU": float(torch.nanmean(compute_iou(seq_conf[name])).item()),
                "mVC8": seq_vc[name].values()[8], "mVC16": seq_vc[name].values()[16],
                "mTC": seq_mtc_sum[name] / max(seq_mtc_count[name], 1), "valid_frame_pairs": seq_mtc_count[name],
            }
        for name in names:
            per_sequence[sequence][name]["pred_mse"] = seq_pred_sum[name] / max(seq_frames[name], 1)
            per_sequence[sequence][name]["copy_mse"] = seq_copy_sum[name] / max(seq_frames[name], 1)
            per_sequence[sequence][name]["Rpred"] = seq_pred_sum[name] / max(seq_copy_sum[name], 1e-12)
    metrics = {}
    for name in all_names:
        metrics[name] = {
            "mIoU": float(torch.nanmean(compute_iou(confusion[name])).item()),
            "mVC8": mvc_sums[name][8] / max(mvc_counts[name][8], 1), "mVC16": mvc_sums[name][16] / max(mvc_counts[name][16], 1),
            "mTC": mtc_sum[name] / max(mtc_count[name], 1), "valid_frame_pairs": mtc_count[name],
        }
    for name in names:
        metrics[name]["pred_mse"] = pred_sum[name] / max(frame_count[name], 1)
        metrics[name]["copy_mse"] = copy_sum[name] / max(frame_count[name], 1)
        metrics[name]["Rpred"] = pred_sum[name] / max(copy_sum[name], 1e-12)
    metrics["per_sequence"] = per_sequence
    metrics["delta_vs_host"] = {name: {key: metrics[name][key] - metrics["host"][key] for key in ("mIoU", "mVC8", "mVC16", "mTC")} for name in names}
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR_DEFAULT)
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_DEFAULT)
    parser.add_argument("--dynamics-checkpoint", default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model, predictors, payloads = load_components_once(args)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    groups = sequence_groups(dataset)
    groups = {sequence: groups[sequence] for sequence in SEQUENCES}
    raft = FrozenRAFT()
    metrics = evaluate(model, predictors, groups, raft)
    output = Path(args.result_output); output.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "Stage-P predicted-feature decode test",
        "inference_only": True,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_epochs": list(EPOCHS),
        "sequences": list(SEQUENCES),
        "metrics": metrics,
        "checkpoint_metadata": {str(epoch): {"epoch": payloads[epoch].get("epoch"), "source": str(Path(args.checkpoint_dir) / f"epoch_{epoch:03d}.pt")} for epoch in EPOCHS},
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text(
        "# Stage-P predicted-feature decode test\n\n"
        "Inference-only Full9 test of E1/E5/E10/E15. Each frozen predictor's "
        "causal `Z4_hat_t` is sent through the existing C4 writeback and Host "
        "decoder; no parameters are changed. Rpred is frame-weighted.\n"
    )
    print(json.dumps({"result": str(output / "summary.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
