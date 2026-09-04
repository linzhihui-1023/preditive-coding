"""No-training sanity check for the C-only low-resolution task space.

Compares: (A) native Host logits, (B) current Host logits downsampled to C1
and upsampled again, and (C) the previous frame's C1 logits persisted to the
current frame.  This isolates low-resolution semantic loss from temporal
one-step persistence loss.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import FAST_B_CHECKPOINT_DEFAULT, load_fast_b_model
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature


FULL9 = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
NUM_CLASSES = 19
IGNORE_LABEL = 255
RESULT_DEFAULT = "results/kitti_step_task_space_prior_sanity.json"


def _miou(confusion):
    return float(torch.nanmean(compute_iou(confusion)).item())


@torch.inference_mode()
def evaluate(model, groups):
    names = ("A_host_native", "B_current_low_then_upsample", "C_previous_low_persistence")
    confusion = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in names}
    per_sequence = {}
    frame_counts = {name: 0 for name in names}
    causal_counts = {name: 0 for name in names}
    for sequence in FULL9:
        seq_confusion = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in names}
        previous_low = None
        for index, sample in enumerate(groups[sequence]):
            image = load_image(sample)
            raw = model.extract_backbone_features(image)
            output_size = tuple(image.shape[-2:])
            logits = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))
            low = F.interpolate(logits, size=tuple(raw.c1.shape[-2:]), mode="bilinear", align_corners=False)
            predictions = {
                "A_host_native": logits.argmax(1).squeeze(0).cpu(),
                "B_current_low_then_upsample": F.interpolate(low, size=output_size, mode="bilinear", align_corners=False).argmax(1).squeeze(0).cpu(),
            }
            if previous_low is not None:
                predictions["C_previous_low_persistence"] = F.interpolate(previous_low, size=output_size, mode="bilinear", align_corners=False).argmax(1).squeeze(0).cpu()
            gt = semantic_mask_from_panoptic_png(sample["mask_path"])
            valid = gt != IGNORE_LABEL
            for name in names:
                if name not in predictions:
                    continue
                update_confusion_matrix(confusion[name], predictions[name], gt)
                update_confusion_matrix(seq_confusion[name], predictions[name], gt)
                frame_counts[name] += 1
                if name == "C_previous_low_persistence":
                    causal_counts[name] += int(valid.sum().item() > 0)
            previous_low = low
        per_sequence[sequence] = {
            "mIoU": {name: (_miou(seq_confusion[name]) if frame_counts[name] else float("nan")) for name in names},
            "frames": len(groups[sequence]),
        }
    metrics = {name: {"mIoU": _miou(confusion[name]), "frames": frame_counts[name]} for name in names}
    return {
        "metrics": metrics,
        "delta_vs_A": {name: metrics[name]["mIoU"] - metrics["A_host_native"]["mIoU"] for name in names if name != "A_host_native"},
        "per_sequence": per_sequence,
        "causal_sequence_first_frames_excluded": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    all_groups = sequence_groups(dataset)
    missing = [s for s in FULL9 if s not in all_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 sequences: {missing}")
    groups = {s: all_groups[s] for s in FULL9}
    result = {
        "experiment": "C-only Task-space Prior Low-resolution Sanity Diagnostic",
        "training": False,
        "weights_modified": False,
        "full9": list(FULL9),
        "definitions": {
            "A_host_native": "Current-frame Host logits decoded at native output resolution",
            "B_current_low_then_upsample": "Current Host logits downsampled to raw C1 resolution then upsampled",
            "C_previous_low_persistence": "Previous frame C1 logits upsampled at current frame (causal persistence)",
        },
        "results": evaluate(model, groups),
    }
    output = Path(args.result_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
