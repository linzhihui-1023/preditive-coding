"""Frozen RAFT-warped task-space persistence diagnostic on KITTI-STEP Full9."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import FAST_B_CHECKPOINT_DEFAULT, load_fast_b_model
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_only import FULL9, _host_frame, _pair_mtc, _new_counts, _add_counts, _rates
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT, flow_grid
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import VideoConsistency


NUM_CLASSES = 19
IGNORE_LABEL = 255
RESULT_DEFAULT = "results/kitti_step_task_space_prior_raft_warp.json"


def _miou(confusion):
    return float(torch.nanmean(compute_iou(confusion)).item())


@torch.inference_mode()
def evaluate(model, groups, raft):
    names = ("host", "persistence", "raft_warped_persistence", "repair_only_label_oracle")
    confusion = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in names}
    vc_sums = {name: {8: 0.0, 16: 0.0} for name in names}
    vc_counts = {name: {8: 0, 16: 0} for name in names}
    mtc_sums = {name: 0.0 for name in names}
    mtc_counts = {name: 0 for name in names}
    global_counts = _new_counts()
    per_sequence = {}
    for sequence in FULL9:
        seq_confusion = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for name in names}
        seq_vc = {name: VideoConsistency() for name in names}
        seq_mtc_sums = {name: 0.0 for name in names}
        seq_mtc_counts = {name: 0 for name in names}
        seq_counts = _new_counts()
        previous_image = None
        previous_low = None
        previous_predictions = {}
        for index, sample in enumerate(groups[sequence]):
            image, host_logits, host_low, output_size = _host_frame(model, sample)
            host_prediction = host_logits.argmax(1)
            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"])
            gt = gt_cpu.cuda(non_blocking=True)
            persistence_prediction = host_prediction if previous_low is None else F.interpolate(
                previous_low, size=output_size, mode="bilinear", align_corners=False
            ).argmax(1)
            warped_prediction = host_prediction.clone()
            flow = None
            if previous_low is not None:
                flow = raft.current_to_previous(image, previous_image)
                grid, flow_valid = flow_grid(flow, output_size[0], output_size[1])
                previous_probability = F.softmax(
                    F.interpolate(previous_low, size=output_size, mode="bilinear", align_corners=False), dim=1
                )
                warped_probability = F.grid_sample(
                    previous_probability.float(), grid, mode="bilinear",
                    padding_mode="zeros", align_corners=True,
                )
                warped_prediction = warped_probability.argmax(1)
                # Out-of-frame flow samples are not a semantic observation;
                # retain Host there instead of introducing padding class 0.
                warped_prediction[:, ~flow_valid.squeeze(0)] = host_prediction[:, ~flow_valid.squeeze(0)]
                valid = gt != IGNORE_LABEL
                host_correct = host_prediction.squeeze(0) == gt
                warped_correct = warped_prediction.squeeze(0) == gt
                recoverable = _add_counts(global_counts, host_correct, warped_correct, valid)
                _add_counts(seq_counts, host_correct, warped_correct, valid)
                oracle_prediction = host_prediction.clone()
                oracle_prediction[recoverable.unsqueeze(0)] = warped_prediction[recoverable.unsqueeze(0)]
            else:
                oracle_prediction = host_prediction
            predictions = {
                "host": host_prediction,
                "persistence": persistence_prediction,
                "raft_warped_persistence": warped_prediction,
                "repair_only_label_oracle": oracle_prediction,
            }
            for name, prediction in predictions.items():
                pred_cpu = prediction.squeeze(0).cpu()
                update_confusion_matrix(confusion[name], pred_cpu, gt_cpu)
                update_confusion_matrix(seq_confusion[name], pred_cpu, gt_cpu)
                seq_vc[name].update(gt_cpu, pred_cpu)
            if previous_image is not None:
                for name, prediction in predictions.items():
                    score = _pair_mtc(previous_predictions[name], prediction, flow)
                    if math.isfinite(score):
                        mtc_sums[name] += score; mtc_counts[name] += 1
                        seq_mtc_sums[name] += score; seq_mtc_counts[name] += 1
            previous_image = image
            previous_low = host_low.detach()
            previous_predictions = {name: prediction.detach() for name, prediction in predictions.items()}
        for name in names:
            stats = seq_vc[name].stats()
            for length in (8, 16):
                vc_sums[name][length] += stats[length]["sum"]
                vc_counts[name][length] += stats[length]["count"]
        per_sequence[sequence] = {
            "complementarity": _rates(seq_counts),
            "metrics": {
                name: {
                    "mIoU": _miou(seq_confusion[name]),
                    "mTC": seq_mtc_sums[name] / seq_mtc_counts[name] if seq_mtc_counts[name] else float("nan"),
                    "mVC8": seq_vc[name].values()[8], "mVC16": seq_vc[name].values()[16],
                    "valid_frame_pairs": seq_mtc_counts[name],
                } for name in names
            },
        }
    metrics = {
        name: {
            "mIoU": _miou(confusion[name]),
            "mTC": mtc_sums[name] / mtc_counts[name] if mtc_counts[name] else float("nan"),
            "mVC8": vc_sums[name][8] / vc_counts[name][8] if vc_counts[name][8] else float("nan"),
            "mVC16": vc_sums[name][16] / vc_counts[name][16] if vc_counts[name][16] else float("nan"),
            "valid_frame_pairs": mtc_counts[name],
        } for name in names
    }
    return {
        "metrics": metrics,
        "delta_vs_host": {name: {key: metrics[name][key] - metrics["host"][key] for key in ("mIoU", "mTC", "mVC8", "mVC16")} for name in names if name != "host"},
        "delta_warp_minus_persistence": {key: metrics["raft_warped_persistence"][key] - metrics["persistence"][key] for key in ("mIoU", "mTC", "mVC8", "mVC16")},
        "host_persistence_gap_recovered_by_warp": (metrics["raft_warped_persistence"]["mIoU"] - metrics["persistence"]["mIoU"]) / max(metrics["host"]["mIoU"] - metrics["persistence"]["mIoU"], 1e-12),
        "complementarity_causal_frames_only": _rates(global_counts),
        "per_sequence": per_sequence,
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
        "experiment": "RAFT-warped Task-space Persistence Diagnostic",
        "training": False, "weights_modified": False, "full9": list(FULL9),
        "definitions": {
            "A_host": "Current-frame Host prediction",
            "B_persistence": "Previous C1 Host logits upsampled at current frame",
            "C_raft_warped_persistence": "Previous C1 Host probabilities warped with RAFT current-to-previous flow",
        },
        "results": evaluate(model, groups, FrozenRAFT()),
    }
    output = Path(args.result_output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
