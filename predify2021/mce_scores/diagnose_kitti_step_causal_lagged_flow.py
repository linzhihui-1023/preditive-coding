"""Causal lagged-flow diagnostic for task-space persistence on KITTI-STEP."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import VideoConsistency
from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import FAST_B_CHECKPOINT_DEFAULT, load_fast_b_model
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_only import FULL9, _host_frame, _pair_mtc
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT, flow_grid


NUM_CLASSES = 19
IGNORE_LABEL = 255
RESULT_DEFAULT = "results/kitti_step_causal_lagged_flow.json"


def _warp_low(previous_low, flow, output_size, fallback):
    previous_probability = F.softmax(
        F.interpolate(previous_low, size=output_size, mode="bilinear", align_corners=False), dim=1
    )
    grid, valid = flow_grid(flow, output_size[0], output_size[1])
    warped_probability = F.grid_sample(
        previous_probability.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    prediction = warped_probability.argmax(1)
    valid = valid.squeeze(0)
    prediction[:, ~valid] = fallback[:, ~valid]
    return prediction


def _miou(confusion):
    return float(torch.nanmean(compute_iou(confusion)).item())


@torch.inference_mode()
def evaluate(model, groups, raft):
    names = ("host", "persistence", "lagged_raft", "oracle_raft_low")
    confusion = {n: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for n in names}
    vc_sums = {n: {8: 0.0, 16: 0.0} for n in names}; vc_counts = {n: {8: 0, 16: 0} for n in names}
    mtc_sums = {n: 0.0 for n in names}; mtc_counts = {n: 0 for n in names}
    per_sequence = {}
    for sequence in FULL9:
        seq_conf = {n: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for n in names}
        seq_vc = {n: VideoConsistency() for n in names}; seq_mtc_s = {n: 0.0 for n in names}; seq_mtc_c = {n: 0 for n in names}
        previous_image = previous_low = previous_flow = None; previous_predictions = {}
        for index, sample in enumerate(groups[sequence]):
            image, host_logits, host_low, output_size = _host_frame(model, sample)
            host_prediction = host_logits.argmax(1)
            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"])
            if previous_image is None:
                current_flow = None
            else:
                current_flow = raft.current_to_previous(image, previous_image)
            persistence = host_prediction if previous_low is None else F.interpolate(previous_low, size=output_size, mode="bilinear", align_corners=False).argmax(1)
            if previous_low is None:
                lagged = host_prediction
                oracle = host_prediction
            else:
                # Oracle uses the flow observed by the current frame: F_{t->t-1}.
                oracle = _warp_low(previous_low, current_flow, output_size, persistence)
                # Lagged prediction uses only the already observed previous flow:
                # F_hat_{t+1->t} = F_{t->t-1}.  At index 1 no older flow exists,
                # so persistence is the strictly causal fallback.
                lagged = persistence if previous_flow is None else _warp_low(previous_low, previous_flow, output_size, persistence)
            predictions = {"host": host_prediction, "persistence": persistence, "lagged_raft": lagged, "oracle_raft_low": oracle}
            for name, prediction in predictions.items():
                pred_cpu = prediction.squeeze(0).cpu(); update_confusion_matrix(confusion[name], pred_cpu, gt_cpu); update_confusion_matrix(seq_conf[name], pred_cpu, gt_cpu); seq_vc[name].update(gt_cpu, pred_cpu)
            if previous_image is not None:
                for name, prediction in predictions.items():
                    score = _pair_mtc(previous_predictions[name], prediction, current_flow)
                    if math.isfinite(score): mtc_sums[name] += score; mtc_counts[name] += 1; seq_mtc_s[name] += score; seq_mtc_c[name] += 1
            previous_image, previous_low, previous_flow = image, host_low.detach(), current_flow
            previous_predictions = {n: p.detach() for n, p in predictions.items()}
        for name in names:
            st = seq_vc[name].stats()
            for length in (8, 16): vc_sums[name][length] += st[length]["sum"]; vc_counts[name][length] += st[length]["count"]
        per_sequence[sequence] = {"metrics": {n: {"mIoU": _miou(seq_conf[n]), "mTC": seq_mtc_s[n] / seq_mtc_c[n] if seq_mtc_c[n] else float('nan'), "mVC8": seq_vc[n].values()[8], "mVC16": seq_vc[n].values()[16], "valid_frame_pairs": seq_mtc_c[n]} for n in names}}
    metrics = {n: {"mIoU": _miou(confusion[n]), "mTC": mtc_sums[n] / mtc_counts[n] if mtc_counts[n] else float('nan'), "mVC8": vc_sums[n][8] / vc_counts[n][8] if vc_counts[n][8] else float('nan'), "mVC16": vc_sums[n][16] / vc_counts[n][16] if vc_counts[n][16] else float('nan'), "valid_frame_pairs": mtc_counts[n]} for n in names}
    delta = {n: {k: metrics[n][k] - metrics["persistence"][k] for k in ("mIoU", "mTC", "mVC8", "mVC16")} for n in ("lagged_raft", "oracle_raft_low")}
    return {"metrics": metrics, "delta_vs_persistence": delta, "per_sequence": per_sequence, "causal_protocol": "Lagged frame t uses only F_{t-1->t-2}; Oracle frame t uses F_{t->t-1}. Frame 0/1 lagged fallback is persistence."}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="/home/lin/predify/kitti_step"); parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT); parser.add_argument("--result-output", default=RESULT_DEFAULT); args = parser.parse_args(argv)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval(); dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val"); all_groups = sequence_groups(dataset); missing = [s for s in FULL9 if s not in all_groups]
    if missing: raise RuntimeError(f"Missing Full9 sequences: {missing}")
    groups = {s: all_groups[s] for s in FULL9}; result = {"experiment": "Causal Lagged-Flow Diagnostic", "training": False, "weights_modified": False, "full9": list(FULL9), "definitions": {"persistence": "zero motion", "lagged_raft": "previous observed flow reused for next-frame transport", "oracle_raft_low": "current-frame true RAFT-low flow"}, "results": evaluate(model, groups, FrozenRAFT())}
    output = Path(args.result_output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__": main()
