"""Frozen selection diagnostic for the RAFT-warped task-space prior."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import VideoConsistency
from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import FAST_B_CHECKPOINT_DEFAULT, load_fast_b_model
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_only import FULL9, _host_frame, _pair_mtc, _new_counts, _add_counts, _rates
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT, flow_grid


NUM_CLASSES = 19
IGNORE_LABEL = 255
RESULT_DEFAULT = "results/kitti_step_warped_prior_selection.json"
GATE_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50)
SAMPLE_PER_FRAME = 512


def _entropy_margin(logits):
    p = F.softmax(logits, dim=1)
    entropy = (-(p * p.clamp_min(1e-12).log()).sum(1) / math.log(NUM_CLASSES)).squeeze(0)
    margin = (p.topk(2, dim=1).values[:, 0] - p.topk(2, dim=1).values[:, 1]).squeeze(0)
    return p, entropy, margin


def _auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64); labels = np.asarray(labels, dtype=np.int64)
    pos = labels == 1; neg = ~pos
    if not pos.any() or not neg.any(): return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def _operating_point(scores, labels, min_recall=0.30, min_precision=0.60):
    scores = np.asarray(scores); labels = np.asarray(labels); order = np.argsort(-scores, kind="mergesort")
    y = labels[order]; tp = np.cumsum(y == 1); fp = np.cumsum(y == 0); total = max(int((labels == 1).sum()), 1)
    precision = tp / np.maximum(tp + fp, 1); recall = tp / total
    eligible = (precision >= min_precision) & (recall >= min_recall)
    if eligible.any():
        idxs = np.where(eligible)[0]; idx = int(idxs[np.argmax(recall[idxs])]); passed = True
    else:
        idx = int(np.argmax(precision)); passed = False
    return {"pass": bool(passed), "threshold": float(scores[order[idx]]), "precision": float(precision[idx]), "recall_hw_pc_coverage": float(recall[idx]), "selected_pixels": int(idx + 1), "true_positive": int(tp[idx]), "false_positive": int(fp[idx]), "minimum_precision": min_precision, "minimum_recall": min_recall}


def _train_loso(data):
    rows = []
    all_scores, all_labels = [], []
    dim = next(iter(data.values()))["x"].shape[1]
    for held_out in FULL9:
        train = [s for s in FULL9 if s != held_out]
        train_x = np.concatenate([data[s]["x"] for s in train]); train_y = np.concatenate([data[s]["y"] for s in train])
        test_x, test_y = data[held_out]["x"], data[held_out]["y"]
        mean, std = train_x.mean(0), train_x.std(0); std[std < 1e-6] = 1.0
        x = torch.from_numpy(((train_x - mean) / std).astype(np.float32)).cuda(); y = torch.from_numpy(train_y.astype(np.float32)).cuda()
        xt = torch.from_numpy(((test_x - mean) / std).astype(np.float32)).cuda()
        probe = nn.Linear(dim, 1).cuda(); nn.init.zeros_(probe.weight); nn.init.zeros_(probe.bias)
        opt = torch.optim.Adam(probe.parameters(), lr=5e-2, weight_decay=1e-3)
        gen = torch.Generator(device="cuda").manual_seed(0)
        for _ in range(20):
            order = torch.randperm(x.shape[0], generator=gen, device="cuda")
            for start in range(0, x.shape[0], 16384):
                idx = order[start:start + 16384]
                loss = F.binary_cross_entropy_with_logits(probe(x[idx]).squeeze(1), y[idx])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad(): scores = torch.sigmoid(probe(xt).squeeze(1)).cpu().numpy()
        rows.append({"held_out_sequence": held_out, "samples": int(test_y.size), "HW_PC_positive": int(test_y.sum()), "HC_PW_negative": int((test_y == 0).sum()), "auroc": _auc(scores, test_y), "operating_point": _operating_point(scores, test_y)})
        all_scores.append(scores); all_labels.append(test_y)
    scores, labels = np.concatenate(all_scores), np.concatenate(all_labels)
    return {"folds": rows, "pooled_auroc": _auc(scores, labels), "pooled_operating_point": _operating_point(scores, labels), "pooled_samples": int(labels.size), "pooled_HW_PC": int(labels.sum()), "pooled_HC_PW": int((labels == 0).sum())}


@torch.inference_mode()
def evaluate(model, groups, raft):
    names = ("host", "persistence", "warped_prior", "repair_only_label_oracle") + tuple(f"entropy_tau_{t:.2f}".replace('.', 'p') for t in GATE_THRESHOLDS)
    confusion = {n: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for n in names}
    vc_sums = {n: {8: 0.0, 16: 0.0} for n in names}; vc_counts = {n: {8: 0, 16: 0} for n in names}
    mtc_sums = {n: 0.0 for n in names}; mtc_counts = {n: 0 for n in names}
    global_counts = _new_counts(); gate_counts = {n: _new_counts() for n in names if n.startswith('entropy_')}
    samples_by_seq = {}
    per_sequence = {}
    for sequence in FULL9:
        seq_counts = _new_counts(); seq_conf = {n: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for n in names}; seq_vc = {n: VideoConsistency() for n in names}; seq_mtc_s = {n: 0.0 for n in names}; seq_mtc_c = {n: 0 for n in names}
        previous_image = previous_low = None; previous_predictions = {}; feature_rows = []; label_rows = []
        for index, sample in enumerate(groups[sequence]):
            image, host_logits, host_low, output_size = _host_frame(model, sample); host_pred = host_logits.argmax(1)
            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"]); gt = gt_cpu.cuda(non_blocking=True)
            persistence = host_pred if previous_low is None else F.interpolate(previous_low, size=output_size, mode="bilinear", align_corners=False).argmax(1)
            warped = host_pred.clone(); flow = None; flow_valid = None
            if previous_low is not None:
                flow = raft.current_to_previous(image, previous_image); grid, flow_valid = flow_grid(flow, output_size[0], output_size[1])
                prev_prob = F.softmax(F.interpolate(previous_low, size=output_size, mode="bilinear", align_corners=False), dim=1)
                warped_prob = F.grid_sample(prev_prob.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
                warped = warped_prob.argmax(1); valid_flow = flow_valid.squeeze(0); warped[:, ~valid_flow] = host_pred[:, ~valid_flow]
                valid = gt != IGNORE_LABEL; hc = host_pred.squeeze(0) == gt; wc = warped.squeeze(0) == gt
                rec = _add_counts(global_counts, hc, wc, valid); _add_counts(seq_counts, hc, wc, valid)
                entropy_h = _entropy_margin(host_logits)[1]; p_h, entropy_h, margin_h = _entropy_margin(host_logits); p_w, entropy_w, margin_w = _entropy_margin(warped_prob.log().clamp_min(-30.0))
                mean_p = 0.5 * (p_h + p_w); js = 0.5 * ((p_h * (p_h.clamp_min(1e-12).log() - mean_p.clamp_min(1e-12).log())).sum(1) + (p_w * (p_w.clamp_min(1e-12).log() - mean_p.clamp_min(1e-12).log())).sum(1)).squeeze(0)
                scalar = torch.stack((entropy_h, margin_h, entropy_w, margin_w, (host_pred != warped).float().squeeze(0), js), dim=0).permute(1, 2, 0)
                diff = (p_h - p_w).squeeze(0).permute(1, 2, 0); features = torch.cat((scalar, diff), dim=2).reshape(-1, 6 + NUM_CLASSES)
                decisive = valid & ((~hc & wc) | (hc & ~wc)); idx = torch.where(decisive.flatten())[0]
                if idx.numel() > SAMPLE_PER_FRAME:
                    stride = max(1, int(math.ceil(idx.numel() / SAMPLE_PER_FRAME))); idx = idx[::stride][:SAMPLE_PER_FRAME]
                if idx.numel(): feature_rows.append(features[idx].float().cpu().numpy()); label_rows.append((~hc.flatten()[idx] & wc.flatten()[idx]).long().cpu().numpy())
                oracle = host_pred.clone(); oracle[rec.unsqueeze(0)] = warped[rec.unsqueeze(0)]
                for gate_name, threshold in zip(gate_counts, GATE_THRESHOLDS):
                    selected = valid & (entropy_h >= threshold); _add_counts(gate_counts[gate_name], hc, wc, selected)
            else:
                oracle = host_pred
            host_entropy = _entropy_margin(host_logits)[1]
            gates = {f"entropy_tau_{t:.2f}".replace('.', 'p'): (warped if previous_low is not None and torch.any(host_entropy >= t) else host_pred) for t in GATE_THRESHOLDS}
            # Above expression is replaced pixelwise below; retain Host on frame 0.
            if previous_low is not None:
                gates = {f"entropy_tau_{t:.2f}".replace('.', 'p'): torch.where((host_entropy >= t).unsqueeze(0), warped, host_pred) for t in GATE_THRESHOLDS}
            predictions = {"host": host_pred, "persistence": persistence, "warped_prior": warped, "repair_only_label_oracle": oracle, **gates}
            for n, pred in predictions.items():
                pc = pred.squeeze(0).cpu(); update_confusion_matrix(confusion[n], pc, gt_cpu); update_confusion_matrix(seq_conf[n], pc, gt_cpu); seq_vc[n].update(gt_cpu, pc)
            if previous_image is not None:
                for n, pred in predictions.items():
                    score = _pair_mtc(previous_predictions[n], pred, flow)
                    if math.isfinite(score): mtc_sums[n] += score; mtc_counts[n] += 1; seq_mtc_s[n] += score; seq_mtc_c[n] += 1
            previous_image, previous_low = image, host_low.detach(); previous_predictions = {n: p.detach() for n, p in predictions.items()}
        for n in names:
            st = seq_vc[n].stats()
            for l in (8, 16): vc_sums[n][l] += st[l]["sum"]; vc_counts[n][l] += st[l]["count"]
        per_sequence[sequence] = {"complementarity": _rates(seq_counts), "metrics": {n: {"mIoU": float(torch.nanmean(compute_iou(seq_conf[n])).item()), "mTC": seq_mtc_s[n] / seq_mtc_c[n] if seq_mtc_c[n] else float('nan'), "mVC8": seq_vc[n].values()[8], "mVC16": seq_vc[n].values()[16], "valid_frame_pairs": seq_mtc_c[n]} for n in names}}
        samples_by_seq[sequence] = {"x": np.concatenate(feature_rows) if feature_rows else np.empty((0, 6 + NUM_CLASSES), np.float32), "y": np.concatenate(label_rows) if label_rows else np.empty((0,), np.int64)}
    metrics = {n: {"mIoU": float(torch.nanmean(compute_iou(confusion[n])).item()), "mTC": mtc_sums[n] / mtc_counts[n] if mtc_counts[n] else float('nan'), "mVC8": vc_sums[n][8] / vc_counts[n][8] if vc_counts[n][8] else float('nan'), "mVC16": vc_sums[n][16] / vc_counts[n][16] if vc_counts[n][16] else float('nan'), "valid_frame_pairs": mtc_counts[n]} for n in names}
    delta = {n: {k: metrics[n][k] - metrics["host"][k] for k in ("mIoU", "mTC", "mVC8", "mVC16")} for n in names if n != "host"}
    selection = {n: {"entropy_threshold": t, **_rates(gate_counts[n]), "intervention_rate_over_causal_valid": gate_counts[n]["valid_pixels"] / max(global_counts["valid_pixels"], 1), "recoverable_error_coverage": gate_counts[n]["host_wrong_prior_correct"] / max(global_counts["host_wrong_prior_correct"], 1), "harmful_error_coverage": gate_counts[n]["host_correct_prior_wrong"] / max(global_counts["host_correct_prior_wrong"], 1)} for n, t in zip(gate_counts, GATE_THRESHOLDS)}
    return {"metrics": metrics, "delta_vs_host": delta, "selection": selection, "complementarity_causal_frames_only": _rates(global_counts), "per_sequence": per_sequence, "probe_data": samples_by_seq}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="/home/lin/predify/kitti_step"); parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT); parser.add_argument("--result-output", default=RESULT_DEFAULT); args = parser.parse_args(argv)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval(); dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val"); all_groups = sequence_groups(dataset); missing = [s for s in FULL9 if s not in all_groups]
    if missing: raise RuntimeError(f"Missing Full9 sequences: {missing}")
    groups = {s: all_groups[s] for s in FULL9}; evaluated = evaluate(model, groups, FrozenRAFT()); probe_data = evaluated.pop("probe_data"); probe = _train_loso(probe_data)
    result = {"experiment": "Frozen Warped Prior Selection Diagnostic", "training": "Only a lightweight logistic selection probe is trained; all inference weights are frozen.", "full9": list(FULL9), "gate_thresholds": list(GATE_THRESHOLDS), "definitions": {"HW_PC": "Host wrong / RAFT-warped prior correct", "HC_PW": "Host correct / RAFT-warped prior wrong"}, "results": evaluated, "switch_benefit_probe": probe, "sample_counts": {s: {"samples": int(d["y"].size), "HW_PC": int(d["y"].sum()), "HC_PW": int((d["y"] == 0).sum())} for s, d in probe_data.items()}}
    output = Path(args.result_output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__": main()
