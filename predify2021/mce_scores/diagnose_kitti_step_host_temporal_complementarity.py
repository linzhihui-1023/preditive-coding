"""Zero-training Host/Stage-T temporal complementarity diagnostic on KITTI-STEP Full9.

The frozen Host and the frozen best Stage-T encoder/predictor are evaluated on
exactly the causal Full9 validation protocol. No correction head is trained or
applied. The Stage-T Predicted-T decode is used only as a proxy for historical
temporal evidence that a future task-space prior could provide.

The diagnostic answers three questions before implementing C+D:
1) How often is the temporal proxy correct when the Host is wrong?
2) How much mIoU headroom exists under a GT selector that chooses the temporal
   proxy only when it repairs a Host error?
3) How much of that recoverable set is exposed by Host predictive entropy?

GT is used only for offline diagnostics and the oracle selector. It is never
an input to Host, temporal encoder, or temporal predictor.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import VideoConsistency
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
    sequence_groups,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import (
    FAST_B_CHECKPOINT_DEFAULT,
    STAGE_T_CHECKPOINT_DEFAULT,
    load_fast_b_model,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT, flow_grid
from predify2021.mce_scores.train_kitti_step_v2_auxiliary import probe_logits
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    AuxiliaryTemporalPredictor,
    AuxiliaryTemporalStateEncoder,
    HostFeature,
)


FULL9 = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
NUM_CLASSES = 19
IGNORE_LABEL = 255
ENTROPY_BINS = 4096
FIXED_ENTROPY_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50)
TOP_UNCERTAIN_FRACTIONS = (0.10, 0.25, 0.50)
RESULT_DEFAULT = "results/kitti_step_host_temporal_complementarity.json"


@torch.inference_mode()
def encode(model, sample):
    image = load_image(sample)
    raw = model.extract_backbone_features(image)
    observation = model.encode_backbone_features(raw)
    return image, observation, raw, tuple(image.shape[-2:])


def _pair_mtc(previous_prediction, current_prediction, flow):
    grid, valid = flow_grid(flow, current_prediction.shape[-2], current_prediction.shape[-1])
    warped = F.grid_sample(
        previous_prediction.float().unsqueeze(0),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0).squeeze(0).to(torch.int64)
    keep = valid.squeeze(0)
    a = warped[keep].cpu()
    b = current_prediction.squeeze(0)[keep].cpu()
    if not a.numel():
        return float("nan")
    confusion = torch.bincount(
        NUM_CLASSES * a + b,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    return float(torch.nanmean(compute_iou(confusion)).item())


def _new_counts():
    return {
        "host_correct_temporal_correct": 0,
        "host_correct_temporal_wrong": 0,
        "host_wrong_temporal_correct": 0,
        "host_wrong_temporal_wrong": 0,
        "valid_pixels": 0,
        "host_correct": 0,
        "host_wrong": 0,
        "temporal_correct": 0,
        "temporal_wrong": 0,
    }


def _add_counts(total, host_correct, temporal_correct, valid):
    hc_tc = valid & host_correct & temporal_correct
    hc_tw = valid & host_correct & ~temporal_correct
    hw_tc = valid & ~host_correct & temporal_correct
    hw_tw = valid & ~host_correct & ~temporal_correct
    increments = {
        "host_correct_temporal_correct": int(hc_tc.sum().item()),
        "host_correct_temporal_wrong": int(hc_tw.sum().item()),
        "host_wrong_temporal_correct": int(hw_tc.sum().item()),
        "host_wrong_temporal_wrong": int(hw_tw.sum().item()),
        "valid_pixels": int(valid.sum().item()),
        "host_correct": int((valid & host_correct).sum().item()),
        "host_wrong": int((valid & ~host_correct).sum().item()),
        "temporal_correct": int((valid & temporal_correct).sum().item()),
        "temporal_wrong": int((valid & ~temporal_correct).sum().item()),
    }
    for key, value in increments.items():
        total[key] += value
    return hw_tc


def _rates(counts):
    valid = max(counts["valid_pixels"], 1)
    host_wrong = max(counts["host_wrong"], 1)
    host_correct = max(counts["host_correct"], 1)
    return {
        **counts,
        "four_way_fraction_of_valid": {
            key: counts[key] / valid
            for key in (
                "host_correct_temporal_correct",
                "host_correct_temporal_wrong",
                "host_wrong_temporal_correct",
                "host_wrong_temporal_wrong",
            )
        },
        "recoverable_rate_given_host_wrong": (
            counts["host_wrong_temporal_correct"] / host_wrong
        ),
        "damage_rate_given_host_correct": (
            counts["host_correct_temporal_wrong"] / host_correct
        ),
        "host_wrong_rate": counts["host_wrong"] / valid,
        "temporal_wrong_rate": counts["temporal_wrong"] / valid,
    }


class EntropyAccumulator:
    """Streaming normalized-entropy histograms without storing per-pixel scores."""

    def __init__(self, bins=ENTROPY_BINS):
        self.bins = int(bins)
        self.hist = {
            "all": np.zeros(self.bins, dtype=np.int64),
            "host_correct": np.zeros(self.bins, dtype=np.int64),
            "host_wrong": np.zeros(self.bins, dtype=np.int64),
            "recoverable": np.zeros(self.bins, dtype=np.int64),
        }
        self.sum = {key: 0.0 for key in self.hist}
        self.count = {key: 0 for key in self.hist}

    def _update_one(self, name, entropy, mask):
        count = int(mask.sum().item())
        if not count:
            return
        values = entropy[mask]
        indices = torch.clamp((values * self.bins).long(), 0, self.bins - 1)
        histogram = torch.bincount(indices, minlength=self.bins).cpu().numpy().astype(np.int64)
        self.hist[name] += histogram
        self.sum[name] += float(values.sum().item())
        self.count[name] += count

    def update(self, entropy, valid, host_correct, recoverable):
        self._update_one("all", entropy, valid)
        self._update_one("host_correct", entropy, valid & host_correct)
        self._update_one("host_wrong", entropy, valid & ~host_correct)
        self._update_one("recoverable", entropy, recoverable)

    def _auc_wrong(self):
        pos = self.hist["host_wrong"].astype(np.float64)
        neg = self.hist["host_correct"].astype(np.float64)
        n_pos = pos.sum()
        n_neg = neg.sum()
        if n_pos <= 0 or n_neg <= 0:
            return float("nan")
        neg_lower = 0.0
        favorable = 0.0
        for index in range(self.bins):
            favorable += pos[index] * (neg_lower + 0.5 * neg[index])
            neg_lower += neg[index]
        return float(favorable / (n_pos * n_neg))

    def _threshold_stats(self, threshold):
        index = min(self.bins, max(0, int(math.ceil(threshold * self.bins))))
        all_selected = int(self.hist["all"][index:].sum())
        recover_selected = int(self.hist["recoverable"][index:].sum())
        all_count = max(self.count["all"], 1)
        recover_count = max(self.count["recoverable"], 1)
        return {
            "entropy_threshold": float(threshold),
            "intervention_rate": all_selected / all_count,
            "recoverable_error_coverage": recover_selected / recover_count,
            "selected_valid_pixels": all_selected,
            "selected_recoverable_pixels": recover_selected,
        }

    def _top_fraction_stats(self, fraction):
        target = max(1, int(math.ceil(fraction * max(self.count["all"], 1))))
        reverse = np.cumsum(self.hist["all"][::-1])
        reverse_index = int(np.searchsorted(reverse, target, side="left"))
        bin_index = max(0, self.bins - 1 - reverse_index)
        threshold = bin_index / self.bins
        row = self._threshold_stats(threshold)
        row["target_top_uncertain_fraction"] = float(fraction)
        return row

    def summary(self):
        means = {
            key: self.sum[key] / self.count[key] if self.count[key] else float("nan")
            for key in self.hist
        }
        return {
            "normalized_entropy_definition": "-sum_c p_c log(p_c) / log(C)",
            "histogram_bins": self.bins,
            "auroc_host_entropy_to_host_wrong": self._auc_wrong(),
            "mean_entropy": means,
            "counts": dict(self.count),
            "recoverable_error_coverage_fixed_thresholds": [
                self._threshold_stats(value) for value in FIXED_ENTROPY_THRESHOLDS
            ],
            "recoverable_error_coverage_top_uncertain": [
                self._top_fraction_stats(value) for value in TOP_UNCERTAIN_FRACTIONS
            ],
        }


def _normalized_entropy(logits, temperature):
    probabilities = F.softmax(logits / temperature, dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    return entropy / math.log(NUM_CLASSES)


def _metric_summary(confusion, vc_sums, vc_counts, mtc_sums, mtc_counts):
    result = {}
    for name in confusion:
        result[name] = {
            "mIoU": float(torch.nanmean(compute_iou(confusion[name])).item()),
            "mTC": mtc_sums[name] / mtc_counts[name] if mtc_counts[name] else float("nan"),
            "mVC8": vc_sums[name][8] / vc_counts[name][8] if vc_counts[name][8] else float("nan"),
            "mVC16": vc_sums[name][16] / vc_counts[name][16] if vc_counts[name][16] else float("nan"),
            "valid_frame_pairs": mtc_counts[name],
        }
    return result


@torch.inference_mode()
def evaluate(model, encoder, predictor, groups, raft, temperature):
    names = ("host", "stage_t_temporal_proxy", "gt_oracle_fusion")
    confusion = {
        name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
        for name in names
    }
    vc_sums = {name: {8: 0.0, 16: 0.0} for name in names}
    vc_counts = {name: {8: 0, 16: 0} for name in names}
    mtc_sums = {name: 0.0 for name in names}
    mtc_counts = {name: 0 for name in names}
    global_counts = _new_counts()
    entropy = EntropyAccumulator()
    per_sequence = {}
    temporal_frames = 0

    for sequence in FULL9:
        samples = groups[sequence]
        seq_counts = _new_counts()
        seq_confusion = {
            name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
            for name in names
        }
        seq_vc = {name: VideoConsistency() for name in names}
        seq_mtc_sums = {name: 0.0 for name in names}
        seq_mtc_counts = {name: 0 for name in names}
        pending = None
        hidden = None
        previous_image = None
        previous_predictions = {}

        for index, sample in enumerate(samples):
            image, observation, raw, output_size = encode(model, sample)
            state = encoder(observation.z4)
            if index == 0:
                pending, hidden = predictor.predict_next(
                    state, torch.zeros_like(state), None
                )

            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"])
            gt = gt_cpu.cuda(non_blocking=True)
            host_logits = model.decode_from_host_feature(
                HostFeature(raw.c4, raw.c1, output_size)
            )
            host_prediction = host_logits.argmax(1)

            if index == 0:
                # Match the existing Stage-T Predicted-T task probe on frame 0
                # for metric reproduction, but exclude frame 0 from all
                # complementarity/entropy statistics because no history exists.
                temporal_logits = probe_logits(
                    model, raw, observation, state, output_size
                )
            else:
                temporal_logits = probe_logits(
                    model, raw, observation, pending, output_size
                )
            temporal_prediction = temporal_logits.argmax(1)

            valid = gt != IGNORE_LABEL
            host_correct = host_prediction.squeeze(0) == gt
            temporal_correct = temporal_prediction.squeeze(0) == gt

            oracle_prediction = host_prediction.clone()
            if index > 0:
                recoverable = _add_counts(
                    global_counts, host_correct, temporal_correct, valid
                )
                _add_counts(seq_counts, host_correct, temporal_correct, valid)
                host_entropy = _normalized_entropy(host_logits, temperature).squeeze(0)
                entropy.update(host_entropy, valid, host_correct, recoverable)
                oracle_mask = recoverable.unsqueeze(0)
                oracle_prediction[oracle_mask] = temporal_prediction[oracle_mask]
                temporal_frames += 1

            predictions = {
                "host": host_prediction,
                "stage_t_temporal_proxy": temporal_prediction,
                "gt_oracle_fusion": oracle_prediction,
            }
            for name, prediction in predictions.items():
                pred_cpu = prediction.squeeze(0).cpu()
                update_confusion_matrix(confusion[name], pred_cpu, gt_cpu)
                update_confusion_matrix(seq_confusion[name], pred_cpu, gt_cpu)
                seq_vc[name].update(gt_cpu, pred_cpu)

            if previous_image is not None:
                flow = raft.current_to_previous(image, previous_image)
                for name, prediction in predictions.items():
                    score = _pair_mtc(previous_predictions[name], prediction, flow)
                    if math.isfinite(score):
                        mtc_sums[name] += score
                        mtc_counts[name] += 1
                        seq_mtc_sums[name] += score
                        seq_mtc_counts[name] += 1

            if index > 0:
                error = state - pending
                pending, hidden = predictor.predict_next(state, error, hidden)

            previous_image = image
            previous_predictions = {
                name: prediction.detach() for name, prediction in predictions.items()
            }

        for name in names:
            stats = seq_vc[name].stats()
            for length in (8, 16):
                vc_sums[name][length] += stats[length]["sum"]
                vc_counts[name][length] += stats[length]["count"]

        seq_metrics = {}
        for name in names:
            seq_metrics[name] = {
                "mIoU": float(torch.nanmean(compute_iou(seq_confusion[name])).item()),
                "mTC": (
                    seq_mtc_sums[name] / seq_mtc_counts[name]
                    if seq_mtc_counts[name]
                    else float("nan")
                ),
                "mVC8": seq_vc[name].values()[8],
                "mVC16": seq_vc[name].values()[16],
                "valid_frame_pairs": seq_mtc_counts[name],
            }
        per_sequence[sequence] = {
            "complementarity": _rates(seq_counts),
            "metrics": seq_metrics,
        }

    metrics = _metric_summary(
        confusion, vc_sums, vc_counts, mtc_sums, mtc_counts
    )
    oracle_gain = {
        key: metrics["gt_oracle_fusion"][key] - metrics["host"][key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }
    temporal_delta = {
        key: metrics["stage_t_temporal_proxy"][key] - metrics["host"][key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }
    return {
        "metrics": metrics,
        "delta_vs_host": {
            "stage_t_temporal_proxy": temporal_delta,
            "gt_oracle_fusion": oracle_gain,
        },
        "complementarity_causal_frames_only": _rates(global_counts),
        "host_uncertainty_causal_frames_only": entropy.summary(),
        "per_sequence": per_sequence,
        "temporal_frames_excluding_sequence_first_frames": temporal_frames,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--stage-t-checkpoint", default=STAGE_T_CHECKPOINT_DEFAULT)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")

    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    payload = torch.load(args.stage_t_checkpoint, map_location="cpu", weights_only=False)
    encoder = AuxiliaryTemporalStateEncoder().cuda()
    predictor = AuxiliaryTemporalPredictor().cuda()
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    encoder.requires_grad_(False).eval()
    predictor.requires_grad_(False).eval()

    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root), "val"
    )
    all_groups = sequence_groups(dataset)
    groups = {sequence: all_groups[sequence] for sequence in FULL9}
    expected_temporal_frames = sum(max(len(groups[sequence]) - 1, 0) for sequence in FULL9)

    result = {
        "experiment": "Host-Temporal Complementarity Diagnostic",
        "inference_only": True,
        "training": False,
        "full9": list(FULL9),
        "fast_b_checkpoint": args.fast_b_checkpoint,
        "stage_t_checkpoint": args.stage_t_checkpoint,
        "temperature": args.temperature,
        "temporal_evidence": (
            "Frozen Stage-T Predicted-T decode used as a proxy; this is not a "
            "trained C+D task-space prior."
        ),
        "oracle_definition": (
            "Frame 0 uses Host. On causal frames t>0, keep Host when Host is "
            "correct; if Host is wrong and temporal proxy is correct, select "
            "the temporal prediction; if both are wrong, keep Host. GT is "
            "used only by this offline selector."
        ),
        "complementarity_protocol": (
            "Four-way pixel counts and Host-entropy analysis exclude the first "
            "frame of each sequence, because no historical prediction exists."
        ),
        "expected_temporal_frames": expected_temporal_frames,
        "results": evaluate(
            model, encoder, predictor, groups, FrozenRAFT(), args.temperature
        ),
    }

    observed = result["results"]["temporal_frames_excluding_sequence_first_frames"]
    if observed != expected_temporal_frames:
        raise RuntimeError(
            f"Temporal-frame protocol mismatch: expected {expected_temporal_frames}, got {observed}"
        )

    output = Path(args.result_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
