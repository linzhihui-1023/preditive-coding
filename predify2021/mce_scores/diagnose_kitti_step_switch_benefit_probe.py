"""Frozen Switch Benefit Probe (HW_TC vs HC_TW) on KITTI-STEP Full9.

The positive label is ``Host wrong / temporal correct`` (HW_TC), while the
negative label is ``Host correct / temporal wrong`` (HC_TW).  Only decisive
pixels are retained.  A small logistic probe is fit with leave-one-sequence-
out validation; model weights and all inference modules remain frozen.
"""

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
from predify2021.mce_scores.diagnose_kitti_step_repairability import (
    FULL9,
    _load_model,
    _upsample,
)
from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import (
    FAST_B_CHECKPOINT_DEFAULT,
    STAGE_T_CHECKPOINT_DEFAULT,
)
from predify2021.mce_scores.train_kitti_step_v2_auxiliary import probe_logits
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    AuxiliaryTemporalPredictor,
    AuxiliaryTemporalStateEncoder,
    HostFeature,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import error_state, zero_state


NUM_CLASSES = 19
IGNORE_LABEL = 255
RESULT_DEFAULT = "results/kitti_step_switch_benefit_probe.json"
SAMPLE_PER_FRAME = 512
FEATURE_NAMES = (
    "host_entropy", "host_margin", "temporal_entropy", "temporal_margin",
    "host_temporal_disagreement", "host_temporal_js",
    "error_abs_mean",
)
FEATURE_DIM = len(FEATURE_NAMES) + NUM_CLASSES


def _entropy_and_margin(logits):
    probabilities = F.softmax(logits, dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1) / math.log(NUM_CLASSES)
    top2 = probabilities.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return probabilities, entropy.squeeze(0), margin.squeeze(0)


def _auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positive = labels == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        return float("nan")
    ranks = rankdata(scores, method="average")
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _operating_point(scores, labels, min_recall=0.30, min_precision=0.60):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    total_positive = max(int((labels == 1).sum()), 1)
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    tp = np.cumsum(ordered == 1)
    fp = np.cumsum(ordered == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_positive
    eligible = (precision >= min_precision) & (recall >= min_recall)
    if eligible.any():
        indices = np.where(eligible)[0]
        index = int(indices[np.argmax(recall[indices])])
        status = True
    else:
        eligible_recall = recall >= min_recall
        if eligible_recall.any():
            indices = np.where(eligible_recall)[0]
            index = int(indices[np.argmax(precision[indices])])
        else:
            index = int(np.argmax(precision))
        status = False
    threshold = float(scores[order[index]])
    return {
        "pass": bool(status),
        "threshold": threshold,
        "precision": float(precision[index]),
        "recall_hw_tc_coverage": float(recall[index]),
        "selected_pixels": int(index + 1),
        "true_positive": int(tp[index]),
        "false_positive": int(fp[index]),
        "minimum_precision": min_precision,
        "minimum_recall": min_recall,
    }


def _feature_matrix(model, role_predictor, encoder, predictor, groups):
    per_sequence = {}
    for sequence in FULL9:
        rows, labels = [], []
        pending = hidden = None
        pending_role = h4_role = h4_sem_role = h1_role = h1_sem_role = None
        for index, sample in enumerate(groups[sequence]):
            image = load_image(sample)
            raw = model.extract_backbone_features(image)
            observation = model.encode_backbone_features(raw)
            output_size = tuple(image.shape[-2:])
            state = encoder(observation.z4)
            if index == 0:
                pending, hidden = predictor.predict_next(state, torch.zeros_like(state), None)
                pending_role, _, h4_role, h4_sem_role, h1_role, h1_sem_role = role_predictor.step(
                    observation, zero_state(observation), None, None
                )
            predicted = state if index == 0 else pending
            host_logits = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))
            temporal_logits = probe_logits(model, raw, observation, predicted, output_size)
            gt = semantic_mask_from_panoptic_png(sample["mask_path"]).cuda(non_blocking=True)
            host_pred = host_logits.argmax(1).squeeze(0)
            temporal_pred = temporal_logits.argmax(1).squeeze(0)
            if index > 0:
                valid = gt != IGNORE_LABEL
                host_correct = host_pred == gt
                temporal_correct = temporal_pred == gt
                c_mask = valid & ~host_correct & temporal_correct
                b_mask = valid & host_correct & ~temporal_correct
                decisive = c_mask | b_mask
                indices = torch.where(decisive.flatten())[0]
                if indices.numel() > SAMPLE_PER_FRAME:
                    stride = max(1, int(math.ceil(indices.numel() / SAMPLE_PER_FRAME)))
                    indices = indices[::stride][:SAMPLE_PER_FRAME]
                if indices.numel():
                    host_p, host_e, host_m = _entropy_and_margin(host_logits)
                    temp_p, temp_e, temp_m = _entropy_and_margin(temporal_logits)
                    mean_p = 0.5 * (host_p + temp_p)
                    js = 0.5 * (
                        (host_p * (host_p.clamp_min(1e-12).log() - mean_p.clamp_min(1e-12).log())).sum(1)
                        + (temp_p * (temp_p.clamp_min(1e-12).log() - mean_p.clamp_min(1e-12).log())).sum(1)
                    )
                    error_abs = _upsample((state - pending).abs().mean(1), output_size)
                    scalar = torch.stack((
                        host_e, host_m, temp_e, temp_m,
                        (host_pred != temporal_pred).float(), js.squeeze(0), error_abs,
                    ), dim=0).permute(1, 2, 0)
                    probability_difference = (host_p - temp_p).squeeze(0).permute(1, 2, 0)
                    features = torch.cat((scalar, probability_difference), dim=2).reshape(-1, FEATURE_DIM)
                    flat_labels = c_mask.flatten()[indices].long()
                    # C is positive (1), B is negative (0); decisive indices only.
                    rows.append(features[indices].float().cpu().numpy())
                    labels.append(flat_labels.cpu().numpy())
                error = state - pending
                pending, hidden = predictor.predict_next(state, error, hidden)
                role_error = error_state(observation, pending_role)
                pending_role, _, h4_role, h4_sem_role, h1_role, h1_sem_role = role_predictor.step(
                    observation, role_error, h4_role, h4_sem_role, h1_role, h1_sem_role
                )
        x = np.concatenate(rows, axis=0) if rows else np.empty((0, FEATURE_DIM), dtype=np.float32)
        y = np.concatenate(labels, axis=0) if labels else np.empty((0,), dtype=np.int64)
        per_sequence[sequence] = {"features": x, "labels": y}
    return per_sequence


def _fit_predict(train_x, train_y, test_x):
    mean = train_x.mean(0)
    std = train_x.std(0)
    std[std < 1e-6] = 1.0
    x_train = torch.from_numpy(((train_x - mean) / std).astype(np.float32)).cuda()
    y_train = torch.from_numpy(train_y.astype(np.float32)).cuda()
    x_test = torch.from_numpy(((test_x - mean) / std).astype(np.float32)).cuda()
    probe = nn.Linear(FEATURE_DIM, 1).cuda()
    nn.init.zeros_(probe.weight)
    nn.init.zeros_(probe.bias)
    optimizer = torch.optim.Adam(probe.parameters(), lr=5e-2, weight_decay=1e-3)
    batch_size = 16384
    generator = torch.Generator(device="cuda").manual_seed(0)
    probe.train()
    for _ in range(20):
        order = torch.randperm(x_train.shape[0], generator=generator, device="cuda")
        for start in range(0, x_train.shape[0], batch_size):
            idx = order[start : start + batch_size]
            loss = F.binary_cross_entropy_with_logits(probe(x_train[idx]).squeeze(1), y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    probe.eval()
    with torch.no_grad():
        scores = torch.sigmoid(probe(x_test).squeeze(1)).cpu().numpy()
    return scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--stage-t-checkpoint", default=STAGE_T_CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model, role_predictor = _load_model(args.fast_b_checkpoint)
    payload = torch.load(args.stage_t_checkpoint, map_location="cpu", weights_only=False)
    encoder = AuxiliaryTemporalStateEncoder().cuda()
    predictor = AuxiliaryTemporalPredictor().cuda()
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    encoder.requires_grad_(False).eval()
    predictor.requires_grad_(False).eval()
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    all_groups = sequence_groups(dataset)
    missing = [s for s in FULL9 if s not in all_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 sequences: {missing}")
    groups = {s: all_groups[s] for s in FULL9}
    data = _feature_matrix(model, role_predictor, encoder, predictor, groups)
    fold_rows = []
    all_scores, all_labels = [], []
    for held_out in FULL9:
        train_sequences = [s for s in FULL9 if s != held_out]
        train_x = np.concatenate([data[s]["features"] for s in train_sequences], axis=0)
        train_y = np.concatenate([data[s]["labels"] for s in train_sequences], axis=0)
        test_x, test_y = data[held_out]["features"], data[held_out]["labels"]
        scores = _fit_predict(train_x, train_y, test_x)
        fold_rows.append({
            "held_out_sequence": held_out,
            "samples": int(test_y.size),
            "C_positive": int(test_y.sum()),
            "B_negative": int((test_y == 0).sum()),
            "auroc": _auc(scores, test_y),
            "operating_point": _operating_point(scores, test_y),
        })
        all_scores.append(scores)
        all_labels.append(test_y)
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    result = {
        "experiment": "Frozen Switch Benefit Probe",
        "inference_only_features": True,
        "training": "Only the lightweight logistic probe is trained; Host, Adapter, Dynamics, Stage-T encoder and predictor are frozen.",
        "full9": list(FULL9),
        "label_definition": {"positive": "HW_TC: Host wrong / Temporal correct", "negative": "HC_TW: Host correct / Temporal wrong"},
        "decisive_only": True,
        "sample_per_frame_max": SAMPLE_PER_FRAME,
        "feature_names": list(FEATURE_NAMES) + [f"probability_difference_class_{i}" for i in range(NUM_CLASSES)],
        "loso": {
            "folds": fold_rows,
            "pooled_auroc": _auc(scores, labels),
            "pooled_operating_point": _operating_point(scores, labels),
        },
        "per_sequence_samples": {
            s: {"samples": int(data[s]["labels"].size), "C_positive": int(data[s]["labels"].sum()), "B_negative": int((data[s]["labels"] == 0).sum())}
            for s in FULL9
        },
        "baseline_entropy_only": {
            "pooled_auroc": _auc(
                np.concatenate([data[s]["features"][:, 0] for s in FULL9]),
                labels,
            ),
            "note": "Entropy-only AUROC is evaluated on the same decisive sampled pixels; its raw orientation is higher entropy -> positive HW_TC.",
        },
    }
    output = Path(args.result_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
