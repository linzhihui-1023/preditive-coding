"""Offline separability diagnostic for the causal Adaptive-Gain inputs.

No model parameters are trained or changed.  For every Full9 validation frame
with a valid Stage-P prediction, the frozen C4 writeback/decoder evaluates
K in {0, .5, .75, 1}.  The ground-truth CE is used only to construct the
offline target B_t; the four classifier features are causal gain inputs only.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
    sequence_groups,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    load_components,
    residual_writeback_host_feature,
)
from predify2021.mce_scores.train_kitti_step_z4_only import (
    FAST_B_DEFAULT,
    z4_predict_next,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorRegulatedSemanticRestorationPredictor,
    UnifiedFeatures,
    HostFeature,
)
from predify2021.model_factory.z4_adaptive_gain import Z4AdaptiveGainHead

K_VALUES = (0.0, 0.5, 0.75, 1.0)
NON_HOST_K = K_VALUES[:-1]
SIGNALS = ("error_magnitude", "relative_error", "error_change", "cosine_discrepancy")
SEQUENCES = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
IGNORE = 255
NUM_CLASSES = 19
CHECKPOINT_DEFAULT = "/home/lin/predify/experiments/kitti_step_z4_only_stage_p/best.pt"
RESULT_DEFAULT = "results/kitti_step_z4_adaptive_gain_signal_separability"


def load_frozen(args):
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
    stage = torch.load(args.stage_p_checkpoint, map_location="cpu", weights_only=False)
    predictor = ErrorRegulatedSemanticRestorationPredictor(
        use_error_temporal_stats=True
    ).cuda()
    predictor.load_state_dict(stage["model_state_dict"], strict=True)
    model.requires_grad_(False).eval().cuda()
    predictor.requires_grad_(False).eval()
    return model, predictor, stage


def encode(model, sample):
    image = load_image(sample)
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        observation = model.encode_backbone_features(raw)
    return image, observation, raw, tuple(image.shape[-2:])


def decode_k_batch(model, raw, observation, error, output_size):
    """Decode K<1 candidates as one batch; K=1 is the unchanged Host path."""
    deltas = UnifiedFeatures(
        torch.zeros((len(NON_HOST_K),) + observation.z1.shape[1:], device=observation.z1.device, dtype=observation.z1.dtype),
        torch.zeros((len(NON_HOST_K),) + observation.z2.shape[1:], device=observation.z2.device, dtype=observation.z2.dtype),
        torch.zeros((len(NON_HOST_K),) + observation.z3.shape[1:], device=observation.z3.device, dtype=observation.z3.dtype),
        torch.cat(tuple((float(k) - 1.0) * error for k in NON_HOST_K), dim=0),
    )
    feature = residual_writeback_host_feature(model, raw, deltas, output_size)
    return model.decode_from_host_feature(feature)


def frame_ce(logits, target):
    valid = target != IGNORE
    if not valid.any():
        return torch.zeros((logits.shape[0],), device=logits.device)
    target_batch = target.unsqueeze(0).expand(logits.shape[0], -1, -1)
    losses = F.cross_entropy(logits, target_batch, ignore_index=IGNORE, reduction="none")
    return losses[:, valid].mean(1)


@torch.inference_mode()
def collect_records(model, predictor, groups):
    records = []
    for sequence in SEQUENCES:
        previous_magnitude = None
        samples = groups[sequence]
        image, observation, raw, output_size = encode(model, samples[0])
        pending, hidden = z4_predict_next(
            predictor, observation.z4, torch.zeros_like(observation.z4), None
        )
        for sample in samples[1:]:
            image, observation, raw, output_size = encode(model, sample)
            target = semantic_mask_from_panoptic_png(sample["mask_path"]).to(
                observation.z4.device, non_blocking=True
            )
            error, stats = Z4AdaptiveGainHead.statistics(
                observation.z4, pending, previous_magnitude
            )
            host_logits = model.decode_from_host_feature(
                HostFeature(raw.c4, raw.c1, output_size)
            )
            candidate_logits = decode_k_batch(
                model, raw, observation, error, output_size
            )
            candidate_losses = frame_ce(candidate_logits, target)
            host_loss = frame_ce(host_logits, target)[0]
            best_index = int(candidate_losses.argmin().item())
            best_k = NON_HOST_K[best_index]
            prediction_benefit = float((host_loss - candidate_losses[best_index]).item())
            records.append({
                "sequence": sequence,
                "frame": sample["frame_id"],
                "B_t": prediction_benefit,
                "label": "Prediction-helpful" if prediction_benefit > 0.0 else "Host-preferred",
                "K_star": best_k,
                "host_ce": float(host_loss.item()),
                "ce_K0": float(candidate_losses[0].item()),
                "ce_K0.5": float(candidate_losses[1].item()),
                "ce_K0.75": float(candidate_losses[2].item()),
                "error_magnitude": float(stats[0, 0].item()),
                "relative_error": float(stats[0, 1].item()),
                "error_change": float(stats[0, 2].item()),
                "cosine_discrepancy": float(stats[0, 3].item()),
            })
            previous_magnitude = stats[:, 0].detach()
            pending, hidden = z4_predict_next(predictor, observation.z4, error, hidden)
    return records


def _quantiles(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "p25": float(torch.quantile(tensor, 0.25).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p75": float(torch.quantile(tensor, 0.75).item()),
    }


def _pearson(x, y):
    if len(x) < 2:
        return float("nan")
    a = torch.tensor(x, dtype=torch.float64); b = torch.tensor(y, dtype=torch.float64)
    a = a - a.mean(); b = b - b.mean()
    den = a.square().sum().sqrt() * b.square().sum().sqrt()
    return float((a.mul(b).sum() / den).item()) if den > 0 else float("nan")


def _rank(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def spearman(x, y):
    return _pearson(_rank(x), _rank(y))


def roc_auc(labels, scores):
    positives = sum(labels); negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _rank(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def pr_auc(labels, scores):
    positives = sum(labels)
    if positives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    true_positive = 0; area = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            true_positive += 1
            area += true_positive / rank
    return area / positives


def balanced_accuracy(labels, scores, threshold=0.5):
    predicted = [score >= threshold for score in scores]
    positive = [i for i, label in enumerate(labels) if label]
    negative = [i for i, label in enumerate(labels) if not label]
    if not positive or not negative:
        return float("nan")
    tpr = sum(predicted[i] for i in positive) / len(positive)
    tnr = sum(not predicted[i] for i in negative) / len(negative)
    return 0.5 * (tpr + tnr)


def fit_logistic(train_records, test_records):
    x_train = torch.tensor([[row[key] for key in SIGNALS] for row in train_records], dtype=torch.float64)
    y_train = torch.tensor([row["label"] == "Prediction-helpful" for row in train_records], dtype=torch.float64)
    x_test = torch.tensor([[row[key] for key in SIGNALS] for row in test_records], dtype=torch.float64)
    mean = x_train.mean(0); std = x_train.std(0, unbiased=False).clamp_min(1e-12)
    x_train = (x_train - mean) / std; x_test = (x_test - mean) / std
    weights = torch.zeros((len(SIGNALS),), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights, bias], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        logits = x_train @ weights + bias
        loss = F.binary_cross_entropy_with_logits(logits, y_train) + 1e-3 * weights.square().sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    scores = torch.sigmoid(x_test @ weights.detach() + bias.detach()).tolist()
    labels = [row["label"] == "Prediction-helpful" for row in test_records]
    return {
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": pr_auc(labels, scores),
        "balanced_accuracy": balanced_accuracy(labels, scores),
        "test_count": len(test_records),
        "test_helpful_fraction": sum(labels) / max(len(labels), 1),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--stage-p-checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_DEFAULT)
    parser.add_argument("--dynamics-checkpoint", default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model, predictor, stage_payload = load_frozen(args)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    all_groups = sequence_groups(dataset)
    groups = {sequence: all_groups[sequence] for sequence in SEQUENCES}
    with torch.inference_mode():
        records = collect_records(model, predictor, groups)

    labels = [row["label"] == "Prediction-helpful" for row in records]
    signal_distributions = {}
    for signal in SIGNALS:
        signal_distributions[signal] = {
            "Prediction-helpful": _quantiles([row[signal] for row in records if row["label"] == "Prediction-helpful"]),
            "Host-preferred": _quantiles([row[signal] for row in records if row["label"] == "Host-preferred"]),
            "auc": roc_auc(labels, [row[signal] for row in records]),
            "spearman_vs_B": spearman([row[signal] for row in records], [row["B_t"] for row in records]),
        }
    folds = []
    for held_out in SEQUENCES:
        train = [row for row in records if row["sequence"] != held_out]
        test = [row for row in records if row["sequence"] == held_out]
        result = fit_logistic(train, test)
        result["held_out_sequence"] = held_out
        folds.append(result)
    valid_folds = [row for row in folds if math.isfinite(row["roc_auc"])]
    combined = {
        "roc_auc": sum(row["roc_auc"] for row in valid_folds) / max(len(valid_folds), 1),
        "pr_auc": sum(row["pr_auc"] for row in valid_folds) / max(len(valid_folds), 1),
        "balanced_accuracy": sum(row["balanced_accuracy"] for row in valid_folds) / max(len(valid_folds), 1),
        "fold_count": len(valid_folds),
    }
    output = Path(args.result_output); output.mkdir(parents=True, exist_ok=True)
    with (output / "frame_records.csv").open("w", newline="") as handle:
        fields = ("sequence", "frame", "B_t", "label", "K_star", "host_ce", "ce_K0", "ce_K0.5", "ce_K0.75") + SIGNALS
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    result = {
        "experiment": "Prediction Error causal signal separability",
        "inference_only": True,
        "checkpoint": args.stage_p_checkpoint,
        "checkpoint_epoch": stage_payload.get("epoch"),
        "sequences": list(SEQUENCES),
        "candidate_K": list(K_VALUES),
        "benefit_definition": "B_t = CE(K=1 Host) - min(CE(K=0), CE(K=.5), CE(K=.75))",
        "label_definition": "Prediction-helpful iff B_t > 0; otherwise Host-preferred",
        "record_count": len(records),
        "helpful_count": sum(labels),
        "host_preferred_count": len(records) - sum(labels),
        "signal_distributions": signal_distributions,
        "spearman_vs_B": {signal: signal_distributions[signal]["spearman_vs_B"] for signal in SIGNALS},
        "single_signal_auc": {signal: signal_distributions[signal]["auc"] for signal in SIGNALS},
        "logistic_loso": {"folds": folds, "mean": combined},
        "input_restrictions": {
            "classifier_inputs": list(SIGNALS),
            "GT_used_only_for_offline_B_label": True,
            "RAFT_used_as_classifier_input": False,
            "future_frame_used": False,
            "oracle_K_used_as_classifier_input": False,
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text(
        "# Prediction Error Causal Signal Separability\n\n"
        "Frozen Stage-P/Host Full9 diagnostic. K candidates are 0, 0.5, 0.75, "
        "and 1; ground truth is used only to define the offline intervention "
        "benefit B_t. Logistic regression is evaluated by sequence-held-out LOSO.\n"
    )
    print(json.dumps({"result": str(output / "summary.json"), "records": len(records)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
