"""Second causal-signal separability diagnostic with Host reliability inputs.

The frame labels and the original four error signals are reused verbatim from
the first Full9 diagnostic.  This pass adds only frozen-Host entropy,
confidence, and prediction-margin statistics, then compares LOSO logistic
regression for error-only, Host-only, and all seven signals.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    sequence_groups,
)
from predify2021.mce_scores.oracle_gain_signal_separability import (
    CHECKPOINT_DEFAULT,
    RESULT_DEFAULT as V1_RESULT_DEFAULT,
    SEQUENCES,
    SIGNALS as ERROR_SIGNALS,
    fit_logistic,
    load_frozen,
    roc_auc,
    spearman,
)
from predify2021.mce_scores.train_kitti_step_z4_adaptive_gain import (
    encode,
)
from predify2021.mce_scores.train_kitti_step_z4_only import (
    FAST_B_DEFAULT,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature
from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset

HOST_SIGNALS = ("host_entropy", "host_confidence", "host_margin")
ALL_SIGNALS = ERROR_SIGNALS + HOST_SIGNALS
RESULT_DEFAULT = "results/kitti_step_z4_adaptive_gain_signal_separability_v2"


def host_statistics(model, groups, records):
    by_frame = {(row["sequence"], row["frame"]): row for row in records}
    for sequence in SEQUENCES:
        for sample in groups[sequence][1:]:
            key = (sequence, sample["frame_id"])
            if key not in by_frame:
                raise RuntimeError(f"Missing first-pass record for {key}")
            _, _, raw, output_size = encode(model, sample)
            logits = model.decode_from_host_feature(
                HostFeature(raw.c4, raw.c1, output_size)
            )
            probability = logits.softmax(1)[0]
            target = semantic_mask_from_panoptic_png(sample["mask_path"]).to(
                probability.device, non_blocking=True
            )
            valid = target != 255
            if not valid.any():
                raise RuntimeError(f"No valid labels for {key}")
            entropy = -(probability.clamp_min(1e-8) * probability.clamp_min(1e-8).log()).sum(0)
            confidence = probability.max(0).values
            top2 = probability.topk(2, dim=0).values
            margin = top2[0] - top2[1]
            row = by_frame[key]
            row["host_entropy"] = float(entropy[valid].mean().item())
            row["host_confidence"] = float(confidence[valid].mean().item())
            row["host_margin"] = float(margin[valid].mean().item())


def _fit_logistic(train_records, test_records, signals):
    x_train = torch.tensor([[row[key] for key in signals] for row in train_records], dtype=torch.float64)
    y_train = torch.tensor([row["label"] == "Prediction-helpful" for row in train_records], dtype=torch.float64)
    x_test = torch.tensor([[row[key] for key in signals] for row in test_records], dtype=torch.float64)
    mean = x_train.mean(0); std = x_train.std(0, unbiased=False).clamp_min(1e-12)
    x_train = (x_train - mean) / std; x_test = (x_test - mean) / std
    weights = torch.zeros((len(signals),), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights, bias], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x_train @ weights + bias, y_train) + 1e-3 * weights.square().sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    scores = torch.sigmoid(x_test @ weights.detach() + bias.detach()).tolist()
    labels = [row["label"] == "Prediction-helpful" for row in test_records]
    predicted = [score >= 0.5 for score in scores]
    positive = [index for index, label in enumerate(labels) if label]
    negative = [index for index, label in enumerate(labels) if not label]
    balanced = float("nan")
    if positive and negative:
        tpr = sum(predicted[index] for index in positive) / len(positive)
        tnr = sum(not predicted[index] for index in negative) / len(negative)
        balanced = 0.5 * (tpr + tnr)
    return {
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": _pr_auc(labels, scores),
        "balanced_accuracy": balanced,
        "test_count": len(test_records),
    }


def _pr_auc(labels, scores):
    positive_count = sum(labels)
    if positive_count == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    true_positive = 0; area = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            true_positive += 1
            area += true_positive / rank
    return area / positive_count


def _loso(records, signals):
    folds = []
    for held_out in SEQUENCES:
        train = [row for row in records if row["sequence"] != held_out]
        test = [row for row in records if row["sequence"] == held_out]
        result = _fit_logistic(train, test, signals)
        result["held_out_sequence"] = held_out
        folds.append(result)
    valid = [row for row in folds if math.isfinite(row["roc_auc"])]
    return {
        "folds": folds,
        "mean": {
            key: sum(row[key] for row in valid) / max(len(valid), 1)
            for key in ("roc_auc", "pr_auc", "balanced_accuracy")
        },
    }


def _summary(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values), "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "p25": float(torch.quantile(tensor, 0.25)),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p75": float(torch.quantile(tensor, 0.75)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--first-pass-result", default=V1_RESULT_DEFAULT)
    parser.add_argument("--stage-p-checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_DEFAULT)
    parser.add_argument("--dynamics-checkpoint", default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    first = json.loads((Path(args.first_pass_result) / "summary.json").read_text())
    records = list(csv.DictReader((Path(args.first_pass_result) / "frame_records.csv").open()))
    for row in records:
        row["B_t"] = float(row["B_t"])
        row["label"] = row["label"]
        for key in ERROR_SIGNALS:
            row[key] = float(row[key])
    model, _, stage_payload = load_frozen(args)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    all_groups = sequence_groups(dataset)
    groups = {sequence: all_groups[sequence] for sequence in SEQUENCES}
    with torch.inference_mode():
        host_statistics(model, groups, records)

    labels = [row["label"] == "Prediction-helpful" for row in records]
    distributions = {}
    for signal in ALL_SIGNALS:
        distributions[signal] = {
            "Prediction-helpful": _summary([row[signal] for row in records if row["label"] == "Prediction-helpful"]),
            "Host-preferred": _summary([row[signal] for row in records if row["label"] == "Host-preferred"]),
            "auc": roc_auc(labels, [row[signal] for row in records]),
            "spearman_vs_B": spearman([row[signal] for row in records], [row["B_t"] for row in records]),
        }

    # Recompute all three LOSO groups with the exact same labels and folds.
    error_only = _loso(records, ERROR_SIGNALS)
    host_only = _loso(records, HOST_SIGNALS)
    combined = _loso(records, ALL_SIGNALS)
    output = Path(args.result_output); output.mkdir(parents=True, exist_ok=True)
    fields = ("sequence", "frame", "B_t", "label", "K_star", "host_ce", "ce_K0", "ce_K0.5", "ce_K0.75") + ALL_SIGNALS
    with (output / "frame_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    result = {
        "experiment": "Prediction Error causal signal separability v2",
        "inference_only": True,
        "source_first_pass": str(Path(args.first_pass_result) / "summary.json"),
        "checkpoint": args.stage_p_checkpoint,
        "checkpoint_epoch": stage_payload.get("epoch"),
        "record_count": len(records),
        "helpful_count": sum(labels),
        "host_preferred_count": len(records) - sum(labels),
        "candidate_K": [0.0, 0.5, 0.75, 1.0],
        "benefit_definition": "B_t = CE(K=1 Host) - min(CE(K=0), CE(K=.5), CE(K=.75))",
        "signals": {"error_only": list(ERROR_SIGNALS), "host_reliability_only": list(HOST_SIGNALS), "combined": list(ALL_SIGNALS)},
        "signal_distributions": distributions,
        "single_signal_auc": {signal: distributions[signal]["auc"] for signal in ALL_SIGNALS},
        "spearman_vs_B": {signal: distributions[signal]["spearman_vs_B"] for signal in ALL_SIGNALS},
        "loso": {"error_only": error_only, "host_reliability_only": host_only, "combined": combined},
        "first_pass_record_count": first.get("record_count"),
        "input_restrictions": {
            "classifier_inputs": list(ALL_SIGNALS),
            "GT_used_only_for_reusing_B_label": True,
            "RAFT_used_as_classifier_input": False,
            "future_frame_used": False,
            "oracle_K_used_as_classifier_input": False,
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text(
        "# Adaptive Gain causal signal separability v2\n\n"
        "The first-pass B_t labels are reused exactly. Frozen Host entropy, "
        "confidence, and margin are added; error-only, Host-only, and seven-"
        "signal logistic regression use sequence-held-out LOSO.\n"
    )
    print(json.dumps({"result": str(output / "summary.json"), "records": len(records)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
