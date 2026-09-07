"""C-V6R: training repair for C-V6 soft temporal correction.

中文：C-V6R 训练修正版。

This revision keeps the C-V6 architecture and the successful probability-space
history transport unchanged. It changes only the two training mechanisms that
were directly implicated by the completed C-V6 3-Epoch result (commit 1982594):

1. Exact-zero trainable soft gate
   - C-V6 used sigmoid(raw) with raw bias=-20, which left validation lambda at
     only ~1e-6 after 3 epochs.
   - C-V6R initializes the raw gate to exactly zero and uses
         lambda = clamp(raw, 0, 1)
     for output mixing. Gate supervision is balanced SmoothL1 on the raw gate.
     Therefore Epoch 0 is exactly C-V3 while rescue pixels have an O(1) direct
     gate gradient from the first optimizer step.

2. Conflict-only history-attention supervision
   - C-V6 supervised attention on every pixel where any history was GT-correct;
     those easy pixels outnumbered true rescue pixels by ~57x.
   - C-V6R supervises attention only where valid history candidates disagree in
     correctness: at least one valid history is GT-correct AND at least one
     valid history is GT-wrong. Correct histories share target mass uniformly.

Persistence supervision is intentionally unchanged. It can only become active
once the gate produces successful corrections, so the C-V6 result did not test
that mechanism independently.

K=4, Prediction Error, Dynamics Error, frozen C-V3 history, no output feedback,
TBPTT=8, probability-space one-warp transport, and all four loss weights remain
unchanged. No sweep is introduced.
"""

import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

import predify2021.mce_scores.train_kitti_step_task_space_prior_c_v6_soft_temporal_correction_impl as base
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_soft_temporal_correction import (
    SetBasedSoftTemporalCorrector,
)


GATE_INIT_RAW = 0.0
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v6r_soft_temporal_correction"
)
RESULT_DEFAULT = (
    "results/kitti_step_task_space_prior_c_v6r_soft_temporal_correction"
)
IMPLEMENTATION_NAME = "c_v6r_exact_zero_gate_conflict_attention"
SOURCE_C_V6_RESULT_COMMIT = "1982594"


_BASE_SEMANTIC_TARGETS = base._semantic_targets


class CV6RSoftTemporalCorrector(SetBasedSoftTemporalCorrector):
    """C-V6 corrector with an exact-zero, directly trainable soft gate."""

    def __init__(self, *args, **kwargs):
        kwargs["gate_init_bias"] = GATE_INIT_RAW
        super().__init__(*args, **kwargs)
        torch.nn.init.zeros_(self.lambda_head.weight)
        torch.nn.init.zeros_(self.lambda_head.bias)
        self.gate_init_bias = GATE_INIT_RAW

    def forward(
        self,
        current_probability,
        history_probabilities,
        history_validities,
        prediction_error,
        dynamics_error,
        transportability_low,
        memory_reliability_low,
        correction_hidden=None,
    ):
        row = super().forward(
            current_probability,
            history_probabilities,
            history_validities,
            prediction_error,
            dynamics_error,
            transportability_low,
            memory_reliability_low,
            correction_hidden,
        )
        raw_gate = row["lambda_logit"]
        row["lambda_probability"] = (
            raw_gate.clamp(0.0, 1.0) * row["history_available"]
        )
        return row


def _full_resolution_soft_output(c_v3_logits, candidate_rows, corrector_row):
    """C-V6 full-resolution fusion with lambda=clamp(raw_gate, 0, 1)."""
    full_size = tuple(c_v3_logits.shape[-2:])
    current_probability = F.softmax(c_v3_logits.detach().float(), dim=1)

    score_full = F.interpolate(
        corrector_row["history_scores"],
        size=full_size,
        mode="bilinear",
        align_corners=False,
    )
    full_validities = []
    history_probabilities = []
    for index in range(corrector_row["history_scores"].shape[1]):
        if index < len(candidate_rows):
            row = candidate_rows[index]
            validity = row["valid_full"].bool().unsqueeze(1)
            full_validities.append(validity)
            history_probabilities.append(row["probability"].detach().float())
        else:
            full_validities.append(
                torch.zeros_like(current_probability[:, :1], dtype=torch.bool)
            )
            history_probabilities.append(current_probability)

    valid_tensor = torch.cat(full_validities, dim=1)
    masked_scores = score_full.masked_fill(~valid_tensor, -1.0e4)
    attention = torch.softmax(masked_scores, dim=1) * valid_tensor.to(score_full.dtype)
    attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

    history_probability = torch.zeros_like(current_probability)
    for index, probability in enumerate(history_probabilities):
        history_probability = (
            history_probability + attention[:, index : index + 1] * probability
        )
    history_available = valid_tensor.any(dim=1, keepdim=True)
    history_probability = base._renormalize_probability(
        history_probability,
        history_available,
    )

    lambda_raw = F.interpolate(
        corrector_row["lambda_logit"],
        size=full_size,
        mode="bilinear",
        align_corners=False,
    )
    lambda_probability = (
        lambda_raw.clamp(0.0, 1.0)
        * history_available.to(lambda_raw.dtype)
    )
    output_probability = (
        (1.0 - lambda_probability) * current_probability
        + lambda_probability * history_probability
    )
    output_probability = base._renormalize_probability(output_probability)
    return {
        "current_probability": current_probability,
        "history_probability": history_probability,
        "output_probability": output_probability,
        "attention": attention,
        # Keep the historical key for compatibility. In C-V6R this is raw gate,
        # not a sigmoid logit that must climb from -20.
        "lambda_logit": lambda_raw,
        "lambda_raw": lambda_raw,
        "lambda_probability": lambda_probability,
        "history_validities": valid_tensor,
    }


def _semantic_targets(full, candidate_rows, gt_cpu):
    """Retain C-V6 gate targets; restrict attention to correctness conflicts."""
    targets = _BASE_SEMANTIC_TARGETS(full, candidate_rows, gt_cpu)
    gt = targets["gt"]
    valid_gt = targets["valid_gt"]

    any_history_wrong = torch.zeros_like(valid_gt)
    for row in candidate_rows:
        valid = row["valid_full"][0].bool() & valid_gt
        pred = row["probability"].argmax(dim=1)[0]
        any_history_wrong |= valid & (pred != gt)

    attention_conflict = (
        valid_gt
        & targets["any_history_correct"]
        & any_history_wrong
    )
    targets["attention_supervised"] = attention_conflict
    targets["any_history_wrong"] = any_history_wrong
    return targets


def _balanced_gate_loss(lambda_raw, gate_target, gate_supervised):
    """Balanced direct regression of raw gate to Rescue=1 / Protection=0."""
    if not bool(gate_supervised.any()):
        return lambda_raw.sum() * 0.0

    raw = lambda_raw[0, 0][gate_supervised]
    target = gate_target[gate_supervised]
    positive = target.sum()
    total = target.new_tensor(float(target.numel()))
    negative = total - positive

    per_pixel = F.smooth_l1_loss(raw, target, reduction="none")
    if positive.item() <= 0.0 or negative.item() <= 0.0:
        return per_pixel.mean()

    positive_weight = total / (2.0 * positive)
    negative_weight = total / (2.0 * negative)
    weights = torch.where(target > 0.5, positive_weight, negative_weight)
    return (per_pixel * weights).mean()


def _arg_value(argv, name, default):
    args = list(sys.argv[1:] if argv is None else argv)
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return default


def _patch_base():
    base.GATE_INIT_BIAS = GATE_INIT_RAW
    base.OUTPUT_DEFAULT = OUTPUT_DEFAULT
    base.RESULT_DEFAULT = RESULT_DEFAULT
    base.SetBasedSoftTemporalCorrector = CV6RSoftTemporalCorrector
    base._full_resolution_soft_output = _full_resolution_soft_output
    base._semantic_targets = _semantic_targets
    base._balanced_gate_loss = _balanced_gate_loss


def _rewrite_metadata(argv):
    result_dir = Path(_arg_value(argv, "--result-output", RESULT_DEFAULT))
    output_dir = Path(_arg_value(argv, "--output", OUTPUT_DEFAULT))

    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open() as handle:
            summary = json.load(handle)
        summary["experiment"] = (
            "C-V6R Set-based Deep Semantic Evidence + Exact-Zero Soft Stateful Correction"
        )
        summary["implementation"] = IMPLEMENTATION_NAME
        summary["source_decision_evidence"]["c_v6_result_commit"] = (
            SOURCE_C_V6_RESULT_COMMIT
        )
        summary["source_decision_evidence"]["c_v6_failure_mode"] = (
            "C-V6 validation lambda stayed near 1e-6 and Full9 differed from C-V3 "
            "at only 62 pixels after Epoch 3; persistence received only 2 pixels."
        )
        summary["architecture"]["gate_parameterization"] = (
            "raw gate initialized at 0; lambda=clamp(raw,0,1)"
        )
        summary["architecture"]["exact_zero_c_v3_initialization"] = True
        summary["losses"]["attention"] = (
            "uniform mass over GT-correct history only where valid history "
            "candidates disagree in correctness"
        )
        summary["losses"]["gate"] = (
            "balanced SmoothL1 raw-gate regression: Rescue=1, Protection=0"
        )
        summary["losses"]["persistence"] = (
            "unchanged from C-V6: RAFT+GT stable correspondence; consecutive "
            "rescue plus previous corrected output actually correct"
        )
        with summary_path.open("w") as handle:
            json.dump(summary, handle, indent=2)

    best_path = output_dir / "best.pt"
    if best_path.exists():
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        payload["experiment"] = "c_v6r_soft_temporal_correction"
        payload["implementation"] = IMPLEMENTATION_NAME
        architecture = payload.setdefault("architecture", {})
        architecture["gate_init_bias"] = GATE_INIT_RAW
        architecture["gate_parameterization"] = (
            "raw gate initialized at 0; lambda=clamp(raw,0,1)"
        )
        architecture["attention_supervision"] = (
            "history correctness-conflict pixels only"
        )
        torch.save(payload, best_path)


def main(argv=None):
    _patch_base()
    result = base.main(argv)
    _rewrite_metadata(argv)
    return result


if __name__ == "__main__":
    main()
