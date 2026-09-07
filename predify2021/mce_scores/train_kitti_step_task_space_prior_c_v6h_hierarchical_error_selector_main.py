"""Formal C-V6H entrypoint.

中文：C-V6H 分层误差选择器正式入口。

This entrypoint applies the reviewed C-V6H safeguards before delegating to the
implementation module:
1. model selection uses the fixed C-V4 E2 mIoU floor;
2. full-resolution hierarchical logits are rebuilt from the unmasked low-res
   recurrent states, so -1e4 validity masking cannot bleed through bilinear
   interpolation at motion boundaries;
3. final result metadata records the actual two-stage evidence roles.
"""

import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

import predify2021.mce_scores.train_kitti_step_task_space_prior_c_v6h_hierarchical_error_selector as impl


MIOU_HARD_FLOOR = 0.6637739071008685


def _arg_value(argv, name, default):
    args = list(sys.argv[1:] if argv is None else argv)
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix):]
    return default


def _selection_key(metrics):
    """Fixed project rule: pass C-V4 E2 mIoU floor first, then maximize mTC."""
    candidate = metrics["c_v6"]
    passed = candidate["mIoU"] >= MIOU_HARD_FLOOR
    if passed:
        return (1, candidate["mTC"], candidate["mIoU"])
    return (0, candidate["mIoU"], candidate["mTC"])


def _selector_evidence_no_mask_bleed(*args, **kwargs):
    """Build full-res hierarchy from raw heads, then apply full-res validity.

    The selector itself masks invalid history cells with -1e4 at controller
    resolution. Interpolating those masked logits would leak the large negative
    sentinel into neighbouring valid pixels.  We therefore reuse the validated
    error/dynamics construction only to obtain the recurrent hidden states, run
    the two heads on those raw states, upsample raw logits, and mask once at the
    final resolution.
    """
    evidence = impl._ORIGINAL_SELECTOR_EVIDENCE(*args, **kwargs)
    selector = args[0] if args else kwargs["selector"]
    c_v3_logits = args[2] if len(args) > 2 else kwargs["c_v3_logits"]
    candidate_rows = args[3] if len(args) > 3 else kwargs["candidate_rows"]

    row = evidence["row"]
    hidden = row["hidden"]
    gate_hidden = hidden[:, : selector.hidden_channels]
    history_hidden = hidden[:, selector.hidden_channels :]
    raw_gate_low = selector.gate_head(gate_hidden)
    raw_history_low = selector.history_head(history_hidden)

    full_size = tuple(c_v3_logits.shape[-2:])
    gate_full = F.interpolate(
        raw_gate_low,
        size=full_size,
        mode="bilinear",
        align_corners=False,
    )
    raw_history_full = F.interpolate(
        raw_history_low,
        size=full_size,
        mode="bilinear",
        align_corners=False,
    )

    any_valid = torch.zeros_like(gate_full[:, :1], dtype=torch.bool)
    history_channels = []
    for index in range(selector.history_length):
        if index < len(candidate_rows):
            valid = candidate_rows[index]["valid_full"].unsqueeze(1).bool()
            any_valid |= valid
            channel = torch.where(
                valid,
                raw_history_full[:, index : index + 1],
                torch.full_like(raw_history_full[:, index : index + 1], -1.0e4),
            )
        else:
            channel = torch.full_like(
                raw_history_full[:, index : index + 1],
                -1.0e4,
            )
        history_channels.append(channel)
    history_full = torch.cat(history_channels, dim=1)

    gate_full = torch.cat(
        (
            gate_full[:, :1],
            torch.where(
                any_valid,
                gate_full[:, 1:2],
                torch.full_like(gate_full[:, 1:2], -1.0e4),
            ),
        ),
        dim=1,
    )

    best_history = history_full.max(dim=1, keepdim=True).values
    history_relative = history_full - best_history
    history_relative = torch.where(
        any_valid,
        history_relative,
        torch.full_like(history_relative, -1.0e4),
    )

    evidence["gate_logits_full"] = gate_full
    evidence["history_logits_full"] = history_full
    evidence["selector_logits_full"] = torch.cat(
        (gate_full[:, :1], gate_full[:, 1:2] + history_relative),
        dim=1,
    )
    return evidence


def _finalize_metadata(argv):
    result_dir = Path(_arg_value(argv, "--result-output", impl.RESULT_DEFAULT))
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        return
    with summary_path.open() as handle:
        summary = json.load(handle)

    summary["experiment"] = (
        "C-V6H Hierarchical Error-Centric Multi-Hypothesis Temporal Coding"
    )
    summary["architecture"].update(
        {
            "decision_decomposition": (
                "Stage1 Current-vs-History; Stage2 t-1..t-K"
            ),
            "stage1_semantic_evidence": (
                "strict validity-gated e1..eK + explicit Dynamics Error from e1"
            ),
            "stage1_dynamics_role": (
                "t-1 error persistence evidence only; not history-age selection"
            ),
            "stage1_loss": "balanced Current/History CE (0.5 / 0.5)",
            "stage2_semantic_evidence": (
                "strict validity-gated e1..eK; no Dynamics Error"
            ),
            "stage2_loss": (
                "history-age CE only on pixels whose task target is History"
            ),
            "training_loss": (
                "balanced gate CE + conditional history CE; no distillation"
            ),
            "five_way_unweighted_ce": False,
            "full_resolution_validity_masking": (
                "upsample raw gate/history logits first; apply -1e4 mask only at full resolution"
            ),
        }
    )
    summary["selection_rule"] = {
        "hard_constraint": (
            f"C-V6H mIoU >= fixed C-V4 E2 floor {MIOU_HARD_FLOOR:.16f}"
        ),
        "objective_after_constraint": "maximize C-V6H mTC, then mIoU",
        "fallback_if_all_fail_floor": "highest mIoU, then mTC",
    }
    candidate = summary.get("best", {}).get("metrics", {}).get("c_v6", {})
    if "mIoU" in candidate:
        summary.setdefault("target", {})["hard_mIoU_floor"] = MIOU_HARD_FLOOR
        summary["target"]["mIoU_floor_passed"] = (
            candidate["mIoU"] >= MIOU_HARD_FLOOR
        )
    summary["diagnostic_intent"] = {
        "gate_history_recall_precision": (
            "tests whether error/dynamics evidence can separate History from Current"
        ),
        "history_age_distribution": (
            "tests whether multi-hypothesis Prediction Error can resolve t-1..t-K"
        ),
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)


def main(argv=None):
    original_impl_evidence = impl._selector_evidence
    original_base_selection_key = impl.base._selection_key
    impl._selector_evidence = _selector_evidence_no_mask_bleed
    impl.base._selection_key = _selection_key
    try:
        result = impl.main(argv)
        _finalize_metadata(argv)
        return result
    finally:
        impl._selector_evidence = original_impl_evidence
        impl.base._selection_key = original_base_selection_key


if __name__ == "__main__":
    main()
