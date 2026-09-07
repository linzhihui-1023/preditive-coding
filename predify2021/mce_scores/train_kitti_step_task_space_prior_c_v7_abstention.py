"""C-V7 fixed abstention-margin runtime.

中文：C-V7 固定拒绝边界运行层。

The trained utility target remains {-1, 0, +1} and the network architecture is
unchanged.  Current keeps semantic utility zero, but the hard decision requires
a historical hypothesis to exceed the midpoint between neutral (0) and rescue
(+1):

    choose history k only if max_k u_k > 0.5

Otherwise the frozen C-V3 Current prediction is retained.  This is a fixed
semantic decision margin, not a learned gate and not an additional loss.
"""

import json
from pathlib import Path

import torch

from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v7_error_utility_estimator as c_v7,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v7_semantic_first as semantic_first,
)


ABSTENTION_MARGIN = 0.5
DECISION_RULE = "history iff max_k utility_k > 0.5; otherwise Current"


def apply_abstention_margin(selector_logits_full, margin=ABSTENTION_MARGIN):
    """Convert zero-reference utilities into fixed-margin decision scores.

    History channels are left unchanged.  The Current decision score is set to
    the fixed margin.  Because Current is channel zero, an exact tie at the
    margin resolves to Current under argmax, implementing the strict > rule.
    """
    if selector_logits_full.ndim != 4 or selector_logits_full.shape[1] < 2:
        raise ValueError("selector_logits_full must be [N,1+K,H,W] with K >= 1")
    current_score = torch.full_like(
        selector_logits_full[:, :1],
        float(margin),
    )
    return torch.cat((current_score, selector_logits_full[:, 1:]), dim=1)


def _protocol_from_argv(argv):
    argv = list(argv or [])
    if "--protocol" not in argv:
        return "full9"
    index = argv.index("--protocol")
    if index + 1 >= len(argv):
        raise ValueError("Missing value for --protocol")
    protocol = argv[index + 1].lower()
    if protocol not in {"dev3", "full9"}:
        raise ValueError("--protocol must be dev3 or full9")
    return protocol


def _inject_isolated_outputs(argv, protocol):
    argv = list(argv or [])
    if protocol == "dev3":
        output_default = c_v7.OUTPUT_DEFAULT + "_dev3_tau05"
        result_default = c_v7.RESULT_DEFAULT + "_dev3_tau05"
    else:
        output_default = c_v7.OUTPUT_DEFAULT + "_tau05"
        result_default = c_v7.RESULT_DEFAULT + "_tau05"
    if "--output" not in argv:
        argv.extend(["--output", output_default])
    if "--result-output" not in argv:
        argv.extend(["--result-output", result_default])
    return argv


def _rewrite_abstention_metadata(output_dir, result_dir, protocol):
    result_dir = Path(result_dir)
    paths = [
        result_dir / "oracle_precheck.json",
        *sorted(result_dir.glob("epoch_*.json")),
        result_dir / "summary.json",
    ]
    for path in paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        payload["abstention_margin"] = ABSTENTION_MARGIN
        payload["history_decision_rule"] = DECISION_RULE

        rows = []
        if path.name == "summary.json":
            architecture = payload.setdefault("architecture", {})
            architecture["current_utility_reference"] = 0.0
            architecture["current_decision_score"] = ABSTENTION_MARGIN
            architecture["abstention_margin"] = ABSTENTION_MARGIN
            architecture["history_decision_rule"] = DECISION_RULE
            selection_rule = payload.setdefault("selection_rule", {})
            selection_rule["history_activation"] = DECISION_RULE
            if isinstance(payload.get("best"), dict):
                rows.append(payload["best"])
            rows.extend(
                row for row in payload.get("history", []) if isinstance(row, dict)
            )
        elif path.name.startswith("epoch_"):
            rows.append(payload)
        elif path.name == "oracle_precheck.json":
            diagnostics = payload.get("diagnostics")
            if isinstance(diagnostics, dict):
                diagnostics["abstention_margin"] = ABSTENTION_MARGIN
                diagnostics["history_decision_rule"] = DECISION_RULE

        for row in rows:
            diagnostics = row.get("diagnostics")
            if isinstance(diagnostics, dict):
                diagnostics["current_utility"] = "fixed zero reference"
                diagnostics["current_decision_score"] = ABSTENTION_MARGIN
                diagnostics["abstention_margin"] = ABSTENTION_MARGIN
                diagnostics["history_decision_rule"] = DECISION_RULE
            train = row.get("train")
            if isinstance(train, dict):
                utility_training = train.get("utility_training")
                if isinstance(utility_training, dict):
                    utility_training["abstention_margin"] = ABSTENTION_MARGIN
                    utility_training["history_decision_rule"] = DECISION_RULE
        path.write_text(json.dumps(payload, indent=2))

    checkpoint_path = Path(output_dir) / "best.pt"
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu")
        payload["abstention_margin"] = ABSTENTION_MARGIN
        payload["history_decision_rule"] = DECISION_RULE
        architecture = payload.setdefault("architecture", {})
        architecture["current_utility_reference"] = 0.0
        architecture["current_decision_score"] = ABSTENTION_MARGIN
        architecture["abstention_margin"] = ABSTENTION_MARGIN
        architecture["history_decision_rule"] = DECISION_RULE
        torch.save(payload, checkpoint_path)


def main(argv=None):
    protocol = _protocol_from_argv(argv)
    forwarded = _inject_isolated_outputs(argv, protocol)
    output_dir = c_v7._arg_value(forwarded, "--output", c_v7.OUTPUT_DEFAULT)
    result_dir = c_v7._arg_value(
        forwarded,
        "--result-output",
        c_v7.RESULT_DEFAULT,
    )

    original_evidence = c_v7._selector_evidence

    def evidence_with_abstention(*args, **kwargs):
        evidence = original_evidence(*args, **kwargs)
        evidence = dict(evidence)
        evidence["selector_logits_full"] = apply_abstention_margin(
            evidence["selector_logits_full"],
            ABSTENTION_MARGIN,
        )
        return evidence

    try:
        c_v7._selector_evidence = evidence_with_abstention
        semantic_first.main(forwarded)
        _rewrite_abstention_metadata(output_dir, result_dir, protocol)
    finally:
        c_v7._selector_evidence = original_evidence


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
