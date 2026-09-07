"""Formal C-V6H entrypoint.

中文：C-V6H 分层误差选择器正式入口。训练逻辑位于
`train_kitti_step_task_space_prior_c_v6h_hierarchical_error_selector.py`。
本入口只补充最终结果元数据，使结构职责与实际代码完全一致。
"""

import json
import sys
from pathlib import Path

import predify2021.mce_scores.train_kitti_step_task_space_prior_c_v6h_hierarchical_error_selector as impl


def _arg_value(argv, name, default):
    args = list(sys.argv[1:] if argv is None else argv)
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix):]
    return default


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
        }
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
    result = impl.main(argv)
    _finalize_metadata(argv)
    return result


if __name__ == "__main__":
    main()
