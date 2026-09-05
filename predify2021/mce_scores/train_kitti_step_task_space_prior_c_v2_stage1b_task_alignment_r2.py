"""C-V2 Stage 1B-2 current mainline: bounded task-alignment residual.

This keeps the Stage 1B-1 Motion Observer as the source of geometric motion and
re-trains only the recurrent residual-motion layer with a structural low-resolution
bound of +/-2 pixels:

    M_hat_{t+1} = M_t^{observer} + delta_M_{t+1}^{task}
    delta_M_{t+1}^{task} = 2 * tanh(Head_M(H_t))

The previous +/-16 Stage 1B-2 experiment is kept unchanged for historical
reproducibility. This entry point is the current architecture definition and
intentionally does not expose --max-residual-low: the bound is structural, not a
hyperparameter sweep.

The reused legacy Stage-1B trainer selects checkpoints using a candidate-level
rule that can prefer Observer-Lagged. That is not suitable for this entry point,
because downstream Role/Mask training needs the best learned bounded residual
itself. After training, this wrapper therefore re-selects best.pt by the
Observer-Residual validation result (mIoU first, mTC as tie-break) and stamps the
checkpoint with an explicit bounded-task-alignment contract.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v2_stage1b_residual as stage1b2,
)


MAX_TASK_ALIGNMENT_RESIDUAL_LOW = 2.0
TASK_ALIGNMENT_ROLE = "bounded_task_alignment"
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v2_stage1b_task_alignment_r2"
)
RESULT_DEFAULT = (
    "results/kitti_step_task_space_prior_c_v2_stage1b_task_alignment_r2"
)


def _reject_residual_bound_override(argv):
    for token in argv:
        if token == "--max-residual-low" or token.startswith("--max-residual-low="):
            raise ValueError(
                "Stage 1B task-alignment residual is structurally fixed at +/-2 "
                "low-resolution pixels; --max-residual-low is not configurable."
            )


def _resolve_output_paths(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    known, _ = parser.parse_known_args(argv)
    return Path(known.output), Path(known.result_output)


def _reselect_task_alignment_checkpoint(output, result_output):
    rows = []
    for path in sorted(result_output.glob("epoch_*.json")):
        row = json.loads(path.read_text())
        metrics = row["metrics"]["metrics"]["observer_residual"]
        rows.append(
            (
                float(metrics["mIoU"]),
                float(metrics["mTC"]),
                int(row["epoch"]),
                row,
            )
        )
    if not rows:
        raise RuntimeError("No Stage 1B-2 epoch results were produced")

    _, _, best_epoch, best_row = max(rows, key=lambda x: (x[0], x[1]))
    epoch_checkpoint = output / f"epoch_{best_epoch:03d}.pt"
    if not epoch_checkpoint.exists():
        raise RuntimeError(f"Missing selected checkpoint: {epoch_checkpoint}")

    payload = torch.load(epoch_checkpoint, map_location="cpu", weights_only=False)
    architecture = payload.setdefault("architecture", {})
    bound = float(architecture.get("max_residual_low", float("nan")))
    if bound != MAX_TASK_ALIGNMENT_RESIDUAL_LOW:
        raise RuntimeError(
            f"Selected checkpoint reports max_residual_low={bound}, expected 2.0"
        )
    architecture["residual_role"] = TASK_ALIGNMENT_ROLE
    architecture["max_task_alignment_residual_low"] = MAX_TASK_ALIGNMENT_RESIDUAL_LOW
    architecture["geometric_motion_source"] = "frozen Stage 1B-1 Motion Observer"
    architecture["task_alignment_formula"] = (
        "M_hat_t_plus_1 = M_t_observer + delta_M_t_plus_1_task"
    )
    payload["task_alignment_selection"] = {
        "criterion": "observer_residual validation mIoU; mTC tie-break",
        "epoch": best_epoch,
        "metrics": best_row["metrics"]["metrics"]["observer_residual"],
    }
    torch.save(payload, output / "best.pt")

    summary_path = result_output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["legacy_stage1b_candidate_selection"] = summary.get("best")
    summary["best"] = {
        "epoch": best_epoch,
        "selected_candidate": "observer_residual",
        "metrics": best_row["metrics"],
        "stage1b2_checks": best_row["stage1b2_checks"],
        "selection_rule": "observer_residual validation mIoU; mTC tie-break",
    }
    summary["task_alignment_contract"] = {
        "residual_role": TASK_ALIGNMENT_ROLE,
        "max_task_alignment_residual_low": MAX_TASK_ALIGNMENT_RESIDUAL_LOW,
        "geometric_motion_source": "frozen Stage 1B-1 Motion Observer",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return best_epoch


def main(argv=None):
    user_argv = list(sys.argv[1:] if argv is None else argv)
    _reject_residual_bound_override(user_argv)
    output, result_output = _resolve_output_paths(user_argv)

    # Reuse the verified Stage 1B-2 causal training/evaluation path while fixing
    # only the architecture decision that changed: residual motion is now a
    # bounded task-alignment correction rather than a second large motion field.
    stage1b2.main(
        [
            "--max-residual-low",
            str(MAX_TASK_ALIGNMENT_RESIDUAL_LOW),
            "--output",
            OUTPUT_DEFAULT,
            "--result-output",
            RESULT_DEFAULT,
            *user_argv,
        ]
    )
    best_epoch = _reselect_task_alignment_checkpoint(output, result_output)
    print(
        json.dumps(
            {
                "task_alignment_best_epoch": best_epoch,
                "task_alignment_bound_low": MAX_TASK_ALIGNMENT_RESIDUAL_LOW,
                "checkpoint": str(output / "best.pt"),
                "result": str(result_output / "summary.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
