"""C-V2 Stage 1B-2 current mainline: bounded task-alignment residual.

This keeps the Stage 1B-1 Motion Observer as the source of geometric motion and
re-trains only the recurrent residual-motion layer with a structural low-resolution
bound of +/-2 pixels:

    M_hat_{t+1} = M_t^{observer} + delta_M_{t+1}^{task}
    delta_M_{t+1}^{task} = 2 * tanh(Head_M(H_t))

The previous +/-16 Stage 1B-2 experiment is kept unchanged for historical
reproducibility.  This entry point is the current architecture definition and
intentionally does not expose --max-residual-low: the bound is structural, not a
hyperparameter sweep.
"""

import sys

from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v2_stage1b_residual as stage1b2,
)


MAX_TASK_ALIGNMENT_RESIDUAL_LOW = 2.0
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


def main(argv=None):
    user_argv = list(sys.argv[1:] if argv is None else argv)
    _reject_residual_bound_override(user_argv)

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


if __name__ == "__main__":
    main()
