"""C-V2 Stage 1B Role-Supervision vs Spatial-Mask on the bounded r=2 motion base.

Both semantic candidates must share a Stage 1B-2 checkpoint whose task-alignment
residual is structurally bounded to +/-2 low-resolution pixels. Old +/-16
residual-motion checkpoints are rejected explicitly. The checkpoint must also
carry the bounded-task-alignment role contract written by the r=2 trainer.
"""

import argparse
import sys

import torch

from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask as role_vs_mask,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_task_alignment_r2 import (
    MAX_TASK_ALIGNMENT_RESIDUAL_LOW,
    TASK_ALIGNMENT_ROLE,
)


RESIDUAL_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v2_stage1b_task_alignment_r2/best.pt"
)
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask_r2"
)
RESULT_DEFAULT = (
    "results/kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask_r2"
)


def _validate_bounded_motion_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != "c_v2_stage1b2_residual_motion":
        raise RuntimeError("Checkpoint is not a Stage 1B-2 residual-motion checkpoint")
    architecture = payload.get("architecture", {})
    bound = float(architecture.get("max_residual_low", float("nan")))
    role = architecture.get("residual_role")
    explicit_task_bound = float(
        architecture.get("max_task_alignment_residual_low", float("nan"))
    )
    if bound != MAX_TASK_ALIGNMENT_RESIDUAL_LOW:
        raise RuntimeError(
            "Role/Mask r=2 mainline requires max_residual_low=2.0, "
            f"but checkpoint reports {bound}. Re-train the bounded task-alignment "
            "layer first; old +/-16 checkpoints are not accepted."
        )
    if role != TASK_ALIGNMENT_ROLE or explicit_task_bound != MAX_TASK_ALIGNMENT_RESIDUAL_LOW:
        raise RuntimeError(
            "Checkpoint has the right numeric bound but is not stamped as the current "
            "bounded-task-alignment mainline. Re-run the r=2 task-alignment trainer."
        )
    return payload


def main(argv=None):
    user_argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--residual-checkpoint", default=RESIDUAL_CHECKPOINT_DEFAULT
    )
    known, remaining = parser.parse_known_args(user_argv)
    _validate_bounded_motion_checkpoint(known.residual_checkpoint)

    role_vs_mask.main(
        [
            "--residual-checkpoint",
            known.residual_checkpoint,
            "--output",
            OUTPUT_DEFAULT,
            "--result-output",
            RESULT_DEFAULT,
            *remaining,
        ]
    )


if __name__ == "__main__":
    main()
