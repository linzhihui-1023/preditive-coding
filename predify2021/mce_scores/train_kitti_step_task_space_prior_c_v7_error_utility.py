"""Canonical CLI entry for C-V7 Prediction-Error Candidate Utility Estimation."""

import sys

from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v7_abstention import (
    main,
)


if __name__ == "__main__":
    main(sys.argv[1:])
