"""C-V6 formal entrypoint.

中文：C-V6 正式入口。

The first draft implementation was replaced after code review. The formal
implementation now lives in
`train_kitti_step_task_space_prior_c_v6_soft_temporal_correction_impl.py` and
includes probability-space history transport, corrected Protection/Persistence
supervision, near-zero gate initialization, zero-step equivalence checking, and
TBPTT boundary state refresh.
"""

from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v6_soft_temporal_correction_impl import (
    main,
)


if __name__ == "__main__":
    main()
