"""C-V6S formal entrypoint after static-review correctness fixes.

中文：C-V6S 正式入口。原始 72b96e 实现保留在
`train_kitti_step_task_space_prior_c_v6s_semantic_budget_refinement_impl.py`，
本入口调用 review-fix implementation。
"""

from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v6s_semantic_budget_refinement_reviewfix_impl import (
    main,
)


if __name__ == "__main__":
    main()
