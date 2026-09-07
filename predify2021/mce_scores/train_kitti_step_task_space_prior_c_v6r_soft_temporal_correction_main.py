"""C-V6R formal entrypoint.

中文：C-V6R 正式入口。

C-V6R keeps the C-V6 K=4 probability-space history architecture and changes
only the training mechanisms directly implicated by the completed C-V6 run:
exact-zero trainable soft gate and correctness-conflict-only attention
supervision. C-V6 and its recorded results remain untouched for reproducibility.
"""

from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v6r_soft_temporal_correction_impl import (
    main,
)


if __name__ == "__main__":
    main()
