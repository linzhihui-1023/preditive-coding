"""Aligned launcher for C-V14 post-writeback reliability acceptance.

中文：C-V14 可靠性空间对齐版训练入口。

The fixed experiment protocol remains identical to C-V14. This launcher swaps
only the aligned training wrapper and aligned corrector class into the original
entrypoint so acceptance supervision, deployed gating, and diagnostics use the
same reliability field.
"""

from predify2021.mce_scores import (
    c_v14_post_writeback_reliability_training_aligned as training,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v14_post_writeback_reliability_acceptance
    as _base_entry,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_post_writeback_reliability_feature_correction_aligned import (
    AlignedPostWritebackReliabilityFeatureCorrector,
)


# Patch only the two implementation dependencies used by the original main().
_base_entry.training = training
_base_entry.PostWritebackReliabilityFeatureCorrector = (
    AlignedPostWritebackReliabilityFeatureCorrector
)

EXPERIMENT = _base_entry.EXPERIMENT
_selection_key = _base_entry._selection_key
_architecture_metadata = _base_entry._architecture_metadata
main = _base_entry.main


if __name__ == "__main__":
    main()
