"""Aligned C-V14 reliability representation.

中文：C-V14 可靠性空间对齐层。

The base C-V14 model applies reliability at c4 resolution. This subclass keeps
all model mathematics unchanged, but exposes ``reliability`` as the actually
deployed c4-resolution gate so training diagnostics use the same quantity that
multiplies the bounded semantic feature proposal. The original low-resolution
probability remains available as ``reliability_low``.
"""

from .task_space_post_writeback_reliability_feature_correction import (
    PostWritebackReliabilityFeatureCorrector,
)


class AlignedPostWritebackReliabilityFeatureCorrector(
    PostWritebackReliabilityFeatureCorrector
):
    """Expose the deployed c4 reliability map under the canonical key."""

    def forward(self, *args, **kwargs):
        row = super().forward(*args, **kwargs)
        row["reliability_low"] = row["reliability"]
        row["reliability"] = row["reliability_c4"]
        return row
