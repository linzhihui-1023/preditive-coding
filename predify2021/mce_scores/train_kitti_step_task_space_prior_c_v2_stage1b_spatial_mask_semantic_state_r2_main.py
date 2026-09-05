"""Current Stage 1B mainline wrapper for calibrated Spatial Mask + carried Semantic State.

This wrapper keeps the validated training/evaluation implementation in
``train_kitti_step_task_space_prior_c_v2_stage1b_spatial_mask_semantic_state_r2``
but replaces only the semantic-correction-state warp semantics.

Temporal-prior logits intentionally use history persistence for out-of-bounds
samples. Semantic Correction State must not: an invalid historical source means
there is no correction state to carry into the newly visible location. Therefore
invalid carried-state samples are exactly zero and current Semantic Innovation is
responsible for rebuilding correction there.
"""

import sys

import torch
from torch.nn import functional as F

from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v2_stage1b_spatial_mask_semantic_state_r2 as base,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    low_flow_grid,
)


def _update_semantic_state_zero_invalid(
    previous_state,
    pending_motion,
    transportability,
    innovation,
):
    """Carry correction state only from geometrically valid historical sources."""
    if previous_state is None:
        previous_state = torch.zeros_like(innovation)
    if previous_state.shape[-2:] != pending_motion.shape[-2:]:
        raise ValueError("Semantic state and pending motion must share spatial size")
    if transportability.shape[-2:] != innovation.shape[-2:]:
        raise ValueError("Transportability and semantic innovation must share spatial size")

    grid, valid = low_flow_grid(pending_motion)
    carried_state = F.grid_sample(
        previous_state.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    carried_state = carried_state * valid.unsqueeze(1).to(carried_state.dtype)
    carried_state = carried_state.to(previous_state.dtype)

    state = transportability * carried_state + (1.0 - transportability) * innovation
    return state, carried_state


def main(argv=None):
    # The base module resolves this global at call time in both training and
    # evaluation, so one replacement keeps the two paths mathematically identical.
    base._update_semantic_state = _update_semantic_state_zero_invalid
    base.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
