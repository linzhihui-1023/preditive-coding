"""Validated entrypoint for Utility-on-E1.

Fixes the alpha* endpoint rule used by the base utility-from-E1 trainer. A
semantic state that has zero first-order benefit at alpha=0 must default to no
writeback (alpha*=0), not full writeback. Only pixels whose CE is strictly
decreasing at alpha=0 are eligible for partial/full semantic-state writeback.
"""

import sys

import torch
from torch.nn import functional as F

from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v2_stage1b_utility_from_e1_r2 as base,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_only import (
    IGNORE_LABEL,
    NUM_CLASSES,
)


@torch.no_grad()
def _optimal_writeback_target_strict(
    base_logits_low,
    semantic_state_low,
    current_gt_cpu,
):
    """CE-minimizing alpha in [0,1], with flat/no-benefit states mapped to zero."""
    if base_logits_low.shape != semantic_state_low.shape:
        raise ValueError("Base logits and Semantic State must share shape")

    target = base._lowres_target(
        current_gt_cpu, tuple(base_logits_low.shape[-2:])
    )
    valid = target != IGNORE_LABEL
    safe_target = target.clamp(0, NUM_CLASSES - 1)

    logits0 = base_logits_low.detach().float()
    state = semantic_state_low.detach().float()
    state_true = state.gather(1, safe_target.unsqueeze(1)).squeeze(1)

    def derivative(alpha):
        logits = logits0 + alpha.unsqueeze(1) * state
        probability = F.softmax(logits, dim=1)
        return (probability * state).sum(dim=1) - state_true

    zero = torch.zeros_like(target, dtype=logits0.dtype)
    one = torch.ones_like(zero)
    d0 = derivative(zero)
    d1 = derivative(one)

    alpha = torch.zeros_like(zero)
    improving = valid & (d0 < 0.0)
    full = improving & (d1 <= 0.0)
    interior = improving & (d1 > 0.0)
    alpha[full] = 1.0

    lo = torch.zeros_like(zero)
    hi = torch.ones_like(zero)
    for _ in range(base.BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        dm = derivative(mid)
        move_lo = interior & (dm < 0.0)
        move_hi = interior & ~move_lo
        lo = torch.where(move_lo, mid, lo)
        hi = torch.where(move_hi, mid, hi)

    alpha[interior] = 0.5 * (lo[interior] + hi[interior])
    return alpha.clamp_(0.0, 1.0), valid


def main(argv=None):
    base._optimal_writeback_target = _optimal_writeback_target_strict
    base.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
