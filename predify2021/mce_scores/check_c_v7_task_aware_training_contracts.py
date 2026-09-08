"""Lightweight checks for C-V7 Task-Aware Rescue/Protection supervision.

中文：C-V7 任务感知 Rescue（救回）/Protection（保护）监督契约检查。

No KITTI-STEP, Host, checkpoint or RAFT is loaded.
"""

import torch

from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v7_task_aware_correction import (
    _build_task_masks,
    _task_aware_losses,
)


def _logits_from_labels(labels, num_classes=19, high=5.0, low=-5.0):
    height, width = labels.shape
    logits = torch.full((1, num_classes, height, width), low)
    logits.scatter_(1, labels.unsqueeze(0).unsqueeze(0), high)
    return logits


def main():
    # Four diagnostic pixels in one row:
    # x0: Current wrong, valid history correct -> Rescue.
    # x1: Current correct, valid history conflicts -> Protection.
    # x2: Current wrong, no history correct -> neither.
    # x3: Current correct, only invalid history conflicts -> neither.
    gt = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    current_labels = torch.tensor([[4, 1, 5, 3]], dtype=torch.long)
    current_logits = _logits_from_labels(current_labels)

    history1_labels = torch.tensor([[0, 6, 7, 8]], dtype=torch.long)
    history1_logits = _logits_from_labels(history1_labels).requires_grad_(True)
    history1_valid = torch.tensor([[[True, True, True, False]]])

    history2_labels = torch.tensor([[9, 1, 10, 11]], dtype=torch.long)
    history2_logits = _logits_from_labels(history2_labels).requires_grad_(True)
    history2_valid = torch.tensor([[[True, True, True, False]]])

    candidate_rows = [
        {"logits": history1_logits, "valid_full": history1_valid},
        {"logits": history2_logits, "valid_full": history2_valid},
    ]

    masks = _build_task_masks(current_logits, candidate_rows, gt)
    expected_rescue = torch.tensor([[True, False, False, False]])
    expected_protect = torch.tensor([[False, True, False, False]])
    if not torch.equal(masks["rescue"].cpu(), expected_rescue):
        raise RuntimeError(f"Unexpected Rescue mask: {masks['rescue'].cpu()}")
    if not torch.equal(masks["protect"].cpu(), expected_protect):
        raise RuntimeError(f"Unexpected Protection mask: {masks['protect'].cpu()}")

    # Make final logits trainable and deliberately worsen the protected pixel.
    final_logits = current_logits.clone().requires_grad_(True)
    with torch.no_grad():
        final_logits[0, 1, 0, 1] -= 1.0
        final_logits[0, 6, 0, 1] += 1.0

    seg, rescue, protect, task, diag = _task_aware_losses(
        current_logits,
        final_logits,
        gt,
        candidate_rows,
    )
    if diag["rescue_pixels"] != 1 or diag["protection_pixels"] != 1:
        raise RuntimeError(f"Unexpected task pixel counts: {diag}")
    if float(rescue.detach().item()) <= 0.0:
        raise RuntimeError("Rescue loss must be positive on an unimproved Rescue pixel")
    if float(protect.detach().item()) <= 0.0:
        raise RuntimeError("Protection loss must activate when a protected margin worsens")
    if not all(bool(torch.isfinite(value).item()) for value in (seg, rescue, protect, task)):
        raise RuntimeError("Task-aware loss contains NaN/Inf")

    task.backward()
    if final_logits.grad is None or float(final_logits.grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("Task-aware loss did not provide gradient to final logits")
    if history1_logits.grad is not None or history2_logits.grad is not None:
        raise RuntimeError("Training-only history correctness mask leaked gradient to history logits")

    # All-invalid history must never create Rescue or Protection roles.
    invalid_rows = [
        {
            "logits": history1_logits.detach(),
            "valid_full": torch.zeros_like(history1_valid),
        }
    ]
    invalid_masks = _build_task_masks(current_logits, invalid_rows, gt)
    if bool(invalid_masks["rescue"].any()) or bool(invalid_masks["protect"].any()):
        raise RuntimeError("Invalid history incorrectly created task-aware supervision")

    print(
        {
            "passed": True,
            "rescue_pixels": diag["rescue_pixels"],
            "protection_pixels": diag["protection_pixels"],
            "invalid_history_creates_task_mask": False,
            "final_logit_gradient_abs_sum": float(final_logits.grad.abs().sum().item()),
            "history_gradient": None,
            "history_age_target": False,
        }
    )


if __name__ == "__main__":
    main()
