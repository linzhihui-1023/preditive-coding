import torch

from predify2021.mce_scores.evaluate_kitti_step_semantic_direct_open_loop import (
    EXPECTED_EVALUATED_FRAMES,
)
from predify2021.mce_scores.train_kitti_step_semantic_direct_correction import (
    semantic_kl_loss,
)


def test_semantic_kl_is_zero_for_identical_teacher_and_prediction():
    logits = torch.randn(1, 19, 3, 4)
    assert semantic_kl_loss(logits, logits).abs().item() < 1e-6


def test_semantic_evaluator_protocol_is_fixed():
    assert EXPECTED_EVALUATED_FRAMES == 2963
