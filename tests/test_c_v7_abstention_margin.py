import unittest

import torch

from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v7_abstention import (
    ABSTENTION_MARGIN,
    apply_abstention_margin,
)


class C_V7AbstentionMarginTest(unittest.TestCase):
    def test_margin_is_fixed_midpoint(self):
        self.assertEqual(ABSTENTION_MARGIN, 0.5)

    def test_history_requires_strictly_more_than_margin(self):
        # Channels are Current, t-1, t-2, t-3, t-4.
        logits = torch.full((1, 5, 1, 3), -1.0)
        logits[:, 0] = 0.0
        logits[:, 1, 0] = torch.tensor([0.49, 0.50, 0.51])

        adjusted = apply_abstention_margin(logits)
        selected = adjusted.argmax(1)[0, 0]

        self.assertTrue(torch.equal(selected, torch.tensor([0, 0, 1])))

    def test_history_utilities_are_not_modified(self):
        logits = torch.randn(2, 5, 3, 4)
        adjusted = apply_abstention_margin(logits)
        self.assertTrue(torch.equal(adjusted[:, 1:], logits[:, 1:]))
        self.assertTrue(
            torch.equal(
                adjusted[:, :1],
                torch.full_like(adjusted[:, :1], ABSTENTION_MARGIN),
            )
        )

    def test_invalid_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_abstention_margin(torch.zeros(1, 1, 3, 4))


if __name__ == "__main__":
    unittest.main()
