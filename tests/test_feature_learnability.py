import unittest

import torch

from predify2021.mce_scores.diagnose_kitti_feature_learnability import (
    best_oracle_translation_mse,
    compute_feature_baselines,
    summarize_rows,
    translate_feature,
)


class FeatureLearnabilityTest(unittest.TestCase):
    def test_translate_feature_uses_zero_fill_without_wraparound(self):
        feature = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        translated = translate_feature(feature, dy=1, dx=-1)
        expected = torch.tensor([[[[0.0, 0.0], [2.0, 0.0]]]])
        torch.testing.assert_close(translated, expected)

    def test_oracle_recovers_known_integer_translation(self):
        current = torch.zeros(1, 1, 5, 5)
        current[0, 0, 2, 2] = 3.0
        target = translate_feature(current, dy=-1, dx=1)
        mse, dy, dx = best_oracle_translation_mse(current, target, max_shift_cells=1)
        torch.testing.assert_close(mse, torch.zeros_like(mse))
        self.assertEqual(dy.item(), -1)
        self.assertEqual(dx.item(), 1)

    def test_constant_velocity_is_causal_and_exact_for_linear_features(self):
        previous = torch.zeros(1, 1, 2, 2)
        current = torch.ones_like(previous)
        next_feature = 2.0 * torch.ones_like(previous)
        target = 4.0 * torch.ones_like(previous)
        metrics = compute_feature_baselines(
            previous,
            current,
            next_feature,
            target,
            horizon=3,
            max_shift_cells=0,
        )
        self.assertEqual(metrics["copy_mse"].item(), 9.0)
        self.assertEqual(metrics["velocity_mse"].item(), 0.0)
        self.assertEqual(metrics["copy_minus_velocity_over_copy"].item(), 1.0)
        self.assertAlmostEqual(metrics["adjacent_delta_cosine"].item(), 1.0)

    def test_summary_uses_ratio_of_mean_mse_for_matrix_result(self):
        rows = []
        for copy_mse, velocity_mse in ((2.0, 1.0), (6.0, 9.0)):
            rows.append(
                {
                    "split": "val",
                    "stage": "stage5",
                    "horizon_frames": 1,
                    "copy_mse": copy_mse,
                    "velocity_mse": velocity_mse,
                    "oracle_translation_mse": 1.0,
                    "copy_minus_velocity_mse": copy_mse - velocity_mse,
                    "copy_minus_velocity_over_copy": (copy_mse - velocity_mse) / copy_mse,
                    "copy_minus_oracle_mse": copy_mse - 1.0,
                    "copy_minus_oracle_over_copy": (copy_mse - 1.0) / copy_mse,
                    "adjacent_delta_cosine": 0.25,
                    "oracle_shift_dy_cells": 0,
                    "oracle_shift_dx_cells": 0,
                }
            )
        summary = summarize_rows(rows)["val/stage5/h1"]
        self.assertAlmostEqual(
            summary["ratio_of_mean_mse"]["copy_minus_velocity_over_copy"],
            -0.25,
        )
        self.assertAlmostEqual(summary["velocity_better_than_copy_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
