import unittest

import torch

from predify2021.mce_scores.diagnose_kitti_feature_learnability import translate_feature
from predify2021.mce_scores.diagnose_kitti_local_motion import (
    candidate_shifts,
    compute_causal_historical_warp,
    compute_local_future_matching,
    estimate_local_displacement,
    forward_splat_discrete,
)
from predify2021.model_factory.targetflow.spatial_motion import (
    align_source_to_target,
)


class LocalMotionDiagnosticTest(unittest.TestCase):
    def test_candidate_order_prefers_zero_on_ties(self):
        self.assertEqual(candidate_shifts(2)[0], (0, 0))

    def test_pointwise_local_oracle_recovers_spatially_varying_matches(self):
        current = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])
        target = torch.tensor([[[[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]]]])
        result = compute_local_future_matching(
            current,
            target,
            radius=1,
            patch_size=1,
        )
        self.assertEqual(result["result_mse"].item(), 0.0)
        self.assertGreater(result["copy_mse"].item(), 0.0)

    def test_invalid_boundary_candidates_are_not_selected(self):
        source = torch.ones(1, 1, 3, 3)
        target = torch.zeros_like(source)
        match = estimate_local_displacement(source, target, radius=1, patch_size=1)
        self.assertTrue(torch.all(match["dy"] == 0))
        self.assertTrue(torch.all(match["dx"] == 0))

    def test_forward_splat_moves_features_and_copy_fills_holes(self):
        feature = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        dy = torch.zeros(1, 1, 4, dtype=torch.int64)
        dx = torch.ones(1, 1, 4, dtype=torch.int64)
        result = forward_splat_discrete(feature, dy, dx, radius=1)
        expected = torch.tensor([[[[1.0, 1.0, 2.0, 3.0]]]])
        torch.testing.assert_close(result["warped"], expected)
        self.assertAlmostEqual(result["coverage_fraction"].item(), 0.75)

    def test_causal_historical_warp_predicts_continued_translation(self):
        generator = torch.Generator().manual_seed(7)
        previous = torch.randn(1, 4, 7, 7, generator=generator)
        current = translate_feature(previous, dy=0, dx=1)
        future = translate_feature(current, dy=0, dx=1)
        result = compute_causal_historical_warp(
            previous,
            current,
            future,
            radius=1,
            patch_size=1,
        )
        self.assertLess(result["result_mse"].item(), result["copy_mse"].item())

    def test_alignment_places_previous_feature_in_current_coordinates(self):
        generator = torch.Generator().manual_seed(11)
        previous = torch.randn(1, 4, 7, 7, generator=generator)
        current = translate_feature(previous, dy=0, dx=1)

        result = align_source_to_target(
            previous,
            current,
            radius=1,
            patch_size=1,
        )

        raw_mse = (previous - current).square().mean()
        aligned_mse = (result["aligned_source"] - current).square().mean()
        self.assertLess(aligned_mse.item(), raw_mse.item())
        torch.testing.assert_close(
            result["aligned_source"][..., 1:],
            current[..., 1:],
        )


if __name__ == "__main__":
    unittest.main()
