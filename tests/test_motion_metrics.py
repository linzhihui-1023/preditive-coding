import unittest

import torch

from predify2021.mce_scores.motion_metrics import compute_motion_diagnostics


class MotionDiagnosticsTest(unittest.TestCase):
    def test_reports_component_mse_and_absolute_error_quantiles(self):
        predictions = torch.tensor([[1.0, -1.0], [3.0, 4.0]])
        targets = torch.tensor([[0.0, 1.0], [1.0, 1.0]])

        metrics = compute_motion_diagnostics(
            predictions,
            targets,
            target_mean=torch.tensor([[0.0, 0.0]]),
            target_std=torch.tensor([[2.0, 1.0]]),
        )

        self.assertEqual(metrics["sample_count"], 2)
        self.assertAlmostEqual(metrics["forward_mse_m2"], 2.5)
        self.assertAlmostEqual(metrics["forward_median_ae_m"], 1.5)
        self.assertAlmostEqual(metrics["forward_p95_ae_m"], 1.95, places=5)
        self.assertAlmostEqual(metrics["yaw_mse_rad2"], 6.5)
        self.assertAlmostEqual(metrics["yaw_median_ae_rad"], 2.5)
        self.assertAlmostEqual(metrics["yaw_p95_ae_rad"], 2.95, places=5)
        self.assertAlmostEqual(metrics["standardized_joint_mse"], 3.5625)

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            compute_motion_diagnostics(torch.zeros(2, 2), torch.zeros(3, 2))


if __name__ == "__main__":
    unittest.main()
