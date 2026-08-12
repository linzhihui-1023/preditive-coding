import unittest
from collections import deque

import torch

from predify2021.mce_scores.evaluate_kitti_prediction_error_separability import (
    _phase,
    binary_auroc,
    causal_error_statistics,
    summarize,
    validate_checkpoint,
)


class PredictionErrorStatisticTest(unittest.TestCase):
    def test_phase_counts_are_exactly_transition_based(self):
        phases = [_phase(index, baseline_frames=40, shift_frames=80) for index in range(150)]
        self.assertEqual(phases.count("baseline"), 40)
        self.assertEqual(phases.count("disturbance"), 80)
        self.assertEqual(phases.count("recovery"), 30)

    def test_causal_statistics_use_only_available_error_window(self):
        errors = deque(maxlen=2)
        errors.append(torch.tensor([1.0, 0.0]))
        first = causal_error_statistics(errors, None, ema_alpha=0.25, rolling_window=2)
        errors.append(torch.tensor([0.0, 1.0]))
        second = causal_error_statistics(
            errors,
            first["prediction_error_l2_ema"],
            ema_alpha=0.25,
            rolling_window=2,
        )

        self.assertAlmostEqual(first["prediction_error_l2"], 1.0)
        self.assertIsNone(first["prediction_error_cosine_previous"])
        self.assertAlmostEqual(first["prediction_error_temporal_variance"], 0.0)
        self.assertAlmostEqual(second["prediction_error_l2_ema"], 1.0)
        self.assertAlmostEqual(second["prediction_error_cosine_previous"], 0.0)
        self.assertAlmostEqual(second["prediction_error_temporal_variance"], 0.25)

    def test_binary_auroc_handles_order_and_ties(self):
        self.assertEqual(binary_auroc([3.0, 2.0, 1.0, 0.0], [1, 1, 0, 0]), 1.0)
        self.assertEqual(binary_auroc([1.0, 1.0], [1, 0]), 0.5)


class SeparabilitySummaryTest(unittest.TestCase):
    def _rows(self):
        rows = []
        values = {
            "calibration": {"persistent_blur": 1.0, "clean": 3.0, "iid_noise": 4.0},
            "validation": {"persistent_blur": 2.0, "clean": 5.0, "iid_noise": 6.0},
        }
        for drive, conditions in values.items():
            for condition, value in conditions.items():
                for _ in range(2):
                    rows.append(
                        {
                            "drive": drive,
                            "condition": condition,
                            "phase": "disturbance",
                            "classification_included": True,
                            "classification_label": int(condition == "persistent_blur"),
                            **{metric: value for metric in (
                                "prediction_error_l2",
                                "prediction_error_l2_ema",
                                "prediction_error_cosine_previous",
                                "prediction_error_temporal_variance",
                            )},
                        }
                    )
        return rows

    def test_direction_is_selected_on_calibration_and_frozen_for_validation(self):
        result = summarize(self._rows(), "calibration", "validation")
        self.assertTrue(
            all(direction == -1 for direction in result[
                "score_directions_selected_on_calibration_drive"
            ].values())
        )
        self.assertEqual(
            result["heldout_selected_metric_joint_negative_auroc"],
            1.0,
        )
        self.assertEqual(result["decision"], "go_promising")


class SeparabilityCheckpointTest(unittest.TestCase):
    def _checkpoint(self):
        return {
            "config": {
                "prediction_task": "future_feature",
                "future_feature_history_mode": "temporal_error",
                "future_feature_prediction_form": "current_residual",
                "future_feature_predictor_kernel_size": 1,
                "git_revision": "revision",
            },
            "checkpoint_kind": "best_val_future_feature_mse",
            "selected_epoch": {"epoch": 2},
        }

    def test_accepts_exact_temporal_error_checkpoint_contract(self):
        result = validate_checkpoint(self._checkpoint(), "revision")
        self.assertEqual(result["history_mode"], "temporal_error")

    def test_rejects_non_temporal_error_history(self):
        checkpoint = self._checkpoint()
        checkpoint["config"]["future_feature_history_mode"] = "none"
        with self.assertRaisesRegex(ValueError, "temporal_error"):
            validate_checkpoint(checkpoint, "revision")


if __name__ == "__main__":
    unittest.main()
