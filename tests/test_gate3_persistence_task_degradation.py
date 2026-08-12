import unittest

from predify2021.mce_scores.evaluate_kitti_gate3_persistence_task_degradation import (
    DETECTOR_SCORE,
    _association,
    _average_ranks,
    _pearson,
    build_windows_and_degradations,
    validate_gate2_reference,
    validate_motion_checkpoint,
    verify_gate2_trace,
)
from predify2021.mce_scores.evaluate_kitti_prediction_error_separability import (
    METRICS,
)


class Gate3ProtocolLockTest(unittest.TestCase):
    def _summary(self):
        return {
            "experiment": "prediction_error_matched_blur_persistence",
            "evaluation_git_revision": "gate2",
            "analysis": {
                "selected_metric": DETECTOR_SCORE,
                "score_directions_selected_on_calibration_drive": {
                    DETECTOR_SCORE: 1
                },
            },
            "protocol": {
                "primary_analysis_unit": "nonoverlapping 8-frame disturbance window",
                "baseline_transitions": 40,
                "disturbance_transitions": 80,
                "recovery_transitions": 30,
                "blur_kernel_size": 11,
                "sigma_levels": [0.75, 1.5, 2.25, 3.0],
                "frames_per_sigma_per_trajectory": 20,
                "replicate_count": 4,
                "shuffle_seed": 17,
            },
            "drives": {"calibration": "d0", "validation": "d1"},
        }

    def test_accepts_frozen_gate2_cosine_detector(self):
        result = validate_gate2_reference(self._summary(), "gate2")
        self.assertEqual(result["detector_score"], DETECTOR_SCORE)
        self.assertEqual(result["window_size"], 8)

    def test_rejects_posthoc_detector_reselection(self):
        summary = self._summary()
        summary["analysis"]["selected_metric"] = "prediction_error_l2"
        with self.assertRaisesRegex(ValueError, "cosine"):
            validate_gate2_reference(summary, "gate2")

    def test_gate2_trace_must_match_frame_by_frame(self):
        gate3 = []
        gate2 = []
        for condition in ("persistent_blur", "shuffled_blur"):
            current = {
                "drive": "d",
                "condition": condition,
                "replicate": 0,
                "stream_index": 0,
                "current_blur_sigma": 0.75,
                "future_blur_sigma": 1.5,
                **{metric: 0.1 for metric in METRICS},
            }
            gate3.append(current)
            gate2.append({key: str(value) for key, value in current.items()})
        audit = verify_gate2_trace(gate3, gate2)
        self.assertEqual(audit["matched_row_count"], 2)

        gate3[0][DETECTOR_SCORE] = 0.2
        with self.assertRaisesRegex(ValueError, "trace mismatch"):
            verify_gate2_trace(gate3, gate2)


class Gate3MotionCheckpointTest(unittest.TestCase):
    def _checkpoint(self):
        return {
            "config": {
                "git_revision": "motion",
                "task_aligned_target": "ego_motion",
                "motion_target_name": "longitudinal_yaw_2dof",
                "temporal_target_mode": "ego_motion",
                "target_flow_mode": "recursive",
                "error_state_mode": "ema",
                "reset_each_frame": False,
                "current_top_duplicate": False,
                "temporal_horizons": (1,),
                "motion_target_stats": {
                    "name": "longitudinal_yaw_2dof",
                    "mean": [[0.4, 0.0]],
                    "std": [[0.1, 0.01]],
                },
            },
            "checkpoint_kind": "best_val_temporal_loss",
            "selected_epoch": {"epoch": 6},
        }

    def test_accepts_formal_group_a_motion_contract(self):
        result = validate_motion_checkpoint(self._checkpoint(), "motion")
        self.assertEqual(result["selected_epoch"], 6)

    def test_rejects_future_feature_checkpoint_as_motion_model(self):
        checkpoint = self._checkpoint()
        checkpoint["config"]["task_aligned_target"] = "none"
        with self.assertRaisesRegex(ValueError, "task_aligned_target"):
            validate_motion_checkpoint(checkpoint, "motion")


def _frame(condition, replicate, index, score, forward_error, yaw_error):
    return {
        "drive": "drive",
        "condition": condition,
        "replicate": replicate,
        "phase": "disturbance",
        "disturbance_index": index,
        "future_blur_sigma": 0.0 if condition == "clean" else 1.0,
        "forward_absolute_error_m": forward_error,
        "yaw_absolute_error_rad": yaw_error,
        DETECTOR_SCORE: score,
    }


class Gate3WindowPairingTest(unittest.TestCase):
    def test_signed_degradation_uses_same_window_clean_counterfactual(self):
        rows = []
        rows.extend(_frame("clean", -1, i, 0.0, 1.0, 0.1) for i in range(8))
        rows.extend(
            _frame("persistent_blur", 0, i, 0.8, 1.5, 0.08)
            for i in range(8)
        )
        rows.extend(
            _frame("shuffled_blur", 0, i, -0.2, 0.75, 0.2)
            for i in range(8)
        )

        windows, degradation = build_windows_and_degradations(rows, 8, 8)
        self.assertEqual(len(windows), 3)
        by_condition = {row["condition"]: row for row in degradation}
        self.assertAlmostEqual(
            by_condition["persistent_blur"]["forward_mae_degradation_m"], 0.5
        )
        self.assertAlmostEqual(
            by_condition["persistent_blur"]["yaw_mae_degradation_rad"], -0.02
        )
        self.assertAlmostEqual(
            by_condition["shuffled_blur"]["forward_mae_degradation_m"], -0.25
        )
        self.assertAlmostEqual(
            by_condition["shuffled_blur"]["yaw_mae_degradation_rad"], 0.1
        )

    def test_window_score_reads_exactly_eight_current_rows(self):
        rows = []
        rows.extend(_frame("clean", -1, i, 0.0, 1.0, 0.1) for i in range(16))
        rows.extend(
            _frame("persistent_blur", 0, i, float(i), 1.0, 0.1)
            for i in range(16)
        )
        windows, _ = build_windows_and_degradations(rows, 8, 16)
        persistent = [row for row in windows if row["condition"] == "persistent_blur"]
        self.assertEqual(persistent[0]["persistence_score"], 3.5)
        self.assertEqual(persistent[1]["persistence_score"], 11.5)


class Gate3AssociationTest(unittest.TestCase):
    def test_pearson_and_spearman_are_not_posthoc_oriented(self):
        self.assertAlmostEqual(_pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(_pearson([1, 2, 3], [6, 4, 2]), -1.0)
        self.assertEqual(_average_ranks([2, 1, 2]), [2.5, 1.0, 2.5])

    def test_association_preserves_signed_degradation(self):
        rows = [
            {
                "persistence_score": score,
                "persistence_score_excess_over_clean": score,
                "forward_mae_degradation_m": degradation,
            }
            for score, degradation in ((0.1, -1.0), (0.2, -0.5), (0.8, 0.5), (0.9, 1.0))
        ]
        result = _association(rows, "forward_mae_degradation_m")
        self.assertGreater(result["pearson_score_vs_signed_degradation"], 0.9)
        self.assertEqual(result["score_auroc_for_positive_degradation"], 1.0)
        self.assertEqual(result["positive_degradation_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
