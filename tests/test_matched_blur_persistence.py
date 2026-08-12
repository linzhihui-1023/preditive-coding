import unittest
from collections import Counter

from predify2021.mce_scores.evaluate_kitti_matched_blur_persistence import (
    CONDITIONS,
    build_counterbalanced_sigma_sequences,
    build_nonoverlapping_windows,
    summarize,
    temporal_organization,
)
from predify2021.mce_scores.evaluate_kitti_prediction_error_separability import (
    METRICS,
)
from predify2021.mce_scores.kitti_controlled_corruption import (
    ExplicitSeveritySchedule,
)


class MatchedSigmaSequenceTest(unittest.TestCase):
    def setUp(self):
        self.levels = (0.75, 1.5, 2.25, 3.0)
        self.sequences = build_counterbalanced_sigma_sequences(
            self.levels,
            frames_per_level=20,
            replicate_count=4,
            shuffle_seed=20260812,
        )

    def test_each_pair_has_exactly_the_same_sigma_multiset(self):
        expected = Counter({sigma: 20 for sigma in self.levels})
        for conditions in self.sequences.values():
            self.assertEqual(Counter(conditions["persistent_blur"]), expected)
            self.assertEqual(Counter(conditions["shuffled_blur"]), expected)

    def test_every_absolute_frame_is_counterbalanced_across_replicates(self):
        expected = Counter(self.levels)
        for condition in CONDITIONS:
            for frame_index in range(80):
                self.assertEqual(
                    Counter(
                        self.sequences[replicate][condition][frame_index]
                        for replicate in range(4)
                    ),
                    expected,
                )

    def test_only_temporal_organization_differs(self):
        persistent = temporal_organization(
            self.sequences[0]["persistent_blur"]
        )
        shuffled = temporal_organization(self.sequences[0]["shuffled_blur"])

        self.assertEqual(persistent["maximum_run_length"], 20)
        self.assertEqual(persistent["run_count"], 4)
        self.assertLessEqual(shuffled["maximum_run_length"], 3)
        self.assertGreater(shuffled["run_count"], 50)

    def test_exact_counterbalancing_rejects_wrong_replicate_count(self):
        with self.assertRaisesRegex(ValueError, "replicate_count"):
            build_counterbalanced_sigma_sequences(
                self.levels,
                frames_per_level=20,
                replicate_count=3,
                shuffle_seed=0,
            )


class ExplicitSeverityScheduleTest(unittest.TestCase):
    def test_explicit_absolute_frame_sequence_has_clean_edges(self):
        schedule = ExplicitSeveritySchedule(
            baseline_frames=3,
            disturbance_severities=(0.25, 0.5, 1.0),
            recovery_frames=2,
        )
        values = [schedule.phase_and_severity(index) for index in range(8)]
        self.assertEqual(
            values,
            [
                ("baseline", 0.0),
                ("baseline", 0.0),
                ("baseline", 0.0),
                ("disturbance", 0.25),
                ("disturbance", 0.5),
                ("disturbance", 1.0),
                ("recovery", 0.0),
                ("recovery", 0.0),
            ],
        )


def _metric_rows(drive, condition, replicate, values):
    rows = []
    for disturbance_index, value in enumerate(values):
        rows.append(
            {
                "drive": drive,
                "condition": condition,
                "replicate": replicate,
                "classification_included": True,
                "classification_label": int(condition == "persistent_blur"),
                "disturbance_index": disturbance_index,
                "future_blur_sigma": 1.0,
                "absolute_sigma_change": 0.0,
                **{metric: float(value) for metric in METRICS},
            }
        )
    return rows


class MatchedPersistenceWindowTest(unittest.TestCase):
    def test_nonoverlapping_window_does_not_read_later_errors(self):
        rows = _metric_rows("drive", "persistent_blur", 0, range(8))
        original = build_nonoverlapping_windows(
            rows,
            window_size=4,
            disturbance_transitions=8,
        )
        for row in rows[4:]:
            for metric in METRICS:
                row[metric] = 1000.0
        changed = build_nonoverlapping_windows(
            rows,
            window_size=4,
            disturbance_transitions=8,
        )

        for metric in METRICS:
            self.assertEqual(original[0][metric], changed[0][metric])
            self.assertNotEqual(original[1][metric], changed[1][metric])

    def test_direction_and_metric_are_frozen_from_calibration_drive(self):
        rows = []
        for drive in ("calibration", "validation"):
            rows.extend(_metric_rows(drive, "persistent_blur", 0, [1.0] * 8))
            rows.extend(_metric_rows(drive, "shuffled_blur", 0, [3.0] * 8))
        windows = build_nonoverlapping_windows(
            rows,
            window_size=4,
            disturbance_transitions=8,
        )
        result = summarize(rows, windows, "calibration", "validation")

        self.assertTrue(
            all(
                direction == -1
                for direction in result[
                    "score_directions_selected_on_calibration_drive"
                ].values()
            )
        )
        self.assertEqual(
            result["heldout_selected_metric_primary_window_auroc"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
