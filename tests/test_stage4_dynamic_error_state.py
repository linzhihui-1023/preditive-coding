import unittest
from collections import Counter

import torch

from predify2021.mce_scores.evaluate_kitti_stage4_dynamic_error_state import (
    CONDITIONS,
    SCORE_FIELDS,
    build_counterbalanced_severity_sequences,
    build_nonoverlapping_windows,
    summarize,
    temporal_organization,
    update_error_score_states,
)


class DynamicErrorStateSequenceTest(unittest.TestCase):
    def setUp(self):
        self.levels = (0.25, 0.5, 0.75, 1.0)
        self.sequences = build_counterbalanced_severity_sequences(
            self.levels,
            frames_per_level=4,
            replicate_count=4,
            shuffle_seed=20260813,
        )

    def test_conditions_match_severity_multisets(self):
        expected = Counter({level: 4 for level in self.levels})
        for conditions in self.sequences.values():
            self.assertEqual(Counter(conditions["persistent"]), expected)
            self.assertEqual(Counter(conditions["shuffled"]), expected)

    def test_each_absolute_frame_is_counterbalanced(self):
        expected = Counter(self.levels)
        for condition in CONDITIONS:
            for frame_index in range(16):
                self.assertEqual(
                    Counter(
                        self.sequences[replicate][condition][frame_index]
                        for replicate in range(4)
                    ),
                    expected,
                )

    def test_persistent_has_longer_runs(self):
        persistent = temporal_organization(self.sequences[0]["persistent"])
        shuffled = temporal_organization(self.sequences[0]["shuffled"])
        self.assertEqual(persistent["maximum_run_length"], 4)
        self.assertLess(persistent["run_count"], shuffled["run_count"])
        self.assertGreater(
            persistent["mean_run_length"],
            shuffled["mean_run_length"],
        )
        self.assertGreater(
            persistent["adjacent_equal_fraction"],
            shuffled["adjacent_equal_fraction"],
        )


class DynamicErrorScoreTest(unittest.TestCase):
    def test_dynamic_matches_tensor_ema_at_unit_gain_but_not_scalar_envelope(self):
        sample_time = 0.1035
        time_constant = 0.5
        error_gain = 1.0
        integration_factor = sample_time / time_constant
        error = torch.tensor([[[[2.0, -2.0]]]])
        dynamic = torch.zeros_like(error) + integration_factor * (
            error - error_gain * torch.zeros_like(error)
        )
        stats, scalar, tensor = update_error_score_states(
            error,
            dynamic,
            previous_scalar_ema=0.0,
            previous_tensor_ema=None,
            ema_alpha=integration_factor,
        )
        self.assertTrue(torch.equal(tensor, dynamic))
        self.assertEqual(stats["dynamic_minus_matched_tensor_ema_max_abs"], 0.0)
        self.assertAlmostEqual(scalar, integration_factor * 2.0)
        self.assertAlmostEqual(stats["dynamic_error_state_rms"], scalar)

        opposite_error = -error
        opposite_dynamic = dynamic + integration_factor * (
            opposite_error - error_gain * dynamic
        )
        stats, scalar, tensor = update_error_score_states(
            opposite_error,
            opposite_dynamic,
            previous_scalar_ema=scalar,
            previous_tensor_ema=tensor,
            ema_alpha=alpha,
        )
        self.assertTrue(torch.allclose(tensor, opposite_dynamic))
        self.assertEqual(stats["dynamic_minus_matched_tensor_ema_max_abs"], 0.0)
        self.assertLess(stats["dynamic_error_state_rms"], scalar)


def _rows(condition, score_values):
    rows = []
    for index, values in enumerate(score_values):
        rows.append(
            {
                "drive": "drive",
                "corruption": "blur",
                "condition": condition,
                "replicate": 0,
                "classification_included": True,
                "classification_label": int(condition == "persistent"),
                "disturbance_index": index,
                "future_severity": 1.0,
                "matched_tensor_ema_error_rms": values[2],
                **dict(zip(SCORE_FIELDS, values)),
            }
        )
    return rows


class DynamicErrorWindowTest(unittest.TestCase):
    def test_windows_use_only_their_own_causal_rows(self):
        rows = _rows("persistent", [(index, index, index) for index in range(8)])
        original = build_nonoverlapping_windows(rows, 4, 8)
        for row in rows[4:]:
            for field in (*SCORE_FIELDS, "matched_tensor_ema_error_rms"):
                row[field] = 1000.0
        changed = build_nonoverlapping_windows(rows, 4, 8)
        for field in SCORE_FIELDS:
            self.assertEqual(original[0][field], changed[0][field])
            self.assertNotEqual(original[1][field], changed[1][field])

    def test_summary_compares_three_fixed_scores_without_selection(self):
        windows = []
        for condition, instant, simple, dynamic in (
            ("persistent", 2.0, 3.0, 5.0),
            ("shuffled", 1.0, 2.0, 1.0),
        ):
            for _ in range(2):
                windows.append(
                    {
                        "drive": "drive",
                        "corruption": "blur",
                        "condition": condition,
                        "classification_label": int(condition == "persistent"),
                        "instant_error_rms": instant,
                        "simple_scalar_ema_error_rms": simple,
                        "dynamic_error_state_rms": dynamic,
                    }
                )
        frame_rows = []
        for condition in CONDITIONS:
            frame_rows.append(
                {
                    "drive": "drive",
                    "corruption": "blur",
                    "condition": condition,
                    "phase": "disturbance",
                    "dynamic_formula_max_abs": 0.0,
                    "dynamic_minus_matched_tensor_ema_max_abs": 0.0,
                    **{field: 1.0 for field in SCORE_FIELDS},
                }
            )
        result = summarize(frame_rows, windows, ("drive",), ("blur",))
        self.assertTrue(result["no_score_or_direction_selection"])
        self.assertEqual(
            set(
                result["primary_aggregate"][
                    "direction_independent_separability_aurocs"
                ]
            ),
            set(SCORE_FIELDS),
        )

    def test_separability_does_not_rank_near_random_above_inverse_signal(self):
        windows = []
        for condition, instant, simple, dynamic in (
            ("persistent", 1.0, 1.0, 1.01),
            ("shuffled", 2.0, 2.0, 1.0),
        ):
            for _ in range(2):
                windows.append(
                    {
                        "drive": "drive",
                        "corruption": "blur",
                        "condition": condition,
                        "classification_label": int(condition == "persistent"),
                        "instant_error_rms": instant,
                        "simple_scalar_ema_error_rms": simple,
                        "dynamic_error_state_rms": dynamic,
                    }
                )
        frame_rows = [
            {
                "drive": "drive",
                "corruption": "blur",
                "condition": condition,
                "phase": "disturbance",
                "dynamic_formula_max_abs": 0.0,
                "dynamic_minus_matched_tensor_ema_max_abs": 0.0,
                **{field: 1.0 for field in SCORE_FIELDS},
            }
            for condition in CONDITIONS
        ]
        result = summarize(frame_rows, windows, ("drive",), ("blur",))
        primary = result["primary_aggregate"]
        self.assertEqual(primary["higher_is_persistent_aurocs"]["instant_error_rms"], 0.0)
        self.assertEqual(
            primary["direction_independent_separability_aurocs"][
                "instant_error_rms"
            ],
            1.0,
        )
        self.assertFalse(
            primary["dynamic_strictly_better_than_both_primary_controls"]
        )

    def test_summary_accepts_csv_string_labels(self):
        windows = []
        for condition, label, value in (
            ("persistent", "1", "2.0"),
            ("shuffled", "0", "1.0"),
        ):
            windows.append(
                {
                    "drive": "drive",
                    "corruption": "blur",
                    "condition": condition,
                    "classification_label": label,
                    **{field: value for field in SCORE_FIELDS},
                }
            )
        frame_rows = [
            {
                "drive": "drive",
                "corruption": "blur",
                "condition": condition,
                "phase": "disturbance",
                "dynamic_formula_max_abs": "0.0",
                "dynamic_minus_matched_tensor_ema_max_abs": "0.0",
                **{field: "1.0" for field in SCORE_FIELDS},
            }
            for condition in CONDITIONS
        ]
        result = summarize(frame_rows, windows, ("drive",), ("blur",))
        self.assertEqual(
            result["primary_aggregate"]["higher_is_persistent_aurocs"][
                "dynamic_error_state_rms"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
