import unittest

import torch

from predify2021.mce_scores.evaluate_kitti_real_frame_pc_phase1 import (
    CONDITIONS,
    normalized_representation_distance,
    phase_for_frame,
    select_protocol_raw_frames,
    summarize_rows,
)


class RealFramePCPhase1Test(unittest.TestCase):
    def test_protocol_phases_are_exact(self):
        self.assertEqual(phase_for_frame(0), ("baseline", 0))
        self.assertEqual(phase_for_frame(39), ("baseline", 39))
        self.assertEqual(phase_for_frame(40), ("disturbance", 0))
        self.assertEqual(phase_for_frame(119), ("disturbance", 79))
        self.assertEqual(phase_for_frame(120), ("recovery", 0))
        self.assertEqual(phase_for_frame(159), ("recovery", 39))

    def test_frame_selection_includes_terminal_real_frame_once(self):
        class Dataset:
            valid_start_indices = tuple(range(170))
            valid_sample_segments = (tuple(range(170)),)

        selected = select_protocol_raw_frames(Dataset())
        self.assertEqual(selected, tuple(range(160)))
        self.assertEqual(len(set(selected)), 160)

    def test_normalized_representation_distance(self):
        clean = torch.tensor([[[[3.0, 4.0]]]])
        corrupted = torch.tensor([[[[6.0, 8.0]]]])
        self.assertAlmostEqual(
            normalized_representation_distance(corrupted, clean),
            1.0,
        )

    def test_go_requires_practical_both_drive_improvement_and_recovery(self):
        rows = []
        drive_names = (
            "2011_09_26/2011_09_26_drive_0011_sync",
            "2011_09_26/2011_09_26_drive_0039_sync",
        )
        disturbance_values = {
            "feedforward": 0.30,
            "pc_no_error": 0.20,
            "pc_dynamic_error": 0.18,
        }
        for condition in CONDITIONS:
            for drive in drive_names:
                for phase, frame_count in (
                    ("baseline", 40),
                    ("disturbance", 80),
                    ("recovery", 40),
                ):
                    for frame_index in range(frame_count):
                        if phase == "baseline":
                            distance = 0.0
                        elif phase == "disturbance":
                            distance = disturbance_values[condition]
                        else:
                            distance = 0.10 if frame_index < 10 else 0.02
                        for layer in range(1, 6):
                            rows.append(
                                {
                                    "condition": condition,
                                    "drive": drive,
                                    "phase": phase,
                                    "phase_frame_index": frame_index,
                                    "layer": layer,
                                    "representation_normalized_l2": distance,
                                    "instantaneous_prediction_error_rms": 1.0,
                                    "dynamic_error_rms": 1.0,
                                }
                            )

        summary = summarize_rows(rows)

        self.assertEqual(summary["decision"], "GO")
        self.assertTrue(summary["primary_comparison"]["passed"])
        self.assertTrue(summary["recovery_check"]["distance_decreased"])


if __name__ == "__main__":
    unittest.main()
