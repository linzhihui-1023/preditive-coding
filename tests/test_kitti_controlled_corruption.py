import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import torch
from PIL import Image

from predify2021.mce_scores.kitti_controlled_corruption import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ControlledCorruptionConfig,
    ControlledCorruptionKITTIDataset,
    ControlledCorruptionSchedule,
    compute_controlled_recovery_metrics,
)
from predify2021.mce_scores.evaluate_kitti_same_drive_controlled_corruption import (
    validate_controlled_checkpoint,
)
from predify2021.mce_scores.kitti_pairs import (
    KITTINextFramePairDataset,
    build_same_drive_train_val_subsets,
    collect_raw_frame_indices,
)


def _make_synthetic_drive(root, frame_count, pixel_value=None):
    drive = "synthetic_drive"
    camera_root = root / drive / "image_02"
    data_root = camera_root / "data"
    data_root.mkdir(parents=True)
    for index in range(frame_count):
        path = data_root / f"{index:010d}.png"
        if pixel_value is None:
            path.touch()
        else:
            Image.new("RGB", (32, 24), color=(pixel_value,) * 3).save(path)

    base = datetime(2011, 9, 26, 13, 0, 0)
    timestamps = [
        (base + timedelta(seconds=0.1 * index)).strftime("%Y-%m-%d %H:%M:%S.%f")
        for index in range(frame_count)
    ]
    (camera_root / "timestamps.txt").write_text("\n".join(timestamps) + "\n")
    return drive


class SameDriveRawFrameSplitTest(unittest.TestCase):
    def test_same_drive_train_val_have_no_shared_raw_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drive = _make_synthetic_drive(root, frame_count=120)
            dataset = KITTINextFramePairDataset(root, drive)

            train, val, metadata = build_same_drive_train_val_subsets(
                dataset,
                train_fraction=0.6,
                val_fraction=0.2,
                gap_frames=20,
            )
            train_frames = collect_raw_frame_indices(train)
            val_frames = collect_raw_frame_indices(val)

            self.assertTrue(train_frames.isdisjoint(val_frames))
            self.assertEqual(train_frames, frozenset(range(72)))
            self.assertEqual(val_frames, frozenset(range(92, 116)))
            self.assertEqual(metadata["actual_gap_frames"], 20)
            self.assertEqual(metadata["shared_raw_frame_count"], 0)


class ControlledCorruptionDatasetTest(unittest.TestCase):
    def _schedule(self):
        return ControlledCorruptionSchedule(
            baseline_frames=0,
            step_frames=1,
            ramp_frames=1,
            persistent_frames=2,
            recovery_frames=1,
            step_level=1.0,
        )

    def test_same_absolute_frame_is_identical_as_future_then_current(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drive = _make_synthetic_drive(root, frame_count=5, pixel_value=128)
            dataset = ControlledCorruptionKITTIDataset(
                root,
                drive,
                schedule=self._schedule(),
                corruption_config=ControlledCorruptionConfig(
                    bias_rgb=(0.05, -0.02, 0.01),
                    noise_std=0.08,
                    seed=17,
                ),
            )

            _, first_future, _, _ = dataset[0]
            second_current, _, _, _ = dataset[1]

            self.assertTrue(torch.equal(first_future[0], second_current))

    def test_corruption_is_applied_before_imagenet_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drive = _make_synthetic_drive(root, frame_count=5, pixel_value=128)
            dataset = ControlledCorruptionKITTIDataset(
                root,
                drive,
                schedule=self._schedule(),
                corruption_config=ControlledCorruptionConfig(
                    bias_rgb=(0.1, 0.1, 0.1),
                    noise_std=0.0,
                ),
            )

            current, _, _, _ = dataset[0]
            expected_pixel = 128.0 / 255.0 + 0.1
            expected = torch.tensor(
                [
                    (expected_pixel - mean) / std
                    for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)
                ]
            )

            self.assertTrue(
                torch.allclose(current.mean(dim=(1, 2)), expected, atol=1e-6)
            )

    def test_disabled_corruption_matches_standard_kitti_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drive = _make_synthetic_drive(root, frame_count=5, pixel_value=128)
            standard = KITTINextFramePairDataset(root, drive)
            controlled = ControlledCorruptionKITTIDataset(
                root,
                drive,
                schedule=self._schedule(),
                corruption_enabled=False,
            )

            standard_current, standard_future, _, _ = standard[0]
            controlled_current, controlled_future, _, _ = controlled[0]

            self.assertTrue(torch.equal(standard_current, controlled_current))
            self.assertTrue(torch.equal(standard_future, controlled_future[0]))

    def test_schedule_has_step_ramp_persistent_and_recovery_phases(self):
        schedule = self._schedule()
        phases = [schedule.phase_and_severity(index)[0] for index in range(5)]
        severities = [schedule.phase_and_severity(index)[1] for index in range(5)]

        self.assertEqual(
            phases,
            [
                "step_change",
                "ramp_change",
                "persistent_bias",
                "persistent_bias",
                "recovery",
            ],
        )
        self.assertEqual(severities, [1.0, 1.0, 1.0, 1.0, 0.0])


class ControlledRecoveryMetricTest(unittest.TestCase):
    def test_peak_recovery_time_and_auec_use_per_frame_curve(self):
        phases = (
            "baseline",
            "baseline",
            "step_change",
            "persistent_bias",
            "recovery",
            "recovery",
            "recovery",
        )
        corrupted = (1.0, 1.0, 5.0, 4.0, 3.0, 1.2, 1.05)
        records = [
            {
                "future_phase": phase,
                "future_raw_frame_index": index,
                "clean_feature_mse": 1.0,
                "corrupted_feature_mse": value,
            }
            for index, (phase, value) in enumerate(zip(phases, corrupted))
        ]

        metrics = compute_controlled_recovery_metrics(
            records,
            sample_time_s=0.1,
            recovery_fraction=0.1,
            recovery_consecutive_frames=2,
        )

        self.assertEqual(metrics["peak_error_mse"], 5.0)
        self.assertEqual(metrics["peak_excess_mse"], 4.0)
        self.assertEqual(metrics["recovery_time_frames"], 1)
        self.assertAlmostEqual(metrics["recovery_time_s"], 0.1)
        self.assertAlmostEqual(metrics["excess_auec_mse_seconds"], 0.925)


class ControlledCheckpointValidationTest(unittest.TestCase):
    def _checkpoint(self):
        split = {
            "raw_frame_count": 120,
            "train_raw_frame_range": (0, 71),
            "val_raw_frame_range": (92, 115),
            "minimum_gap_frames": 20,
            "train_sample_indices": tuple(range(71)),
            "val_sample_indices": tuple(range(92, 115)),
        }
        return {
            "config": {
                "prediction_task": "future_feature",
                "future_feature_history_mode": "temporal_error",
                "future_feature_predictor_kernel_size": 1,
                "same_drive_split": True,
                "same_drive_drive": "synthetic_drive",
                "same_drive_split_metadata": split,
                "train_pairs": 71,
                "val_pairs": 23,
            },
            "state_dict": {
                "future_feature_predictor.0.weight": torch.empty(1, 1, 1, 1)
            },
        }, split

    def test_checkpoint_must_match_same_drive_split(self):
        checkpoint, split = self._checkpoint()
        validation = validate_controlled_checkpoint(
            checkpoint,
            drive="synthetic_drive",
            split_metadata=split,
        )
        self.assertEqual(validation["history_mode"], "temporal_error")

        checkpoint["config"]["same_drive_split"] = False
        with self.assertRaisesRegex(ValueError, "SAME_DRIVE_SPLIT"):
            validate_controlled_checkpoint(
                checkpoint,
                drive="synthetic_drive",
                split_metadata=split,
            )


if __name__ == "__main__":
    unittest.main()
