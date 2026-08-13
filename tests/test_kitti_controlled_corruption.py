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
    apply_controlled_corruption,
    compute_controlled_recovery_metrics,
)
from predify2021.mce_scores.evaluate_kitti_same_drive_controlled_corruption import (
    build_independent_trajectory_specs,
    compute_history_utilization,
    summarize_predictor_input_weights,
    validate_controlled_checkpoint,
)
from predify2021.mce_scores.kitti_pairs import (
    KITTINextFramePairDataset,
    build_same_drive_train_val_test_subsets,
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
    def test_three_way_split_has_disjoint_chronological_raw_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drive = _make_synthetic_drive(root, frame_count=120)
            dataset = KITTINextFramePairDataset(root, drive)

            train, val, test, metadata = build_same_drive_train_val_test_subsets(
                dataset,
            )
            train_frames = collect_raw_frame_indices(train)
            val_frames = collect_raw_frame_indices(val)
            test_frames = collect_raw_frame_indices(test)

            self.assertEqual(train_frames, frozenset(range(72)))
            self.assertEqual(val_frames, frozenset(range(72, 96)))
            self.assertEqual(test_frames, frozenset(range(96, 120)))
            self.assertTrue(train_frames.isdisjoint(val_frames))
            self.assertTrue(train_frames.isdisjoint(test_frames))
            self.assertTrue(val_frames.isdisjoint(test_frames))
            self.assertEqual(metadata["train_sample_count"], 71)
            self.assertEqual(metadata["val_sample_count"], 23)
            self.assertEqual(metadata["test_sample_count"], 23)
            self.assertEqual(metadata["excluded_boundary_sample_count"], 2)
            self.assertEqual(metadata["chronological_order"], ("train", "val", "test"))
            self.assertFalse(metadata["shuffle"])

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
            trajectory="step_hold_recovery",
            baseline_frames=0,
            transition_frames=1,
            hold_frames=3,
            recovery_frames=1,
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
                    corruption_type="iid_gaussian",
                    bias_rgb=(0.0, 0.0, 0.0),
                    noise_std=0.08,
                    seed=17,
                ),
            )

            _, first_future, _, _ = dataset[0]
            second_current, _, _, _ = dataset[1]

            self.assertTrue(torch.equal(first_future[0], second_current))

    def test_blurred_absolute_frame_is_identical_as_future_then_current(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drive = _make_synthetic_drive(root, frame_count=5, pixel_value=128)
            dataset = ControlledCorruptionKITTIDataset(
                root,
                drive,
                schedule=self._schedule(),
                corruption_config=ControlledCorruptionConfig(
                    corruption_type="gaussian_blur",
                    blur_kernel_size=11,
                    blur_sigma=3.0,
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

    def test_step_schedule_has_independent_step_hold_and_recovery_phases(self):
        schedule = self._schedule()
        phases = [schedule.phase_and_severity(index)[0] for index in range(5)]
        severities = [schedule.phase_and_severity(index)[1] for index in range(5)]

        self.assertEqual(
            phases,
            [
                "step_change",
                "persistent_bias",
                "persistent_bias",
                "persistent_bias",
                "recovery",
            ],
        )
        self.assertEqual(severities, [1.0, 1.0, 1.0, 1.0, 0.0])

    def test_bias_and_iid_noise_are_separate_corruption_types(self):
        image = torch.full((3, 8, 8), 0.5)
        bias_config = ControlledCorruptionConfig(
            corruption_type="bias",
            bias_rgb=(0.1, 0.0, 0.0),
            noise_std=0.0,
        )
        noise_config = ControlledCorruptionConfig(
            corruption_type="iid_gaussian",
            bias_rgb=(0.0, 0.0, 0.0),
            noise_std=0.1,
            seed=3,
        )

        biased = apply_controlled_corruption(
            image, 1.0, bias_config, "drive", "image_02", "000.png"
        )
        noisy = apply_controlled_corruption(
            image, 1.0, noise_config, "drive", "image_02", "000.png"
        )

        self.assertTrue(torch.allclose(biased[0], torch.full((8, 8), 0.6)))
        self.assertTrue(torch.equal(biased[1:], image[1:]))
        self.assertFalse(torch.equal(noisy, biased))

    def test_gaussian_blur_is_deterministic_and_not_identity(self):
        image = torch.zeros(3, 17, 17)
        image[:, 8, 8] = 1.0
        config = ControlledCorruptionConfig(
            corruption_type="gaussian_blur",
            blur_kernel_size=11,
            blur_sigma=3.0,
        )

        first = apply_controlled_corruption(
            image, 1.0, config, "drive", "image_02", "000.png"
        )
        second = apply_controlled_corruption(
            image, 1.0, config, "drive", "image_02", "000.png"
        )

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, image))
        self.assertGreater(first[:, 8, 7].mean().item(), 0.0)

    def test_independent_trajectory_specs_each_fill_validation_segment(self):
        specs = build_independent_trajectory_specs(
            validation_raw_frame_count=46,
            baseline_frames=10,
            recovery_frames=18,
            ramp_frames=8,
            bias_rgb=(0.15, -0.08, 0.05),
            noise_std=0.03,
            seed=0,
        )

        self.assertEqual(set(specs), {"step_bias", "ramp_bias", "iid_noise"})
        self.assertTrue(
            all(spec["schedule"].total_frames == 46 for spec in specs.values())
        )
        self.assertEqual(specs["step_bias"]["corruption"].corruption_type, "bias")
        self.assertEqual(specs["ramp_bias"]["corruption"].noise_std, 0.0)
        self.assertEqual(
            specs["iid_noise"]["corruption"].corruption_type,
            "iid_gaussian",
        )


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
        self.assertEqual(metrics["peak_signed_excess_mse"], 4.0)
        self.assertEqual(metrics["recovery_time_frames"], 1)
        self.assertAlmostEqual(metrics["recovery_time_s"], 0.1)
        self.assertAlmostEqual(metrics["signed_excess_auec_mse_seconds"], 0.925)
        self.assertAlmostEqual(metrics["positive_excess_auec_mse_seconds"], 0.925)

    def test_signed_excess_preserves_negative_overcompensation(self):
        phases = ("baseline", "step_change", "recovery", "recovery")
        corrupted = (1.0, 2.0, 0.8, 1.0)
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
            recovery_fraction=0.25,
            recovery_consecutive_frames=1,
        )

        self.assertAlmostEqual(records[2]["signed_excess_feature_mse"], -0.2)
        self.assertAlmostEqual(records[2]["positive_excess_feature_mse"], 0.0)
        self.assertAlmostEqual(metrics["minimum_signed_excess_mse"], -0.2)


class HistoryUtilizationDiagnosticTest(unittest.TestCase):
    def test_reports_state_scale_and_counterfactual_history_contribution(self):
        model = torch.nn.Module()
        model.future_feature_history_mode = "temporal_error"
        model.future_feature_predictor = torch.nn.Sequential(
            torch.nn.Conv2d(4, 2, kernel_size=1, bias=False)
        )
        with torch.no_grad():
            model.future_feature_predictor[0].weight.zero_()
            model.future_feature_predictor[0].weight[0, 2, 0, 0] = 1.0
            model.future_feature_predictor[0].weight[1, 3, 0, 0] = 1.0

        current = torch.ones(1, 2, 2, 2)
        history = torch.full_like(current, 0.25)
        predicted_delta = model.future_feature_predictor(
            torch.cat([current, history], dim=1)
        )
        outputs = {
            "current_top": current,
            "history_top": history,
            "predicted_delta_top": predicted_delta,
            "future_top_target": current + predicted_delta,
        }

        utilization = compute_history_utilization(model, outputs)
        weights = summarize_predictor_input_weights(model)

        self.assertTrue(utilization["history_utilization_applicable"])
        self.assertAlmostEqual(utilization["current_feature_rms"], 1.0)
        self.assertAlmostEqual(utilization["history_input_rms"], 0.25)
        self.assertAlmostEqual(utilization["history_contribution_rms"], 0.25)
        self.assertAlmostEqual(utilization["zero_history_feature_mse"], 0.0625)
        self.assertAlmostEqual(utilization["history_feature_mse_change"], -0.0625)
        self.assertGreater(weights["history_input_weight_rms"], 0.0)
        self.assertEqual(weights["feature_input_weight_rms"], 0.0)


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
