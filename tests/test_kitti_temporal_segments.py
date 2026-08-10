import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

import predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs as training
from predify2021.mce_scores.kitti_pairs import KITTINextFramePairDataset


class _MotionTargetDataset(Dataset):
    def __init__(self, targets):
        self.targets = [torch.tensor(target, dtype=torch.float32) for target in targets]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.targets[index]

    def get_motion_target(self, index):
        return self.targets[index]


class KITTITemporalSegmentTest(unittest.TestCase):
    def test_fixed_dt_gap_creates_independent_stream_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drive = "synthetic_drive"
            camera_root = root / drive / "image_02"
            data_root = camera_root / "data"
            data_root.mkdir(parents=True)
            for index in range(6):
                (data_root / f"{index:010d}.png").touch()

            base = datetime(2011, 9, 26, 13, 0, 0)
            offsets = (0.0, 0.1035, 0.2070, 0.5000, 0.6035, 0.7070)
            timestamps = [
                (base + timedelta(seconds=offset)).strftime("%Y-%m-%d %H:%M:%S.%f")
                for offset in offsets
            ]
            (camera_root / "timestamps.txt").write_text("\n".join(timestamps) + "\n")

            dataset = KITTINextFramePairDataset(
                root,
                drive,
                fixed_dt_s=0.1035,
                dt_tolerance_s=0.001,
            )
            self.assertEqual(dataset.valid_start_indices, [0, 1, 3, 4])
            self.assertEqual(dataset.valid_sample_segments, ((0, 1), (2, 3)))

            with patch.object(training, "KITTI_ROOT", str(root)), patch.object(
                training, "KITTI_CAMERA", "image_02"
            ), patch.object(training, "FIXED_TS_S", 0.1035), patch.object(
                training, "FIXED_TS_TOL_S", 0.001
            ), patch.object(training, "TASK_ALIGNED_TARGET", ""):
                records = training._make_stream_sequence_records("train", drive, 0)

            self.assertEqual([len(record["dataset"]) for record in records], [2, 2])
            self.assertEqual(len(records), 2)


class MotionNormalizationTest(unittest.TestCase):
    def test_training_only_per_horizon_statistics_round_trip(self):
        train_dataset = _MotionTargetDataset(
            [
                [[1.0, 0.1], [2.0, 0.2]],
                [[3.0, 0.3], [6.0, 0.6]],
            ]
        )
        training_records = [{"dataset": train_dataset}]

        stats = training.compute_motion_target_stats(training_records)
        targets = torch.stack(train_dataset.targets)
        normalized = training.normalize_motion_targets(targets, stats)
        reconstructed = training.denormalize_motion_targets(normalized, stats)

        self.assertTrue(torch.allclose(stats["mean"], torch.tensor([[2.0, 0.2], [4.0, 0.4]])))
        self.assertTrue(torch.allclose(stats["std"], torch.tensor([[1.0, 0.1], [2.0, 0.2]])))
        self.assertTrue(torch.allclose(normalized.mean(dim=0), torch.zeros(2, 2), atol=1e-6))
        self.assertTrue(torch.allclose(normalized.std(dim=0, unbiased=False), torch.ones(2, 2)))
        self.assertTrue(torch.allclose(reconstructed, targets))


if __name__ == "__main__":
    unittest.main()
