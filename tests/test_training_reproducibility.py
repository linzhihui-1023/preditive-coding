import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs import (
    save_model_checkpoint,
    seed_everything,
)


class TrainingReproducibilityTest(unittest.TestCase):
    def test_seed_reproduces_all_random_sources(self):
        seed_everything(17)
        first = (random.random(), np.random.rand(), torch.rand(3))
        seed_everything(17)
        second = (random.random(), np.random.rand(), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_checkpoint_records_selection_metadata(self):
        model = torch.nn.Linear(2, 1)
        epoch_record = {"epoch": 3, "val": {"mean_temporal_loss": 0.25}}
        config = {"seed": 17}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_model_checkpoint(
                model,
                path,
                config,
                epoch_record,
                checkpoint_kind="best_val_temporal_loss",
            )
            checkpoint = torch.load(path, map_location="cpu")

        self.assertEqual(checkpoint["config"], config)
        self.assertEqual(checkpoint["selected_epoch"], epoch_record)
        self.assertEqual(checkpoint["checkpoint_kind"], "best_val_temporal_loss")


if __name__ == "__main__":
    unittest.main()
