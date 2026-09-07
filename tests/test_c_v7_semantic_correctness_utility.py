import unittest
from pathlib import Path

import torch

from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v7_semantic_first import (
    DEV3_EPOCH_SEGMENTS,
    DEV3_FRAMES_PER_SEQUENCE,
    DEV3_TRAIN,
    DEV3_VAL,
    SEMANTIC_TIE_DELTA,
    _dev3_epoch_groups,
    _dev3_sequence_groups,
    build_semantic_correctness_utility_targets,
)


class C_V7SemanticCorrectnessUtilityTest(unittest.TestCase):
    @staticmethod
    def _row(predictions, num_classes=3, valid=None):
        # Build logits whose argmax follows predictions. Exact probabilities are
        # irrelevant to the semantic-correctness utility target.
        height, width = predictions.shape
        logits = torch.full((1, num_classes, height, width), -4.0)
        for y in range(height):
            for x in range(width):
                logits[0, int(predictions[y, x]), y, x] = 4.0
        if valid is None:
            valid = torch.ones(height, width, dtype=torch.bool)
        return {"logits": logits, "valid_full": valid.unsqueeze(0)}

    def test_utility_is_exact_rescue_damage_or_neutral(self):
        # GT:        [0, 1, 2]
        # Current:   [1, 1, 2] -> wrong, correct, correct
        # History-1: [0, 2, 2] -> correct, wrong, correct
        # Utility:   [+1, -1, 0]
        gt = torch.tensor([[0, 1, 2]], dtype=torch.long)
        current = self._row(torch.tensor([[1, 1, 2]]))["logits"]
        history = self._row(torch.tensor([[0, 2, 2]]))

        target = build_semantic_correctness_utility_targets(
            current,
            [history],
            1,
            gt,
            previous_cv3_logits=None,
            teacher_full=None,
        )
        gains = target["semantic_gains"][0, 0, 0]
        self.assertTrue(torch.equal(gains, torch.tensor([1.0, -1.0, 0.0])))

    def test_invalid_history_is_not_supervised(self):
        gt = torch.tensor([[0, 1]], dtype=torch.long)
        current = self._row(torch.tensor([[1, 1]]))["logits"]
        history = self._row(
            torch.tensor([[0, 2]]),
            valid=torch.tensor([[True, False]]),
        )
        target = build_semantic_correctness_utility_targets(
            current,
            [history],
            1,
            gt,
            previous_cv3_logits=None,
            teacher_full=None,
        )
        self.assertTrue(
            torch.equal(
                target["valid_mask"][0, 0, 0],
                torch.tensor([True, False]),
            )
        )
        self.assertTrue(
            torch.equal(
                target["semantic_gains"][0, 0, 0],
                torch.tensor([1.0, 0.0]),
            )
        )

    def test_probability_improvement_cannot_override_hard_correctness(self):
        # Current predicts GT=class 0 correctly with p(GT)=0.40.
        # History has a higher p(GT)=0.45 but predicts class 1, so utility MUST
        # be -1 rather than positive.
        gt = torch.tensor([[0]], dtype=torch.long)
        current_prob = torch.tensor([0.40, 0.35, 0.25]).view(1, 3, 1, 1)
        history_prob = torch.tensor([0.45, 0.50, 0.05]).view(1, 3, 1, 1)
        current_logits = current_prob.log()
        history_logits = history_prob.log()
        history = {
            "logits": history_logits,
            "valid_full": torch.ones(1, 1, 1, dtype=torch.bool),
        }
        target = build_semantic_correctness_utility_targets(
            current_logits,
            [history],
            1,
            gt,
            previous_cv3_logits=None,
            teacher_full=None,
        )
        self.assertEqual(float(target["semantic_gains"][0, 0, 0, 0]), -1.0)

    def test_dev3_train_and_validation_sequences_are_disjoint(self):
        self.assertEqual(DEV3_VAL, ("0002", "0010", "0018"))
        self.assertEqual(len(DEV3_TRAIN), 12)
        self.assertTrue(set(DEV3_TRAIN).isdisjoint(DEV3_VAL))
        self.assertEqual(DEV3_FRAMES_PER_SEQUENCE, 96)
        self.assertEqual(DEV3_EPOCH_SEGMENTS, ("front", "middle", "back"))
        self.assertLess(SEMANTIC_TIE_DELTA, 1.0)

    def test_dev3_epoch_windows_are_front_middle_back(self):
        # Synthetic 200-frame sequences make all three windows distinct.
        groups = {
            sequence: [
                {"frame_id": f"{index:06d}"}
                for index in range(200)
            ]
            for sequence in DEV3_TRAIN
        }
        starts = []
        for epoch_index, expected_segment in enumerate(DEV3_EPOCH_SEGMENTS):
            sampled, metadata = _dev3_epoch_groups(groups, epoch_index)
            self.assertEqual(metadata["segment"], expected_segment)
            self.assertEqual(set(sampled.keys()), set(DEV3_TRAIN))
            self.assertTrue(
                all(len(samples) == DEV3_FRAMES_PER_SEQUENCE for samples in sampled.values())
            )
            starts.append(int(sampled[DEV3_TRAIN[0]][0]["frame_id"]))
        self.assertEqual(starts, [0, 52, 104])

    def test_dev3_split_router_keeps_official_train_and_val_separate(self):
        class DummyDataset:
            def __init__(self, split):
                self.mask_root = Path("/tmp") / split

        train_groups = {sequence: [sequence] for sequence in DEV3_TRAIN}
        val_groups = {sequence: [sequence] for sequence in DEV3_VAL}

        def original_sequence_groups(dataset):
            if dataset.mask_root.name == "train":
                return train_groups
            return val_groups

        routed_train = _dev3_sequence_groups(
            DummyDataset("train"),
            original_sequence_groups,
        )
        routed_val = _dev3_sequence_groups(
            DummyDataset("val"),
            original_sequence_groups,
        )
        self.assertEqual(tuple(routed_train.keys()), DEV3_TRAIN)
        self.assertEqual(tuple(routed_val.keys()), DEV3_VAL)


if __name__ == "__main__":
    unittest.main()
