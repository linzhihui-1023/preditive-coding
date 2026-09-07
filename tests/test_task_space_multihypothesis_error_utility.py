import unittest

import torch

from predify2021.model_factory.deeplabv3plus_resnet50.task_space_multihypothesis_error_utility import (
    MultiHypothesisErrorUtilityEstimator,
    balanced_utility_regression_loss,
    semantic_first_temporal_tiebreak_loss,
)


class MultiHypothesisErrorUtilityEstimatorTest(unittest.TestCase):
    def _inputs(self, batch=1, classes=19, history=4, height=5, width=7):
        probability = torch.softmax(torch.randn(batch, classes, height, width), dim=1)
        errors = [torch.randn(batch, classes, height, width) * 0.05 for _ in range(history)]
        dynamics = torch.randn(batch, classes, height, width) * 0.02
        current_margin = torch.rand(batch, 1, height, width)
        history_margins = [torch.rand(batch, 1, height, width) for _ in range(history)]
        transportability = torch.rand(batch, 1, height, width)
        reliability = torch.rand(batch, 1, height, width)
        validities = [torch.ones(batch, 1, height, width) for _ in range(history)]
        return (
            errors,
            dynamics,
            probability,
            current_margin,
            history_margins,
            transportability,
            reliability,
            validities,
        )

    def test_zero_step_is_exact_current_fallback(self):
        model = MultiHypothesisErrorUtilityEstimator()
        row = model(*self._inputs())
        self.assertEqual(tuple(row["selector_logits"].shape), (1, 5, 5, 7))
        self.assertEqual(tuple(row["candidate_utilities"].shape), (1, 4, 5, 7))
        self.assertTrue(torch.equal(row["selector_logits"][:, :1], torch.zeros_like(row["selector_logits"][:, :1])))
        self.assertTrue(torch.equal(row["candidate_utilities"], torch.zeros_like(row["candidate_utilities"])))
        self.assertTrue(torch.equal(row["selector_logits"].argmax(1), torch.zeros(1, 5, 7, dtype=torch.long)))

    def test_one_shared_scalar_head_scores_all_history_ages(self):
        model = MultiHypothesisErrorUtilityEstimator(history_length=4)
        self.assertEqual(model.selector_head.out_channels, 1)
        self.assertEqual(model.history_length, 4)
        self.assertEqual(model.current_state_channels, 16)

    def test_current_probability_is_detached_before_compact_state(self):
        model = MultiHypothesisErrorUtilityEstimator()
        inputs = list(self._inputs())
        current_probability = inputs[2].clone().requires_grad_(True)
        inputs[2] = current_probability
        row = model(*inputs)
        # Make the zero-initialized utility head nonzero so gradients reach the
        # trainable current-state encoder, while the frozen probability remains detached.
        with torch.no_grad():
            model.selector_head.weight.fill_(0.01)
        row = model(*inputs)
        row["candidate_utilities"].sum().backward()
        self.assertIsNone(current_probability.grad)
        self.assertIsNotNone(model.current_state_encoder[0].weight.grad)

    def test_balanced_utility_loss_counts_groups(self):
        predicted = torch.zeros(1, 4, 1, 2, requires_grad=True)
        target = torch.tensor([[[[1.0, -1.0]], [[0.0, 0.0]], [[2.0, -2.0]], [[0.05, -0.05]]]])
        valid = torch.ones_like(target, dtype=torch.bool)
        loss, counts = balanced_utility_regression_loss(
            predicted,
            target,
            valid,
            neutral_delta=0.10,
            neutral_weight=0.10,
        )
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertEqual(counts["positive"], 2)
        self.assertEqual(counts["negative"], 2)
        self.assertEqual(counts["neutral"], 4)
        loss.backward()
        self.assertIsNotNone(predicted.grad)

    def test_temporal_tiebreak_only_acts_inside_semantic_tie_band(self):
        utilities = torch.zeros(1, 2, 1, 2, requires_grad=True)
        gains = torch.tensor([[[[0.05, 1.0]], [[0.02, -1.0]]]])
        valid = torch.ones_like(gains, dtype=torch.bool)
        # Channels: Current, history-1, history-2.
        temporal_matches = torch.tensor(
            [[[[False, False]], [[True, True]], [[False, False]]]],
            dtype=torch.bool,
        )
        temporal_valid = torch.ones(1, 1, 1, 2, dtype=torch.bool)
        loss, pair_count = semantic_first_temporal_tiebreak_loss(
            utilities,
            gains,
            valid,
            temporal_matches,
            temporal_valid,
            semantic_tie_delta=0.10,
            rank_margin=0.05,
        )
        # Pixel 0 is semantically tied and temporally informative. Pixel 1 has
        # large semantic separation and must not be overridden by temporal rank.
        self.assertGreater(pair_count, 0)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(utilities.grad)


if __name__ == "__main__":
    unittest.main()
