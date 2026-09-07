import inspect
import unittest

import torch

from predify2021.model_factory.deeplabv3plus_resnet50.task_space_multihypothesis_error_selector import (
    MultiHypothesisErrorSelector,
    build_multihypothesis_error_evidence,
    strict_controller_validity,
)


def _probability(batch=1, classes=19, height=6, width=8):
    logits = torch.randn(batch, classes, height, width)
    return torch.softmax(logits, dim=1)


class MultiHypothesisErrorSelectorTest(unittest.TestCase):
    def test_controller_signature_excludes_raw_semantic_probabilities(self):
        parameters = inspect.signature(MultiHypothesisErrorSelector.forward).parameters
        self.assertNotIn("current_probability", parameters)
        self.assertNotIn("history_probabilities", parameters)
        self.assertIn("prediction_errors", parameters)

    def test_k4_input_channel_count_is_error_only_205(self):
        selector = MultiHypothesisErrorSelector(
            num_classes=19,
            history_length=4,
            hidden_channels=32,
        )
        self.assertEqual(selector.input_channels, 205)

    def test_multihypothesis_errors_are_validity_gated_and_detached(self):
        current = _probability().requires_grad_(True)
        histories = [_probability().requires_grad_(True) for _ in range(4)]
        validities = [torch.ones(1, 1, 6, 8) for _ in range(4)]
        validities[2][:, :, :, 4:] = 0.0

        evidence = build_multihypothesis_error_evidence(
            current,
            histories,
            validities,
        )

        expected = (current.detach() - histories[2].detach()) * validities[2]
        torch.testing.assert_close(evidence["prediction_errors"][2], expected)
        self.assertEqual(
            int(torch.count_nonzero(evidence["prediction_errors"][2][..., 4:])),
            0,
        )
        self.assertEqual(
            int(torch.count_nonzero(evidence["history_margins"][2][..., 4:])),
            0,
        )
        self.assertTrue(
            all(not error.requires_grad for error in evidence["prediction_errors"])
        )
        self.assertFalse(evidence["current_margin"].requires_grad)
        self.assertTrue(
            all(not margin.requires_grad for margin in evidence["history_margins"])
        )

    def test_strict_validity_requires_low_path_and_all_full_warp_pixels(self):
        low_path = torch.ones(1, 1, 2, 2, requires_grad=True)
        full_warp = torch.ones(1, 4, 4)
        # One invalid full-res pixel in the top-right 2x2 block invalidates
        # the corresponding low-res controller cell conservatively.
        full_warp[:, 0, 2] = 0.0
        full_warp.requires_grad_(True)

        valid = strict_controller_validity(low_path, full_warp, (2, 2))
        expected = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
        torch.testing.assert_close(valid, expected)
        self.assertFalse(valid.requires_grad)

        low_path_zero = torch.ones(1, 1, 2, 2)
        low_path_zero[:, :, 1, 0] = 0.0
        valid = strict_controller_validity(low_path_zero, torch.ones(1, 4, 4), (2, 2))
        self.assertEqual(float(valid[0, 0, 1, 0]), 0.0)

    def test_zero_initialized_selector_falls_back_to_current(self):
        selector = MultiHypothesisErrorSelector(
            num_classes=19,
            history_length=4,
            hidden_channels=32,
        )
        errors = [torch.randn(1, 19, 6, 8) for _ in range(4)]
        dynamics = torch.randn(1, 19, 6, 8)
        current_margin = torch.rand(1, 1, 6, 8)
        history_margins = [torch.rand(1, 1, 6, 8) for _ in range(4)]
        transportability = torch.rand(1, 1, 6, 8)
        reliability = torch.rand(1, 1, 6, 8)
        validities = [torch.ones(1, 1, 6, 8) for _ in range(4)]

        row = selector(
            errors,
            dynamics,
            current_margin,
            history_margins,
            transportability,
            reliability,
            validities,
        )

        self.assertEqual(tuple(row["selector_logits"].shape), (1, 5, 6, 8))
        self.assertEqual(int(torch.count_nonzero(row["selector_logits"])), 0)
        self.assertTrue(
            torch.equal(
                row["selector_logits"].argmax(dim=1),
                torch.zeros((1, 6, 8), dtype=torch.long),
            )
        )


if __name__ == "__main__":
    unittest.main()
