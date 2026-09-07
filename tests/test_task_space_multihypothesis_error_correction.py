import inspect
import unittest

import torch

from predify2021.model_factory.deeplabv3plus_resnet50.task_space_multihypothesis_error_correction import (
    MultiHypothesisErrorDirectCorrection,
)


class MultiHypothesisErrorDirectCorrectionTest(unittest.TestCase):
    def _inputs(self, module, height=6, width=8, valid_value=1.0):
        n, c = 1, module.num_classes
        errors = [
            torch.randn(n, c, height, width)
            for _ in range(module.history_length)
        ]
        dynamics = torch.randn(n, c, height, width)
        current_probability = torch.softmax(torch.randn(n, c, height, width), dim=1)
        current_margin = torch.rand(n, 1, height, width)
        transportability = torch.rand(n, 1, height, width)
        reliability = torch.rand(n, 1, height, width)
        validities = [
            torch.full((n, 1, height, width), float(valid_value))
            for _ in range(module.history_length)
        ]
        return (
            errors,
            dynamics,
            current_probability,
            current_margin,
            transportability,
            reliability,
            validities,
        )

    def test_zero_step_is_exactly_zero_19d_correction(self):
        module = MultiHypothesisErrorDirectCorrection(
            num_classes=19,
            history_length=4,
        )
        row = module(*self._inputs(module), previous_error_state=None)
        self.assertEqual(tuple(row["delta_logits"].shape), (1, 19, 6, 8))
        self.assertEqual(float(row["delta_logits"].abs().max()), 0.0)
        self.assertEqual(tuple(row["error_state"].shape), (1, 32, 6, 8))

    def test_all_invalid_history_structurally_forces_zero_correction(self):
        module = MultiHypothesisErrorDirectCorrection(
            num_classes=19,
            history_length=4,
        )
        # Make the readout intentionally nonzero. The structural validity mask
        # must still force DeltaL=0 everywhere.
        with torch.no_grad():
            module.delta_head.bias.fill_(2.0)
        row = module(
            *self._inputs(module, valid_value=0.0),
            previous_error_state=None,
        )
        self.assertEqual(float(row["delta_logits"].abs().max()), 0.0)
        self.assertEqual(float(row["any_history_valid"].abs().max()), 0.0)

    def test_forward_has_no_raw_history_probability_argument(self):
        parameters = inspect.signature(
            MultiHypothesisErrorDirectCorrection.forward
        ).parameters
        self.assertIn("prediction_errors", parameters)
        self.assertIn("current_probability", parameters)
        forbidden = {
            "history_probability",
            "history_probabilities",
            "candidate_probability",
            "candidate_probabilities",
            "history_logits",
        }
        self.assertTrue(forbidden.isdisjoint(parameters.keys()))

    def test_gradient_reaches_error_encoder_when_readout_is_active(self):
        module = MultiHypothesisErrorDirectCorrection(
            num_classes=3,
            history_length=2,
            hidden_channels=8,
            current_state_channels=8,
            branch_channels=8,
        )
        with torch.no_grad():
            module.delta_head.weight.fill_(0.01)
        row = module(*self._inputs(module, height=5, width=7), previous_error_state=None)
        loss = row["delta_logits"].square().mean()
        loss.backward()
        error_grad = module.error_pre[0].weight.grad
        gate_grad = module.error_recurrent.gates.weight.grad
        candidate_grad = module.error_recurrent.candidate.weight.grad
        self.assertIsNotNone(error_grad)
        self.assertIsNotNone(gate_grad)
        self.assertIsNotNone(candidate_grad)
        self.assertGreater(float(error_grad.abs().sum()), 0.0)
        self.assertGreater(
            float(gate_grad.abs().sum() + candidate_grad.abs().sum()),
            0.0,
        )

    def test_previous_error_state_shape_is_preserved(self):
        module = MultiHypothesisErrorDirectCorrection(
            num_classes=3,
            history_length=2,
            hidden_channels=8,
            current_state_channels=8,
            branch_channels=8,
        )
        previous = torch.randn(1, 8, 5, 7)
        row = module(
            *self._inputs(module, height=5, width=7),
            previous_error_state=previous,
        )
        self.assertEqual(tuple(row["error_state"].shape), tuple(previous.shape))


if __name__ == "__main__":
    unittest.main()
