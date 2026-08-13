import unittest

import torch
from torchvision.models import vgg16

from predify2021.model_factory.pvgg16_targetflow import PVGG16TargetFlow


class RealFramePredictiveCodingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.model = PVGG16TargetFlow(
            backbone=vgg16(weights=None),
            task="real_frame_pc",
            dynamic_error=True,
            error_state_mode="ema",
        ).eval()
        cls.frame1 = torch.randn(1, 3, 32, 32)
        cls.frame2 = torch.randn(1, 3, 32, 32)

    def setUp(self):
        self.model.reset()

    @staticmethod
    def _clone_states(states):
        return [
            {
                "representation": state.representation.clone(),
                "prediction": state.prediction.clone(),
                "instant_error": state.instant_error.clone(),
                "dynamic_error": state.dynamic_error.clone(),
            }
            for state in states
        ]

    def test_real_frame_mode_creates_no_future_predictor_and_freezes_parameters(self):
        self.assertIsNone(self.model.temporal_predictor)
        self.assertIsNone(self.model.future_feature_predictor)
        self.assertIsNone(self.model.temporal_fusion_module)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.model.parameters()))
        self.assertEqual(self.model.collect_learn_flow_losses(), ([], None))

    def test_r1_epsilon1_frame2_r2_epsilon2_chain(self):
        self.model.step_frame(self.frame1)
        first = self._clone_states(self.model.layer_states)

        self.assertEqual(self.model.real_frame_update_count, 1)
        for state in self.model.layer_states:
            self.assertEqual(state.frame_index, 1)
            self.assertIsNone(state.previous_representation)
            self.assertIsNone(state.previous_dynamic_error)
            self.assertTrue(
                torch.allclose(state.dynamic_error, 0.207 * state.instant_error)
            )

        self.model.step_frame(self.frame2)
        self.assertEqual(self.model.real_frame_update_count, 2)
        self.assertEqual(self.model.recurrence_outputs["updates_per_layer"], (1, 1, 1, 1, 1))

        for layer_index, state in enumerate(self.model.layer_states):
            self.assertEqual(state.frame_index, 2)
            self.assertTrue(
                torch.equal(
                    state.previous_representation,
                    first[layer_index]["representation"],
                )
            )
            self.assertTrue(
                torch.equal(
                    state.previous_prediction,
                    first[layer_index]["prediction"],
                )
            )
            self.assertTrue(
                torch.equal(
                    state.previous_dynamic_error,
                    first[layer_index]["dynamic_error"],
                )
            )
            expected_dynamic_error = (
                0.207 * state.instant_error
                + 0.793 * first[layer_index]["dynamic_error"]
            )
            self.assertTrue(torch.allclose(state.dynamic_error, expected_dynamic_error))

    def test_second_frame_uses_only_previous_frame_feedback_and_error_correction(self):
        self.model.step_frame(self.frame1)
        first = self._clone_states(self.model.layer_states)
        self.model.step_frame(self.frame2)

        for layer_index, state in enumerate(self.model.layer_states):
            previous_representation = first[layer_index]["representation"]
            ff_multiplier = self.model.pc_ff_multipliers[layer_index]
            fb_multiplier = self.model.pc_fb_multipliers[layer_index]
            error_multiplier = self.model.pc_error_multipliers[layer_index]
            expected_representation = previous_representation + ff_multiplier * (
                state.feedforward_drive - previous_representation
            )

            if layer_index + 1 < self.model.number_of_layers:
                expected_feedback = first[layer_index + 1]["prediction"]
                self.assertTrue(
                    torch.equal(state.previous_feedback_prediction, expected_feedback)
                )
                expected_representation = expected_representation + fb_multiplier * (
                    expected_feedback - previous_representation
                )
            else:
                self.assertIsNone(state.previous_feedback_prediction)

            self.assertIsNotNone(state.error_correction)
            expected_representation = expected_representation - (
                error_multiplier * state.error_correction
            )
            self.assertTrue(
                torch.allclose(state.representation, expected_representation)
            )

    def test_future_inputs_are_rejected_before_provider_can_run(self):
        provider_called = False

        def forbidden_provider():
            nonlocal provider_called
            provider_called = True
            return torch.zeros(1)

        with self.assertRaisesRegex(ValueError, "accepts only the current observed frame"):
            self.model.step_frame(self.frame1, top_target_provider=forbidden_provider)
        self.assertFalse(provider_called)
        self.assertEqual(self.model.real_frame_update_count, 0)

    def test_memories_are_detached_and_parameters_do_not_change(self):
        before = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        self.model.step_frame(self.frame1)
        self.model.step_frame(self.frame2)

        memories = (
            self.model.representation_state_memory
            + self.model.prediction_state_memory
            + self.model.instant_error_state_memory
            + self.model.error_state_memory
        )
        self.assertTrue(all(memory is not None for memory in memories))
        self.assertTrue(all(not memory.requires_grad for memory in memories))
        self.assertTrue(all(memory.grad_fn is None for memory in memories))
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_reset_starts_a_new_segment(self):
        self.model.step_frame(self.frame1)
        self.model.step_frame(self.frame2)
        self.model.reset()

        self.assertEqual(self.model.real_frame_update_count, 0)
        self.assertIsNone(self.model.recurrence_outputs)
        self.assertTrue(
            all(memory is None for memory in self.model.representation_state_memory)
        )
        self.assertTrue(all(memory is None for memory in self.model.prediction_state_memory))
        self.assertTrue(all(memory is None for memory in self.model.error_state_memory))

        self.model.step_frame(self.frame2)
        self.assertEqual(self.model.real_frame_update_count, 1)
        self.assertTrue(
            all(state.previous_dynamic_error is None for state in self.model.layer_states)
        )


if __name__ == "__main__":
    unittest.main()
