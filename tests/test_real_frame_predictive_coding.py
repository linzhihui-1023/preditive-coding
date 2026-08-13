import copy
import unittest

import torch
import torch.nn as nn
from predify.modules import PCoderN
from torchvision.models import vgg16

from predify2021.model_factory.pvgg16_targetflow import PVGG16TargetFlow


class PredictionOutputAdapter(nn.Module):
    def __init__(self, prediction_module):
        super().__init__()
        self.prediction_module = copy.deepcopy(prediction_module)

    def forward(self, representation):
        output = self.prediction_module(representation)
        return output[-1] if isinstance(output, tuple) else output


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

    def test_second_frame_error_correction_matches_pcoder_n_k_over_c_sqrt(self):
        self.model.step_frame(self.frame1)
        self.model.step_frame(self.frame2)

        for layer_index, state in enumerate(self.model.layer_states):
            prediction_module = PredictionOutputAdapter(
                self.model._prediction_module_for_layer(layer_index)
            )
            previous_representation = (
                state.previous_representation.detach().clone().requires_grad_(True)
            )
            previous_prediction = prediction_module(previous_representation)
            self.assertTrue(
                torch.allclose(previous_prediction, state.previous_prediction)
            )
            pseudo_target = (
                state.previous_prediction + state.previous_dynamic_error
            ).detach()
            correction_loss = nn.functional.mse_loss(
                previous_prediction,
                pseudo_target,
            )
            raw_gradient = torch.autograd.grad(
                correction_loss,
                previous_representation,
            )[0].detach()
            expected_error_scale = (
                state.previous_prediction.numel() / state.c_sqrt
            )
            expected_scaled_correction = expected_error_scale * raw_gradient

            self.assertTrue(torch.allclose(state.error_scale, expected_error_scale))
            self.assertTrue(
                torch.allclose(
                    state.error_correction,
                    expected_scaled_correction,
                    rtol=1e-5,
                    atol=1e-7,
                )
            )

            reference_pcoder = PCoderN(
                prediction_module,
                has_feedback=layer_index + 1 < self.model.number_of_layers,
                random_init=False,
            )
            reference_pcoder.rep = state.previous_representation.detach().clone()
            reference_pcoder.prd = state.previous_prediction.detach().clone()
            reference_pcoder.grd = raw_gradient
            reference_pcoder.prediction_error = correction_loss.detach()
            reference_pcoder.C_sqrt.copy_(state.c_sqrt)
            reference_representation, _ = reference_pcoder(
                ff=state.feedforward_drive,
                fb=state.previous_feedback_prediction,
                target=state.prediction_target,
                build_graph=False,
                ffm=self.model.pc_ff_multipliers[layer_index],
                fbm=self.model.pc_fb_multipliers[layer_index],
                erm=self.model.pc_error_multipliers[layer_index],
            )

            self.assertTrue(
                torch.allclose(
                    state.representation,
                    reference_representation,
                    rtol=1e-5,
                    atol=1e-7,
                ),
                f"Layer {layer_index + 1} does not match PCoderN scaling.",
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
        calibrated_c_sqrt = self.model.pc_error_c_sqrt.clone()
        self.model.reset()

        self.assertEqual(self.model.real_frame_update_count, 0)
        self.assertIsNone(self.model.recurrence_outputs)
        self.assertTrue(
            all(memory is None for memory in self.model.representation_state_memory)
        )
        self.assertTrue(all(memory is None for memory in self.model.prediction_state_memory))
        self.assertTrue(all(memory is None for memory in self.model.error_state_memory))
        self.assertTrue(torch.equal(self.model.pc_error_c_sqrt, calibrated_c_sqrt))

        self.model.step_frame(self.frame2)
        self.assertEqual(self.model.real_frame_update_count, 1)
        self.assertTrue(
            all(state.previous_dynamic_error is None for state in self.model.layer_states)
        )


class LearnedRecurrentErrorTransitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(11)
        cls.model = PVGG16TargetFlow(
            backbone=vgg16(weights=None),
            task="real_frame_pc",
            dynamic_error=True,
            error_state_mode="ema",
            real_frame_transition_mode="convgru_error",
        ).eval()
        cls.frame1 = torch.randn(1, 3, 32, 32)
        cls.frame2 = torch.randn(1, 3, 32, 32)

    def setUp(self):
        self.model.reset()
        self.model.zero_grad(set_to_none=True)

    def test_convgru_replaces_gradient_projection_and_keeps_feedback_update(self):
        self.model.step_frame(self.frame1)
        first_states = RealFramePredictiveCodingTest._clone_states(
            self.model.layer_states
        )
        self.model.step_frame(self.frame2)

        self.assertEqual(
            self.model.recurrence_outputs["transition_mode"],
            "convgru_error",
        )
        for layer_index, state in enumerate(self.model.layer_states):
            previous_representation = first_states[layer_index]["representation"]
            base_representation = previous_representation + (
                self.model.pc_ff_multipliers[layer_index]
                * (state.feedforward_drive - previous_representation)
            )
            feedback_drive = torch.zeros_like(state.feedforward_drive)
            if layer_index + 1 < self.model.number_of_layers:
                expected_feedback = first_states[layer_index + 1]["prediction"]
                self.assertTrue(
                    torch.equal(state.previous_feedback_prediction, expected_feedback)
                )
                feedback_drive = expected_feedback
                base_representation = base_representation + (
                    self.model.pc_fb_multipliers[layer_index]
                    * (expected_feedback - previous_representation)
                )
            else:
                self.assertIsNone(state.previous_feedback_prediction)
            with torch.no_grad():
                error_drive = self.model.forward_stages[layer_index](
                    first_states[layer_index]["dynamic_error"]
                )
                expected = self.model.recurrent_transition_modules[layer_index](
                    previous_representation,
                    state.feedforward_drive,
                    error_drive,
                    feedback_drive,
                    base_representation,
                )
            self.assertIsNone(state.error_correction)
            self.assertIsNone(state.error_scale)
            self.assertIsNone(state.c_sqrt)
            self.assertTrue(torch.allclose(state.representation, expected))
            expected_dynamic_error = (
                0.207 * state.instant_error
                + 0.793 * first_states[layer_index]["dynamic_error"]
            )
            self.assertTrue(torch.allclose(state.dynamic_error, expected_dynamic_error))

    def test_only_transition_parameters_train_and_cross_frame_state_is_detached(self):
        trainable_names = {
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable_names)
        self.assertTrue(
            all(name.startswith("recurrent_transition_modules.") for name in trainable_names)
        )

        self.model.step_frame(self.frame1)
        self.model.step_frame(self.frame2)
        loss = self.model.collect_recurrent_transition_loss()
        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)
        loss.backward()

        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        self.assertTrue(any(parameter.grad is not None for parameter in trainable_parameters))
        frozen_parameters = [
            parameter for parameter in self.model.parameters() if not parameter.requires_grad
        ]
        self.assertTrue(all(parameter.grad is None for parameter in frozen_parameters))
        memories = (
            self.model.representation_state_memory
            + self.model.prediction_state_memory
            + self.model.instant_error_state_memory
            + self.model.error_state_memory
        )
        self.assertTrue(all(not memory.requires_grad for memory in memories))
        self.assertTrue(all(memory.grad_fn is None for memory in memories))

    def test_zeroed_control_changes_only_the_transition_error_input(self):
        zeroed = copy.deepcopy(self.model)
        zeroed.real_frame_recurrent_error_input = "zeroed"
        with torch.no_grad():
            for transition in self.model.recurrent_transition_modules:
                channels = transition.candidate.out_channels
                transition.candidate.weight[:, channels : 2 * channels].zero_()
                diagonal = torch.arange(channels)
                transition.candidate.weight[diagonal, channels + diagonal, 0, 0] = 0.1
            zeroed.recurrent_transition_modules.load_state_dict(
                self.model.recurrent_transition_modules.state_dict()
            )

        self.model.reset()
        zeroed.reset()
        self.model.step_frame(self.frame1)
        zeroed.step_frame(self.frame1)
        self.model.step_frame(self.frame2)
        zeroed.step_frame(self.frame2)

        for dynamic_state, zeroed_state in zip(
            self.model.layer_states, zeroed.layer_states
        ):
            self.assertTrue(
                torch.equal(
                    dynamic_state.previous_dynamic_error,
                    zeroed_state.previous_dynamic_error,
                )
            )
            self.assertTrue(
                torch.equal(
                    dynamic_state.previous_feedback_prediction,
                    zeroed_state.previous_feedback_prediction,
                )
                if dynamic_state.previous_feedback_prediction is not None
                else zeroed_state.previous_feedback_prediction is None
            )
        self.assertTrue(
            any(
                not torch.allclose(dynamic_state.representation, zeroed_state.representation)
                for dynamic_state, zeroed_state in zip(
                    self.model.layer_states, zeroed.layer_states
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
