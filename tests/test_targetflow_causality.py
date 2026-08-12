import unittest

import torch
from torchvision.models import vgg16

from predify2021.model_factory.pvgg16_targetflow import PVGG16TargetFlow


class TargetFlowCausalityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = PVGG16TargetFlow(
            backbone=vgg16(weights=None),
            temporal_target_mode="ego_motion",
            dynamic_error=True,
            error_sample_time=0.1,
            error_time_constant=0.5,
        )
        cls.model.eval()
        cls.current = torch.randn(1, 3, 32, 32)
        cls.future_a = torch.randn(1, 3, 32, 32)
        cls.future_b = torch.randn(1, 3, 32, 32)
        cls.ego_motion = torch.tensor([[[0.25, -0.05]]])

    def setUp(self):
        self.model.task = "motion"
        self.model.future_feature_history_mode = "none"
        self.model.reset()

    def _future_feature_step(self, current, future, history_mode="none"):
        self.model.task = "future_feature"
        self.model.future_feature_history_mode = history_mode
        with torch.no_grad():
            self.model.step_frame(
                current,
                top_target_provider=lambda: self.model.extract_top_forward_feature(
                    future,
                    detach=True,
                ),
            )
        return {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in self.model.future_prediction_outputs.items()
        }

    def _predict_with_future(self, future):
        self.model.reset()
        with torch.no_grad():
            future_top = self.model.extract_top_forward_feature(future)
            self.model.step_frame(
                self.current,
                top_target=future_top,
                temporal_target_override=self.ego_motion,
            )
        return (
            self.model.temporal_prediction.clone(),
            self.model.layer_states[-1].instant_error.clone(),
            self.model.temporal_context.clone(),
        )

    def test_prediction_does_not_depend_on_current_future_target(self):
        prediction_a, error_a, context_a = self._predict_with_future(self.future_a)
        prediction_b, error_b, context_b = self._predict_with_future(self.future_b)

        self.assertTrue(torch.equal(context_a, context_b))
        self.assertTrue(torch.equal(prediction_a, prediction_b))
        self.assertFalse(torch.equal(error_a, error_b))

    def test_recursive_local_losses_produce_feedback_gradients(self):
        self.model.reset()
        self.model.zero_grad(set_to_none=True)
        future_top = self.model.extract_top_forward_feature(self.future_a)
        self.model.step_frame(
            self.current,
            top_target=future_top,
            temporal_target_override=self.ego_motion,
        )

        _, total_local_loss = self.model.collect_learn_flow_losses()
        total_local_loss.backward()
        feedback_gradient_sum = sum(
            parameter.grad.detach().abs().sum().item()
            for parameter in self.model.feedback_modules.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(feedback_gradient_sum, 0.0)

    def test_temporal_loss_produces_predictor_gradients(self):
        self.model.reset()
        self.model.zero_grad(set_to_none=True)
        future_top = self.model.extract_top_forward_feature(self.future_a)
        self.model.step_frame(
            self.current,
            top_target=future_top,
            temporal_target_override=self.ego_motion,
        )

        temporal_loss = self.model.collect_temporal_prediction_loss()
        temporal_loss.backward()
        gradient_sum = sum(
            parameter.grad.detach().abs().sum().item()
            for parameter in self.model.temporal_predictor.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_sum, 0.0)

    def test_current_top_duplicate_adds_only_current_top_representation(self):
        self.model.reset()
        with torch.no_grad():
            current_top = self.model.extract_top_forward_feature(self.current)
            future_top = self.model.extract_top_forward_feature(self.future_a)
            self.model.step_frame(
                self.current,
                top_target=future_top,
                temporal_target_override=self.ego_motion,
                duplicate_current_top_context=True,
            )

        error_start = self.model.stage_channels[-1]
        prediction_start = error_start + sum(self.model.stage_channels)
        current_top_start = prediction_start + sum(self.model.stage_channels[:-1])
        self.assertEqual(
            torch.count_nonzero(self.model.temporal_context[:, error_start:prediction_start]).item(),
            0,
        )
        self.assertEqual(
            torch.count_nonzero(self.model.temporal_context[:, prediction_start:current_top_start]).item(),
            0,
        )
        self.assertTrue(
            torch.equal(
                self.model.temporal_context[:, current_top_start:],
                current_top.mean(dim=(-1, -2)),
            )
        )

    def test_recurrent_memories_are_detached_between_frames(self):
        self.model.reset()
        future_top = self.model.extract_top_forward_feature(self.future_a)
        self.model.step_frame(
            self.current,
            top_target=future_top,
            temporal_target_override=self.ego_motion,
        )

        memories = (
            self.model.error_state_memory
            + self.model.instant_error_state_memory
            + self.model.prediction_state_memory
        )
        self.assertTrue(all(memory is not None for memory in memories))
        self.assertTrue(all(not memory.requires_grad for memory in memories))
        self.assertTrue(all(memory.grad_fn is None for memory in memories))

    def test_reset_each_frame_removes_inherited_context(self):
        self.model.reset()
        with torch.no_grad():
            future_a_top = self.model.extract_top_forward_feature(self.future_a)
            self.model.step_frame(
                self.current,
                top_target=future_a_top,
                temporal_target_override=self.ego_motion,
            )
            future_b_top = self.model.extract_top_forward_feature(self.future_b)
            self.model.step_frame(
                self.future_a,
                top_target=future_b_top,
                temporal_target_override=self.ego_motion,
            )
            inherited_memory_context = self.model.temporal_context[:, 512:].clone()

            self.model.reset()
            self.model.step_frame(
                self.future_a,
                top_target=future_b_top,
                temporal_target_override=self.ego_motion,
            )
            reset_memory_context = self.model.temporal_context[:, 512:].clone()

        self.assertGreater(torch.count_nonzero(inherited_memory_context).item(), 0)
        self.assertEqual(torch.count_nonzero(reset_memory_context).item(), 0)

    def test_ema_state_does_not_rescale_instant_local_loss_gradient(self):
        self.model.reset()
        future_top = self.model.extract_top_forward_feature(self.future_a)
        self.model.step_frame(
            self.current,
            top_target=future_top,
            temporal_target_override=self.ego_motion,
        )

        self.assertEqual(self.model.error_state_mode, "ema")
        self.assertFalse(
            torch.allclose(
                self.model.layer_states[-1].error,
                self.model.layer_states[-1].instant_error,
            )
        )

        for state in self.model.layer_states:
            expected_loss = torch.mean(state.instant_error.pow(2))
            expected_gradient = 2.0 * state.instant_error / state.instant_error.numel()
            actual_gradient = torch.autograd.grad(
                state.local_loss,
                state.forward_output,
                retain_graph=True,
            )[0]

            self.assertIs(state.loss_error, state.instant_error)
            self.assertTrue(torch.allclose(state.local_loss, expected_loss))
            self.assertTrue(torch.allclose(actual_gradient, expected_gradient))

    def test_future_predictor_is_independent_of_two_dof_motion_head(self):
        self.assertEqual(self.model.temporal_predictor[-1].out_features, 2)
        self.assertEqual(self.model.future_feature_predictor[-1].out_channels, 512)
        self.assertIsNot(self.model.temporal_predictor, self.model.future_feature_predictor)

        predictor_id = id(self.model.future_feature_predictor)
        parameter_count = sum(
            parameter.numel()
            for parameter in self.model.future_feature_predictor.parameters()
        )
        for mode in ("none", "latest", "two_tap", "recursive"):
            self.model.future_feature_history_mode = mode
            self.assertEqual(id(self.model.future_feature_predictor), predictor_id)
            self.assertEqual(
                sum(
                    parameter.numel()
                    for parameter in self.model.future_feature_predictor.parameters()
                ),
                parameter_count,
            )

    def test_future_predictor_can_use_a_three_by_three_spatial_layer(self):
        spatial_model = PVGG16TargetFlow(
            backbone=vgg16(weights=None),
            task="future_feature",
            future_feature_predictor_kernel_size=3,
        )
        first_layer = spatial_model.future_feature_predictor[0]
        final_layer = spatial_model.future_feature_predictor[-1]

        self.assertEqual(first_layer.kernel_size, (3, 3))
        self.assertEqual(first_layer.padding, (1, 1))
        self.assertEqual(final_layer.kernel_size, (1, 1))
        sample = torch.randn(1, 1024, 14, 14)
        self.assertEqual(
            spatial_model.future_feature_predictor(sample).shape,
            (1, 512, 14, 14),
        )

    def test_future_target_provider_runs_after_predictor(self):
        events = []
        hook = self.model.future_feature_predictor.register_forward_hook(
            lambda *_: events.append("predict")
        )
        self.model.task = "future_feature"
        try:
            with torch.no_grad():
                self.model.step_frame(
                    self.current,
                    top_target_provider=lambda: (
                        events.append("target")
                        or self.model.extract_top_forward_feature(self.future_a)
                    ),
                )
        finally:
            hook.remove()

        self.assertEqual(events[:2], ["predict", "target"])

    def test_future_prediction_does_not_depend_on_current_target(self):
        self.model.reset()
        prediction_a = self._future_feature_step(
            self.current,
            self.future_a,
        )["predicted_future_top"]
        error_a = self.model.layer_states[-1].instant_error.clone()

        self.model.reset()
        prediction_b = self._future_feature_step(
            self.current,
            self.future_b,
        )["predicted_future_top"]
        error_b = self.model.layer_states[-1].instant_error.clone()

        self.assertTrue(torch.equal(prediction_a, prediction_b))
        self.assertFalse(torch.equal(error_a, error_b))

    def test_instant_history_cannot_see_current_pair_target(self):
        def predict_second(current_pair_target):
            self.model.reset()
            self._future_feature_step(
                self.current,
                self.future_a,
                history_mode="latest",
            )
            return self._future_feature_step(
                self.future_a,
                current_pair_target,
                history_mode="latest",
            )["predicted_future_top"]

        prediction_a = predict_second(self.future_a)
        prediction_b = predict_second(self.future_b)
        self.assertTrue(torch.equal(prediction_a, prediction_b))

    def test_future_feature_history_uses_previous_completed_pair(self):
        self.model.reset()
        self._future_feature_step(self.current, self.future_a, history_mode="recursive")
        previous_recursive = self.model.recursive_error_state_memory[-1].clone()
        second = self._future_feature_step(
            self.future_a,
            self.future_b,
            history_mode="recursive",
        )

        self.assertTrue(torch.equal(second["history_top"], previous_recursive))

        self.model.reset()
        self._future_feature_step(self.current, self.future_a, history_mode="latest")
        previous_instant = self.model.instant_error_state_memory[-1].clone()
        second = self._future_feature_step(
            self.future_a,
            self.future_b,
            history_mode="latest",
        )
        self.assertTrue(torch.equal(second["history_top"], previous_instant))

    def test_future_prediction_error_uses_target_minus_prediction_sign(self):
        self.model.reset()
        outputs = self._future_feature_step(self.current, self.future_a)
        expected_error = (
            outputs["future_top_target"] - outputs["predicted_future_top"]
        )
        self.assertTrue(torch.equal(outputs["prediction_error_top"], expected_error))

    def test_future_and_delta_mse_are_equivalent(self):
        self.model.reset()
        self._future_feature_step(self.current, self.future_a)
        losses = self.model.collect_future_feature_prediction_losses()
        self.assertTrue(torch.allclose(losses["future_mse"], losses["delta_mse"]))

    def test_copy_current_bypasses_predictor(self):
        predictor_calls = []
        hook = self.model.future_feature_predictor.register_forward_hook(
            lambda *_: predictor_calls.append(True)
        )
        try:
            outputs = self._future_feature_step(
                self.current,
                self.future_a,
                history_mode="copy_current",
            )
        finally:
            hook.remove()

        self.assertEqual(predictor_calls, [])
        self.assertTrue(
            torch.equal(outputs["predicted_future_top"], outputs["current_top"])
        )
        self.assertEqual(torch.count_nonzero(outputs["predicted_delta_top"]).item(), 0)

    def test_feature_loss_produces_only_future_predictor_gradients(self):
        self.model.task = "future_feature"
        self.model.future_feature_history_mode = "none"
        self.model.zero_grad(set_to_none=True)
        future_top = self.model.extract_top_forward_feature(self.future_a)
        self.model.step_frame(self.current, top_target=future_top)
        losses = self.model.collect_future_feature_prediction_losses()
        losses["future_mse"].backward()

        future_gradient_sum = sum(
            parameter.grad.detach().abs().sum().item()
            for parameter in self.model.future_feature_predictor.parameters()
            if parameter.grad is not None
        )
        motion_gradient_count = sum(
            parameter.grad is not None
            for parameter in self.model.temporal_predictor.parameters()
        )
        self.assertGreater(future_gradient_sum, 0.0)
        self.assertEqual(motion_gradient_count, 0)


if __name__ == "__main__":
    unittest.main()
