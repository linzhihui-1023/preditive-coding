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

    def test_current_teacher_control_adds_only_current_top_representation(self):
        self.model.reset()
        with torch.no_grad():
            current_teacher_top = self.model.extract_top_forward_feature(self.current)
            future_top = self.model.extract_top_forward_feature(self.future_a)
            self.model.step_frame(
                self.current,
                top_target=future_top,
                temporal_target_override=self.ego_motion,
                current_teacher_top_context=current_teacher_top,
            )

        error_start = self.model.stage_channels[-1]
        prediction_start = error_start + sum(self.model.stage_channels)
        teacher_top_start = prediction_start + sum(self.model.stage_channels[:-1])
        self.assertEqual(
            torch.count_nonzero(self.model.temporal_context[:, error_start:prediction_start]).item(),
            0,
        )
        self.assertEqual(
            torch.count_nonzero(self.model.temporal_context[:, prediction_start:teacher_top_start]).item(),
            0,
        )
        self.assertTrue(
            torch.equal(
                self.model.temporal_context[:, teacher_top_start:],
                current_teacher_top.mean(dim=(-1, -2)),
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


if __name__ == "__main__":
    unittest.main()
