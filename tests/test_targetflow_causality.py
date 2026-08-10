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


if __name__ == "__main__":
    unittest.main()
