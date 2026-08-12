import unittest

import torch
from torchvision.models import vgg16

from predify2021.model_factory.pvgg16_targetflow import PVGG16TargetFlow


class TemporalFusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.model = PVGG16TargetFlow(
            backbone=vgg16(weights=None),
            task="future_feature",
            future_feature_history_mode="none",
            future_feature_temporal_fusion_mode="two_frame_residual",
            future_feature_prediction_form="current_residual",
        )
        cls.model.eval()
        cls.frame_a = torch.randn(1, 3, 32, 32)
        cls.frame_b = torch.randn(1, 3, 32, 32)
        cls.frame_c = torch.randn(1, 3, 32, 32)
        cls.frame_d = torch.randn(1, 3, 32, 32)

    def setUp(self):
        self.model.reset()
        self.model.zero_grad(set_to_none=True)

    def _step(self, current, future):
        self.model.step_frame(
            current,
            top_target_provider=lambda: self.model.extract_top_forward_feature(
                future,
                detach=True,
            ),
        )
        return self.model.future_prediction_outputs

    def test_fusion_module_is_minimal_residual_conv_stack(self):
        module = self.model.temporal_fusion_module
        self.assertEqual(module[0].in_channels, 1024)
        self.assertEqual(module[0].out_channels, 512)
        self.assertEqual(module[0].kernel_size, (1, 1))
        self.assertIsInstance(module[1], torch.nn.ReLU)
        self.assertEqual(module[2].in_channels, 512)
        self.assertEqual(module[2].out_channels, 512)
        self.assertEqual(module[2].kernel_size, (1, 1))
        self.assertEqual(torch.count_nonzero(module[2].weight).item(), 0)
        self.assertEqual(torch.count_nonzero(module[2].bias).item(), 0)

    def test_first_frame_reduces_exactly_to_current_feature(self):
        with torch.no_grad():
            outputs = self._step(self.frame_a, self.frame_b)

        self.assertFalse(outputs["temporal_fusion_applied"])
        self.assertIsNone(outputs["fusion_previous_top"])
        self.assertTrue(torch.equal(outputs["fused_top"], outputs["current_top"]))
        self.assertEqual(torch.count_nonzero(outputs["fusion_residual_top"]).item(), 0)

    def test_next_frame_uses_detached_previous_top(self):
        with torch.no_grad():
            first = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in self._step(self.frame_a, self.frame_b).items()
            }
            second = self._step(self.frame_b, self.frame_c)

        previous = second["fusion_previous_top"]
        self.assertTrue(second["temporal_fusion_applied"])
        self.assertTrue(torch.equal(previous, first["current_top"]))
        self.assertFalse(previous.requires_grad)
        self.assertIsNone(previous.grad_fn)
        self.assertTrue(
            torch.equal(
                self.model.future_feature_previous_top_memory,
                second["current_top"],
            )
        )

    def test_current_future_target_cannot_change_fused_or_predicted_feature(self):
        def predict_with_target(target):
            self.model.reset()
            with torch.no_grad():
                self._step(self.frame_a, self.frame_b)
                outputs = self._step(self.frame_b, target)
            return outputs["fused_top"].clone(), outputs["predicted_future_top"].clone()

        fused_c, predicted_c = predict_with_target(self.frame_c)
        fused_d, predicted_d = predict_with_target(self.frame_d)
        self.assertTrue(torch.equal(fused_c, fused_d))
        self.assertTrue(torch.equal(predicted_c, predicted_d))

    def test_feature_loss_reaches_fusion_and_predictor_without_bptt(self):
        with torch.no_grad():
            self._step(self.frame_a, self.frame_b)
        self._step(self.frame_b, self.frame_c)
        self.model.collect_future_feature_prediction_losses()["future_mse"].backward()

        fusion_gradient = sum(
            parameter.grad.detach().abs().sum().item()
            for parameter in self.model.temporal_fusion_module.parameters()
            if parameter.grad is not None
        )
        predictor_gradient = sum(
            parameter.grad.detach().abs().sum().item()
            for parameter in self.model.future_feature_predictor.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(fusion_gradient, 0.0)
        self.assertGreater(predictor_gradient, 0.0)
        self.assertFalse(self.model.future_prediction_outputs["fusion_previous_top"].requires_grad)

    def test_reset_removes_previous_feature(self):
        with torch.no_grad():
            self._step(self.frame_a, self.frame_b)
        self.assertIsNotNone(self.model.future_feature_previous_top_memory)
        self.model.reset()
        self.assertIsNone(self.model.future_feature_previous_top_memory)

    def test_fusion_rejects_other_history_and_prediction_forms(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            PVGG16TargetFlow(
                backbone=vgg16(weights=None),
                task="future_feature",
                future_feature_history_mode="recursive",
                future_feature_temporal_fusion_mode="two_frame_residual",
            )
        with self.assertRaisesRegex(ValueError, "requires current_residual"):
            PVGG16TargetFlow(
                backbone=vgg16(weights=None),
                task="future_feature",
                future_feature_history_mode="none",
                future_feature_temporal_fusion_mode="two_frame_residual",
                future_feature_prediction_form="historical_warp",
            )


if __name__ == "__main__":
    unittest.main()
