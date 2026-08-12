import unittest
from unittest.mock import patch

import torch
from torchvision.models import vgg16

from predify2021.model_factory.pvgg16_targetflow import PVGG16TargetFlow


class AlignedTemporalDifferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(19)
        cls.model = PVGG16TargetFlow(
            backbone=vgg16(weights=None),
            task="future_feature",
            future_feature_history_mode="aligned_difference",
            future_feature_temporal_fusion_mode="none",
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

    def test_first_frame_uses_zero_difference_and_current_base(self):
        predictor_inputs = []
        hook = self.model.future_feature_predictor[0].register_forward_pre_hook(
            lambda _, inputs: predictor_inputs.append(inputs[0].detach().clone())
        )
        try:
            with torch.no_grad():
                outputs = self._step(self.frame_a, self.frame_b)
        finally:
            hook.remove()

        self.assertIsNone(self.model.temporal_fusion_module)
        self.assertFalse(outputs["aligned_difference_applied"])
        self.assertFalse(outputs["temporal_fusion_applied"])
        self.assertTrue(torch.equal(outputs["prediction_base_top"], outputs["current_top"]))
        self.assertTrue(torch.equal(outputs["fused_top"], outputs["current_top"]))
        self.assertEqual(
            torch.count_nonzero(outputs["aligned_temporal_difference_top"]).item(),
            0,
        )
        self.assertTrue(torch.equal(predictor_inputs[0][:, :512], outputs["current_top"]))
        self.assertEqual(torch.count_nonzero(predictor_inputs[0][:, 512:]).item(), 0)

    def test_predictor_receives_current_and_aligned_difference_without_splat(self):
        with torch.no_grad():
            first = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in self._step(self.frame_a, self.frame_b).items()
            }
        aligned = torch.full_like(first["current_top"], 0.25)
        diagnostics = {
            "aligned_source": aligned,
            "dy": torch.zeros_like(aligned[:, 0], dtype=torch.int64),
            "dx": torch.ones_like(aligned[:, 0], dtype=torch.int64),
            "patch_matching_cost": torch.zeros_like(aligned[:, 0]),
        }
        predictor_inputs = []
        hook = self.model.future_feature_predictor[0].register_forward_pre_hook(
            lambda _, inputs: predictor_inputs.append(inputs[0].detach().clone())
        )
        try:
            with (
                patch(
                    "predify2021.model_factory.pvgg16_targetflow.align_source_to_target",
                    return_value=diagnostics,
                ),
                patch(
                    "predify2021.model_factory.pvgg16_targetflow.forward_splat_discrete",
                    side_effect=AssertionError(
                        "Aligned temporal difference must not use future splatting."
                    ),
                ),
                torch.no_grad(),
            ):
                outputs = self._step(self.frame_b, self.frame_c)
        finally:
            hook.remove()

        expected_difference = outputs["current_top"] - aligned
        self.assertTrue(outputs["aligned_difference_applied"])
        self.assertFalse(outputs["temporal_fusion_applied"])
        self.assertIsNone(outputs["fusion_previous_top"])
        self.assertEqual(torch.count_nonzero(outputs["fusion_residual_top"]).item(), 0)
        self.assertTrue(torch.equal(outputs["prediction_base_top"], outputs["current_top"]))
        self.assertTrue(
            torch.equal(
                outputs["aligned_difference_raw_previous_top"],
                first["current_top"],
            )
        )
        self.assertTrue(torch.equal(outputs["aligned_difference_previous_top"], aligned))
        self.assertTrue(
            torch.equal(outputs["aligned_temporal_difference_top"], expected_difference)
        )
        self.assertTrue(torch.equal(outputs["history_top"], expected_difference))
        self.assertTrue(torch.equal(predictor_inputs[0][:, :512], outputs["current_top"]))
        self.assertTrue(torch.equal(predictor_inputs[0][:, 512:], expected_difference))
        self.assertFalse(outputs["aligned_temporal_difference_top"].requires_grad)
        self.assertFalse(outputs["aligned_difference_previous_top"].requires_grad)
        torch.testing.assert_close(
            outputs["predicted_future_top"],
            outputs["current_top"] + outputs["predicted_residual_top"],
        )

    def test_future_target_cannot_change_difference_or_prediction(self):
        def predict_with_target(target):
            self.model.reset()
            with torch.no_grad():
                self._step(self.frame_a, self.frame_b)
                outputs = self._step(self.frame_b, target)
            return (
                outputs["aligned_temporal_difference_top"].clone(),
                outputs["predicted_future_top"].clone(),
            )

        difference_c, predicted_c = predict_with_target(self.frame_c)
        difference_d, predicted_d = predict_with_target(self.frame_d)
        self.assertTrue(torch.equal(difference_c, difference_d))
        self.assertTrue(torch.equal(predicted_c, predicted_d))

    def test_aligned_difference_rejects_a_warp_prediction_base(self):
        with self.assertRaisesRegex(ValueError, "requires current_residual"):
            PVGG16TargetFlow(
                backbone=vgg16(weights=None),
                task="future_feature",
                future_feature_history_mode="aligned_difference",
                future_feature_prediction_form="historical_warp_residual",
            )


if __name__ == "__main__":
    unittest.main()
