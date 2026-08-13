import unittest
from unittest.mock import patch

import torch
from torchvision.models import vgg16

from predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs import (
    build_future_feature_target_providers,
)
from predify2021.model_factory.pvgg16_targetflow import PVGG16TargetFlow


class FutureFeatureStageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(31)
        cls.model = PVGG16TargetFlow(
            backbone=vgg16(weights=None),
            task="future_feature",
            future_feature_stage=4,
            future_feature_history_mode="aligned_difference",
            future_feature_prediction_form="current_residual",
            future_motion_radius=1,
        )
        cls.model.eval()
        cls.current = torch.randn(1, 3, 32, 32)
        cls.future = torch.randn(1, 3, 32, 32)

    def setUp(self):
        self.model.reset()

    def _targets(self, frame):
        return self.model.extract_forward_features_at_stages(
            frame,
            stages=(4, 5),
            detach=True,
        )

    def _step(self, current, future):
        targets = self._targets(future)
        self.model.step_frame(
            current,
            top_target_provider=lambda: targets[5],
            future_feature_target_provider=lambda: targets[4],
        )
        return self.model.future_prediction_outputs, targets

    def test_stage4_prediction_target_is_separate_from_stage5_target_flow(self):
        targets = self._targets(self.future)
        events = []
        hook = self.model.future_feature_predictor[0].register_forward_pre_hook(
            lambda _, __: events.append("predict")
        )
        try:
            with torch.no_grad():
                self.model.step_frame(
                    self.current,
                    top_target_provider=lambda: (
                        events.append("target_flow_target") or targets[5]
                    ),
                    future_feature_target_provider=lambda: (
                        events.append("future_prediction_target") or targets[4]
                    ),
                )
        finally:
            hook.remove()

        outputs = self.model.future_prediction_outputs
        self.assertEqual(events[0], "predict")
        self.assertEqual(outputs["future_feature_stage"], 4)
        self.assertEqual(outputs["target_flow_top_stage"], 5)
        self.assertTrue(torch.equal(outputs["target_flow_top_target"], targets[5]))
        self.assertTrue(torch.equal(outputs["future_prediction_target"], targets[4]))
        self.assertTrue(
            torch.equal(self.model.layer_states[-1].target_output, targets[5])
        )
        self.assertEqual(outputs["predicted_future_feature"].shape, targets[4].shape)
        self.assertNotEqual(targets[4].shape[-2:], targets[5].shape[-2:])

    def test_prediction_stage_memory_is_detached_stage4_feature(self):
        with torch.no_grad():
            outputs, _ = self._step(self.current, self.future)

        memory = self.model.future_feature_previous_prediction_stage_memory
        self.assertTrue(torch.equal(memory, outputs["current_prediction_feature"]))
        self.assertFalse(memory.requires_grad)
        self.assertIsNone(memory.grad_fn)
        self.assertFalse(hasattr(self.model, "future_feature_previous_top_memory"))
        self.model.reset()
        self.assertIsNone(
            self.model.future_feature_previous_prediction_stage_memory
        )

    def test_stage4_requires_its_own_future_prediction_target(self):
        targets = self._targets(self.future)
        with self.assertRaisesRegex(ValueError, "separate future-feature target"):
            with torch.no_grad():
                self.model.step_frame(
                    self.current,
                    top_target_provider=lambda: targets[5],
                )

    def test_target_shape_mismatch_is_rejected(self):
        targets = self._targets(self.future)
        with self.assertRaisesRegex(ValueError, "target shape does not match"):
            with torch.no_grad():
                self.model.step_frame(
                    self.current,
                    top_target=targets[5],
                    future_feature_target=targets[5],
                )

    def test_stage3_predictor_uses_stage3_channels(self):
        model = PVGG16TargetFlow(
            backbone=vgg16(weights=None),
            task="future_feature",
            future_feature_stage=3,
        )
        self.assertEqual(model.future_feature_channels, 256)
        self.assertEqual(model.future_feature_predictor[0].in_channels, 512)
        self.assertEqual(model.future_feature_predictor[0].out_channels, 512)
        self.assertEqual(model.future_feature_predictor[2].out_channels, 256)

    def test_invalid_prediction_stage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "one of 3, 4, or 5"):
            PVGG16TargetFlow(
                backbone=vgg16(weights=None),
                task="future_feature",
                future_feature_stage=2,
            )


class FutureFeatureTargetProviderTest(unittest.TestCase):
    def test_stage4_and_stage5_targets_share_one_future_extraction(self):
        class TargetModel:
            number_of_layers = 5

            def __init__(self):
                self.calls = []

            def extract_forward_features_at_stages(self, frames, stages, detach):
                self.calls.append((frames, stages, detach))
                return {4: torch.tensor([4.0]), 5: torch.tensor([5.0])}

        teacher = TargetModel()
        frames = object()
        with (
            patch(
                "predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs."
                "TOP_TARGET_SOURCE",
                "ema_teacher",
            ),
            patch(
                "predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs."
                "FUTURE_FEATURE_STAGE",
                4,
            ),
        ):
            top_provider, future_provider = build_future_feature_target_providers(
                student=None,
                teacher=teacher,
                next_frames=frames,
            )
            self.assertEqual(top_provider().item(), 5.0)
            self.assertEqual(future_provider().item(), 4.0)

        self.assertEqual(len(teacher.calls), 1)
        self.assertIs(teacher.calls[0][0], frames)
        self.assertEqual(teacher.calls[0][1], (5, 4))
        self.assertTrue(teacher.calls[0][2])


if __name__ == "__main__":
    unittest.main()
