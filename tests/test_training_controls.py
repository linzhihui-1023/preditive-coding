import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs import (
    TemporalFeatureVarianceWindow,
    build_optimizer,
    configure_student_trainability,
    reset_stream_state,
    validate_formal_drive_split,
    validate_same_drive_configuration,
    validate_variance_configuration,
)


class _SmallTargetFlowStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_stages = nn.ModuleList([nn.Linear(2, 2)])
        self.feedback_modules = nn.ModuleList([nn.Linear(2, 2)])
        self.temporal_predictor = nn.Linear(2, 2)


class _SmallFutureFeatureStudent(_SmallTargetFlowStudent):
    def __init__(self, with_fusion):
        super().__init__()
        self.future_feature_predictor = nn.Linear(2, 2)
        self.temporal_fusion_module = nn.Linear(2, 2) if with_fusion else None


class FeedbackOptimizerTest(unittest.TestCase):
    def test_feedback_parameters_are_updated_and_cleared_by_optimizer(self):
        student = _SmallTargetFlowStudent()
        optimizer = build_optimizer(student)
        feedback_parameter = next(student.feedback_modules.parameters())
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        before = feedback_parameter.detach().clone()

        self.assertIn(id(feedback_parameter), optimizer_parameter_ids)

        loss = student.feedback_modules[0](torch.ones(1, 2)).pow(2).mean()
        loss.backward()
        self.assertIsNotNone(feedback_parameter.grad)
        optimizer.step()
        self.assertFalse(torch.equal(feedback_parameter.detach(), before))

        optimizer.zero_grad(set_to_none=True)
        self.assertIsNone(feedback_parameter.grad)

    def test_backbone_is_frozen_by_default_and_opted_in_explicitly(self):
        student = _SmallTargetFlowStudent()
        configure_student_trainability(student)
        optimizer = build_optimizer(student)
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        forward_parameter = next(student.forward_stages.parameters())

        self.assertFalse(forward_parameter.requires_grad)
        self.assertNotIn(id(forward_parameter), optimizer_parameter_ids)

        with patch(
            "predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs.TRAIN_BACKBONE",
            True,
        ):
            configure_student_trainability(student)
            optimizer = build_optimizer(student)
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertTrue(forward_parameter.requires_grad)
        self.assertIn(id(forward_parameter), optimizer_parameter_ids)

    def test_current_only_trains_only_future_predictor(self):
        student = _SmallFutureFeatureStudent(with_fusion=False)
        with (
            patch(
                "predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs.PREDICTION_TASK",
                "future_feature",
            ),
            patch(
                "predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs.FEATURE_HISTORY_MODE",
                "none",
            ),
            patch(
                "predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs.TRAIN_BACKBONE",
                False,
            ),
            patch(
                "predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs.TRAIN_FEEDBACK_DECODERS",
                False,
            ),
        ):
            configure_student_trainability(student)
            optimizer = build_optimizer(student)

        trainable = {
            name for name, parameter in student.named_parameters() if parameter.requires_grad
        }
        self.assertEqual(
            trainable,
            {"future_feature_predictor.weight", "future_feature_predictor.bias"},
        )
        self.assertIsNotNone(optimizer)

    def test_temporal_fusion_trains_fusion_and_future_predictor_only(self):
        student = _SmallFutureFeatureStudent(with_fusion=True)
        with (
            patch(
                "predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs.PREDICTION_TASK",
                "future_feature",
            ),
            patch(
                "predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs.FEATURE_HISTORY_MODE",
                "none",
            ),
            patch(
                "predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs.TRAIN_BACKBONE",
                False,
            ),
            patch(
                "predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs.TRAIN_FEEDBACK_DECODERS",
                False,
            ),
        ):
            configure_student_trainability(student)
            optimizer = build_optimizer(student)

        trainable = {
            name for name, parameter in student.named_parameters() if parameter.requires_grad
        }
        self.assertEqual(
            trainable,
            {
                "future_feature_predictor.weight",
                "future_feature_predictor.bias",
                "temporal_fusion_module.weight",
                "temporal_fusion_module.bias",
            },
        )
        self.assertIsNotNone(optimizer)

    def test_copy_current_has_no_optimizer_when_all_model_parts_are_frozen(self):
        student = _SmallFutureFeatureStudent(with_fusion=False)
        with (
            patch(
                "predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs.PREDICTION_TASK",
                "future_feature",
            ),
            patch(
                "predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs.FEATURE_HISTORY_MODE",
                "copy_current",
            ),
            patch(
                "predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs.TRAIN_BACKBONE",
                False,
            ),
            patch(
                "predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs.TRAIN_FEEDBACK_DECODERS",
                False,
            ),
        ):
            configure_student_trainability(student)
            optimizer = build_optimizer(student)

        self.assertEqual(
            [name for name, parameter in student.named_parameters() if parameter.requires_grad],
            [],
        )
        self.assertIsNone(optimizer)


class TemporalVarianceWindowTest(unittest.TestCase):
    def test_window_activates_without_backpropagating_through_history(self):
        window = TemporalFeatureVarianceWindow(window_size=3)
        first = torch.zeros(1, 2, requires_grad=True)
        second = torch.tensor([[0.002, 0.004]], requires_grad=True)

        first_loss, _, first_count = window.compute(first)
        second_loss, _, second_count = window.compute(second)

        self.assertEqual(first_count, 1)
        self.assertEqual(float(first_loss.detach().item()), 0.0)
        self.assertEqual(second_count, 2)
        self.assertGreater(float(second_loss.detach().item()), 0.0)
        self.assertTrue(all(not feature.requires_grad for feature in window.history))

        second_loss.backward()
        self.assertIsNone(first.grad)
        self.assertGreater(second.grad.abs().sum().item(), 0.0)

        window.compute(torch.tensor([[0.004, 0.008]]))
        _, _, fourth_count = window.compute(torch.tensor([[0.006, 0.012]]))
        self.assertEqual(fourth_count, 3)
        self.assertEqual(len(window.history), 2)

        window.reset()
        _, _, reset_count = window.compute(torch.tensor([[0.008, 0.016]]))
        self.assertEqual(reset_count, 1)

    def test_stream_reset_clears_model_and_variance_history(self):
        class ResetCounter:
            def __init__(self):
                self.count = 0

            def reset(self):
                self.count += 1

        student = ResetCounter()
        teacher = ResetCounter()
        window = TemporalFeatureVarianceWindow(window_size=3)
        window.compute(torch.ones(1, 2))

        reset_stream_state(student, teacher, window)

        self.assertEqual(student.count, 1)
        self.assertEqual(teacher.count, 1)
        self.assertEqual(window.history, [])


class FormalExperimentConfigurationTest(unittest.TestCase):
    def test_formal_split_requires_explicit_disjoint_drives(self):
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            validate_formal_drive_split(True, "drive_train", "")
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            validate_formal_drive_split(True, "drive_shared", "drive_shared")

        validate_formal_drive_split(True, "drive_train", "drive_val")
        validate_formal_drive_split(False, "", "")

    def test_same_drive_control_requires_fixed_split_and_gap(self):
        valid_arguments = (True, False, "", "", 0.6, 0.2, 20, True)
        validate_same_drive_configuration(*valid_arguments)

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            validate_same_drive_configuration(
                True, True, "", "", 0.6, 0.2, 20, True
            )
        with self.assertRaisesRegex(ValueError, "train_fraction=0.6"):
            validate_same_drive_configuration(
                True, False, "", "", 0.8, 0.2, 20, True
            )
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            validate_same_drive_configuration(
                True, False, "", "", 0.6, 0.2, 19, True
            )

    def test_positive_variance_weight_requires_trainable_backbone(self):
        with self.assertRaisesRegex(ValueError, "frozen top feature"):
            validate_variance_configuration(False, 0.001)

        validate_variance_configuration(False, 0.0)
        validate_variance_configuration(True, 0.001)


if __name__ == "__main__":
    unittest.main()
