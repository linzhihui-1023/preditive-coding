import unittest

import torch
import torch.nn as nn

from predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs import (
    TemporalFeatureVarianceWindow,
    build_optimizer,
)


class _SmallTargetFlowStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_stages = nn.ModuleList([nn.Linear(2, 2)])
        self.feedback_modules = nn.ModuleList([nn.Linear(2, 2)])
        self.temporal_predictor = nn.Linear(2, 2)


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


if __name__ == "__main__":
    unittest.main()
