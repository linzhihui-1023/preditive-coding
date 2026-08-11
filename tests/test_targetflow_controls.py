import unittest

import torch
import torch.nn as nn

from predify2021.model_factory.targetflow import (
    TargetFlowFeedbackModule,
    TargetFlowLayerState,
    build_targetflow_error,
    run_backward_target_flow,
)


class TargetFlowErrorStateTest(unittest.TestCase):
    def test_ema_and_two_tap_use_distinct_memories_with_matched_coefficients(self):
        forward = torch.tensor([10.0])
        target = torch.tensor([0.0])
        previous_state = torch.tensor([7.0])
        previous_instant = torch.tensor([4.0])

        ema = build_targetflow_error(
            target,
            forward,
            previous_error=previous_state,
            previous_instant_error=previous_instant,
            sample_time=0.1,
            time_constant=0.5,
            error_gain=1.0,
            mode="ema",
        )
        two_tap = build_targetflow_error(
            target,
            forward,
            previous_error=previous_state,
            previous_instant_error=previous_instant,
            sample_time=0.1,
            time_constant=0.5,
            error_gain=1.0,
            mode="two_tap",
        )

        self.assertTrue(torch.allclose(ema, torch.tensor([7.6])))
        self.assertTrue(torch.allclose(two_tap, torch.tensor([5.2])))

        legacy_lag1 = build_targetflow_error(
            target,
            forward,
            previous_error=previous_state,
            previous_instant_error=previous_instant,
            sample_time=0.1,
            time_constant=0.5,
            error_gain=1.0,
            mode="lag1",
        )
        self.assertTrue(torch.equal(two_tap, legacy_lag1))

    def test_unstable_error_dynamics_are_rejected(self):
        common = {
            "target_output": torch.tensor([0.0]),
            "forward_output": torch.tensor([1.0]),
            "previous_error": torch.tensor([0.0]),
            "sample_time": 0.1,
            "time_constant": 0.5,
            "mode": "ema",
        }
        with self.assertRaisesRegex(ValueError, "Unstable"):
            build_targetflow_error(error_gain=-1.0, **common)
        with self.assertRaisesRegex(ValueError, "Unstable"):
            build_targetflow_error(error_gain=10.0, **common)


class RecursiveTargetFlowTest(unittest.TestCase):
    @staticmethod
    def _states():
        return [
            TargetFlowLayerState(layer_index=index, forward_output=torch.tensor([float(index * 10)]))
            for index in range(1, 6)
        ]

    @staticmethod
    def _feedback_modules():
        return nn.ModuleList(
            [TargetFlowFeedbackModule(nn.Identity()) for _ in range(4)]
        )

    def test_recursive_mode_propagates_future_top_target_through_all_layers(self):
        states = self._states()
        future_top_target = torch.tensor([99.0])

        run_backward_target_flow(
            states,
            self._feedback_modules(),
            top_target=future_top_target,
            mode="recursive",
        )

        for state in states:
            self.assertTrue(torch.equal(state.target_output, future_top_target))
        for index in range(4):
            self.assertIs(states[index].target_source, states[index + 1].target_output)

    def test_quasi_steady_mode_uses_current_forward_sources_below_top(self):
        states = self._states()

        run_backward_target_flow(
            states,
            self._feedback_modules(),
            top_target=torch.tensor([99.0]),
            mode="quasi_steady",
        )

        expected_targets = (20.0, 30.0, 40.0, 50.0, 99.0)
        self.assertEqual(
            tuple(float(state.target_output.item()) for state in states),
            expected_targets,
        )


if __name__ == "__main__":
    unittest.main()
