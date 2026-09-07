import inspect
import unittest

from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v8_direct_error_correction as c_v8,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_multihypothesis_error_correction import (
    MultiHypothesisErrorDirectCorrection,
)


class C_V8DirectCorrectionTrainingContractTest(unittest.TestCase):
    def test_trainer_imports_and_zero_step_runs_on_cpu(self):
        module = MultiHypothesisErrorDirectCorrection(
            num_classes=3,
            history_length=2,
            hidden_channels=8,
            current_state_channels=8,
            branch_channels=8,
        )
        row = c_v8._zero_step_check(module)
        self.assertEqual(row["delta_logits_abs_max"], 0.0)
        self.assertTrue(row["c_v8_equals_c_v3"])

    def test_checkpoint_selection_is_miou_first(self):
        metrics_a = {"c_v8": {"mIoU": 0.60, "mTC": 0.80}}
        metrics_b = {"c_v8": {"mIoU": 0.61, "mTC": 0.70}}
        self.assertGreater(c_v8._selection_key(metrics_b), c_v8._selection_key(metrics_a))

    def test_training_path_has_no_candidate_selection_or_utility_objective(self):
        source = inspect.getsource(c_v8._train_sequence)
        forbidden = (
            "selector_logits",
            "_apply_selection",
            "utility_regression",
            "abstention",
            "semantic_correctness_utility",
        )
        for token in forbidden:
            self.assertNotIn(token, source)
        self.assertIn("segmentation_ce", source)
        self.assertIn("_strict_temporal_l1", source)

    def test_metrics_contract_contains_direct_c_v8_output(self):
        self.assertIn("c_v8", c_v8.CANDIDATES)
        self.assertIn("c_v3_base", c_v8.CANDIDATES)
        self.assertNotIn("c_v7", c_v8.CANDIDATES)

    def test_development_gate_is_half_percentage_point(self):
        self.assertEqual(c_v8.DEV_MIOU_GAIN_TARGET, 0.005)


if __name__ == "__main__":
    unittest.main()
