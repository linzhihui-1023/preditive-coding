import unittest

import torch

from predify2021.mce_scores.diagnose_kitti_future_feature_delta import (
    validate_current_only_checkpoint,
)


def _checkpoint(
    task="future_feature",
    history="none",
    kernel=1,
    record_kernel=True,
    prediction_form="current_residual",
):
    config = {
        "prediction_task": task,
        "future_feature_history_mode": history,
        "future_feature_prediction_form": prediction_form,
    }
    if record_kernel:
        config["future_feature_predictor_kernel_size"] = kernel
    return {
        "config": config,
        "state_dict": {
            "future_feature_predictor.0.weight": torch.empty(
                1024,
                1024,
                kernel,
                kernel,
            )
        },
    }


class FutureFeatureDiagnosticCheckpointTest(unittest.TestCase):
    def test_accepts_verified_current_only_checkpoint(self):
        validation = validate_current_only_checkpoint(_checkpoint(kernel=3))
        self.assertEqual(validation["future_feature_predictor_kernel_size"], 3)
        self.assertEqual(
            validation["kernel_validation"],
            "config_verified_against_weight",
        )

    def test_rejects_wrong_task_or_history(self):
        with self.assertRaisesRegex(ValueError, "prediction_task"):
            validate_current_only_checkpoint(_checkpoint(task="motion"))
        with self.assertRaisesRegex(ValueError, "Current-only"):
            validate_current_only_checkpoint(_checkpoint(history="recursive"))
        with self.assertRaisesRegex(ValueError, "current_residual"):
            validate_current_only_checkpoint(
                _checkpoint(prediction_form="historical_warp_residual")
            )

    def test_rejects_kernel_config_weight_mismatch(self):
        checkpoint = _checkpoint(kernel=3)
        checkpoint["config"]["future_feature_predictor_kernel_size"] = 1
        with self.assertRaisesRegex(ValueError, "kernel mismatch"):
            validate_current_only_checkpoint(checkpoint)

    def test_legacy_kernel_inference_is_limited_to_one_by_one(self):
        validation = validate_current_only_checkpoint(
            _checkpoint(kernel=1, record_kernel=False)
        )
        self.assertEqual(
            validation["kernel_validation"],
            "legacy_inferred_from_1x1_weight",
        )
        with self.assertRaisesRegex(ValueError, "legacy inference"):
            validate_current_only_checkpoint(
                _checkpoint(kernel=3, record_kernel=False)
            )


if __name__ == "__main__":
    unittest.main()
