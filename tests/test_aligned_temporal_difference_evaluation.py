import unittest

from predify2021.mce_scores.evaluate_kitti_aligned_temporal_difference import (
    ALIGNED_DIFFERENCE_DEFINITION,
    COPY_CURRENT_REFERENCE,
    TEMPORAL_ALIGNMENT_DEFINITION,
    evaluate_gate,
    validate_checkpoint,
)


REVISION = "a" * 40


def _checkpoint():
    return {
        "checkpoint_kind": "best_val_future_feature_mse",
        "selected_epoch": {"epoch": 1},
        "config": {
            "prediction_task": "future_feature",
            "git_revision": REVISION,
            "future_feature_history_mode": "aligned_difference",
            "future_feature_temporal_fusion_mode": "none",
            "future_feature_temporal_fusion_architecture": "none",
            "future_feature_temporal_alignment": TEMPORAL_ALIGNMENT_DEFINITION,
            "future_feature_history_definition": ALIGNED_DIFFERENCE_DEFINITION,
            "future_feature_prediction_form": "current_residual",
            "future_feature_predictor_kernel_size": 1,
            "future_motion_radius": 1,
            "future_motion_patch_size": 3,
            "pretrained": True,
            "train_backbone": False,
            "feedback_decoder_trainable": False,
            "top_target_source": "student_self",
            "stream_mode": True,
            "reset_each_frame": False,
            "shuffle_train_pairs": False,
            "shuffle_val_pairs": False,
            "local_reconstruction_weight": 0.0,
            "optimizer_created": True,
            "optimized_parameter_names": (
                "future_feature_predictor.0.weight",
                "future_feature_predictor.2.weight",
            ),
        },
    }


class AlignedTemporalDifferenceEvaluationTest(unittest.TestCase):
    def test_checkpoint_contract_is_strictly_aligned_difference(self):
        validation = validate_checkpoint(_checkpoint(), REVISION)
        self.assertEqual(validation["history_mode"], "aligned_difference")
        self.assertEqual(validation["fusion_mode"], "none")

    def test_temporal_fusion_checkpoint_is_rejected(self):
        checkpoint = _checkpoint()
        checkpoint["config"]["future_feature_temporal_fusion_mode"] = (
            "two_frame_residual"
        )
        with self.assertRaisesRegex(ValueError, "none"):
            validate_checkpoint(checkpoint, REVISION)

    def test_temporal_fusion_parameters_are_rejected(self):
        checkpoint = _checkpoint()
        checkpoint["config"]["optimized_parameter_names"] += (
            "temporal_fusion_module.0.weight",
        )
        with self.assertRaisesRegex(ValueError, "only the Future Predictor"):
            validate_checkpoint(checkpoint, REVISION)

    def test_wrong_difference_definition_is_rejected(self):
        checkpoint = _checkpoint()
        checkpoint["config"]["future_feature_history_definition"] = (
            "use_aligned_previous_directly"
        )
        with self.assertRaisesRegex(ValueError, "D_current"):
            validate_checkpoint(checkpoint, REVISION)

    def test_historical_future_warp_semantics_are_rejected(self):
        checkpoint = _checkpoint()
        checkpoint["config"]["future_feature_temporal_alignment"] = (
            "historical_warp_current_toward_future"
        )
        with self.assertRaisesRegex(ValueError, "target_coordinates"):
            validate_checkpoint(checkpoint, REVISION)

    def test_gate_requires_all_three_metrics_to_improve(self):
        passing_metrics = {
            "feature_mse": {"mean": COPY_CURRENT_REFERENCE["feature_mse"] - 0.001},
            "feature_cosine": {
                "mean": COPY_CURRENT_REFERENCE["feature_cosine"] + 0.001
            },
            "normalized_feature_error": {
                "mean": COPY_CURRENT_REFERENCE["normalized_feature_error"] - 0.001
            },
        }
        self.assertTrue(evaluate_gate(passing_metrics)["all_three_pass"])
        passing_metrics["feature_cosine"]["mean"] = COPY_CURRENT_REFERENCE[
            "feature_cosine"
        ]
        gate = evaluate_gate(passing_metrics)
        self.assertFalse(gate["all_three_pass"])
        self.assertFalse(gate["checks"]["feature_cosine_above_copy"])


if __name__ == "__main__":
    unittest.main()
