import unittest

from predify2021.mce_scores.evaluate_kitti_temporal_fusion_matrix import (
    compare_conditions,
    validate_checkpoint,
)


REVISION = "0123456789abcdef"


def _checkpoint(label):
    settings = {
        "copy_current": ("copy_current", "none", False, ()),
        "current_only": (
            "none",
            "none",
            True,
            ("future_feature_predictor.0.weight",),
        ),
        "temporal_fusion": (
            "none",
            "two_frame_residual",
            True,
            (
                "future_feature_predictor.0.weight",
                "temporal_fusion_module.0.weight",
            ),
        ),
    }
    history, fusion, optimizer, names = settings[label]
    return {
        "checkpoint_kind": "best_val_future_feature_mse",
        "selected_epoch": {"epoch": 2},
        "config": {
            "prediction_task": "future_feature",
            "future_feature_history_mode": history,
            "future_feature_temporal_fusion_mode": fusion,
            "future_feature_prediction_form": "current_residual",
            "future_feature_predictor_kernel_size": 1,
            "git_revision": REVISION,
            "pretrained": True,
            "train_backbone": False,
            "feedback_decoder_trainable": False,
            "top_target_source": "student_self",
            "stream_mode": True,
            "reset_each_frame": False,
            "shuffle_train_pairs": False,
            "shuffle_val_pairs": False,
            "local_reconstruction_weight": 0.0,
            "optimizer_created": optimizer,
            "optimized_parameter_names": names,
        },
    }


class TemporalFusionEvaluationTest(unittest.TestCase):
    def test_three_checkpoint_contracts_are_accepted(self):
        for label in ("copy_current", "current_only", "temporal_fusion"):
            validation = validate_checkpoint(label, _checkpoint(label), REVISION)
            self.assertEqual(validation["checkpoint_revision"], REVISION)

    def test_fusion_checkpoint_rejects_current_only_trainable_scope(self):
        checkpoint = _checkpoint("temporal_fusion")
        checkpoint["config"]["optimized_parameter_names"] = (
            "future_feature_predictor.0.weight",
        )
        with self.assertRaisesRegex(ValueError, "does not optimize"):
            validate_checkpoint("temporal_fusion", checkpoint, REVISION)

    def test_checkpoint_rejects_unfrozen_feedback(self):
        checkpoint = _checkpoint("current_only")
        checkpoint["config"]["feedback_decoder_trainable"] = True
        with self.assertRaisesRegex(ValueError, "feedback_decoder_trainable"):
            validate_checkpoint("current_only", checkpoint, REVISION)

    def test_comparisons_use_all_three_requested_metrics(self):
        matrix = {}
        values = {
            "copy_current": (4.0, 0.4, 2.0),
            "current_only": (3.0, 0.5, 1.5),
            "temporal_fusion": (2.0, 0.6, 1.0),
        }
        for split in ("train", "val"):
            for condition, (mse, cosine, normalized) in values.items():
                matrix[f"{condition}/{split}"] = {
                    "metrics": {
                        "feature_mse": {"mean": mse},
                        "feature_cosine": {"mean": cosine},
                        "normalized_feature_error": {"mean": normalized},
                    }
                }
        comparisons = compare_conditions(matrix)
        fusion_vs_current = comparisons["val"]["temporal_fusion_vs_current_only"]
        self.assertEqual(fusion_vs_current["feature_mse_reduction_fraction"], 1 / 3)
        self.assertAlmostEqual(fusion_vs_current["feature_cosine_increase"], 0.1)
        self.assertEqual(
            fusion_vs_current["normalized_error_reduction_fraction"], 1 / 3
        )


if __name__ == "__main__":
    unittest.main()
