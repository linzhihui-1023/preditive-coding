import unittest

from predify2021.mce_scores.evaluate_kitti_warp_residual_matrix import (
    validate_checkpoint,
)


REVISION = "a" * 40


def _checkpoint(label, *, revision=REVISION, form=None, history=None):
    defaults = {
        "copy_current": ("copy_current", "current_residual"),
        "historical_warp": ("none", "historical_warp"),
        "warp_residual": ("none", "historical_warp_residual"),
    }
    default_history, default_form = defaults[label]
    return {
        "config": {
            "prediction_task": "future_feature",
            "future_feature_history_mode": history or default_history,
            "future_feature_prediction_form": form or default_form,
            "future_feature_predictor_kernel_size": 1,
            "future_motion_radius": 1,
            "future_motion_patch_size": 3,
            "git_revision": revision,
        },
        "selected_epoch": {"epoch": 2},
        "checkpoint_kind": "best_val_future_feature_mse",
    }


class WarpResidualCheckpointValidationTest(unittest.TestCase):
    def test_accepts_exact_three_group_contract(self):
        for label in ("copy_current", "historical_warp", "warp_residual"):
            with self.subTest(label=label):
                validated = validate_checkpoint(label, _checkpoint(label), REVISION)
                self.assertEqual(validated["motion_radius"], 1)
                self.assertEqual(validated["motion_patch_size"], 3)

    def test_rejects_prediction_form_mismatch(self):
        with self.assertRaisesRegex(ValueError, "history/form"):
            validate_checkpoint(
                "warp_residual",
                _checkpoint("warp_residual", form="current_residual"),
                REVISION,
            )

    def test_rejects_revision_mismatch(self):
        with self.assertRaisesRegex(ValueError, "checkpoint revision"):
            validate_checkpoint(
                "historical_warp",
                _checkpoint("historical_warp", revision="b" * 40),
                REVISION,
            )

    def test_rejects_non_best_checkpoint(self):
        checkpoint = _checkpoint("copy_current")
        checkpoint["checkpoint_kind"] = "final_student"
        with self.assertRaisesRegex(ValueError, "best-validation"):
            validate_checkpoint("copy_current", checkpoint, REVISION)


if __name__ == "__main__":
    unittest.main()
