import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from predify2021.mce_scores.evaluate_kitti_aligned_temporal_difference import (
    ALIGNED_DIFFERENCE_DEFINITION,
    TEMPORAL_ALIGNMENT_DEFINITION,
    evaluate_gate,
    validate_checkpoint,
)
from predify2021.mce_scores.evaluate_kitti_stage4_same_drive_diagnostic import (
    validate_same_drive_checkpoint,
)
from predify2021.mce_scores.evaluate_kitti_stage4_multidrive import (
    FROZEN_TEST_DRIVES,
    TRAIN_DRIVES,
    VAL_DRIVES,
    claim_frozen_test_access,
    evaluate_drive,
    finalize_frozen_test_receipt,
    validate_multidrive_checkpoint,
)


REVISION = "a" * 40
STAGE = 4


def _checkpoint():
    return {
        "checkpoint_kind": "best_val_future_feature_mse",
        "selected_epoch": {"epoch": 1},
        "config": {
            "prediction_task": "future_feature",
            "git_revision": REVISION,
            "future_feature_stage": STAGE,
            "target_flow_top_stage": 5,
            "future_feature_channels": 512,
            "target_flow_top_target_definition": "T_TF=F_next_stage5",
            "future_prediction_target_definition": "T_future=F_next_stage4",
            "future_target_separated_from_target_flow_top": True,
            "future_feature_history_mode": "aligned_difference",
            "future_feature_temporal_fusion_mode": "none",
            "future_feature_temporal_fusion_architecture": "none",
            "future_feature_temporal_alignment": TEMPORAL_ALIGNMENT_DEFINITION,
            "future_feature_history_definition": ALIGNED_DIFFERENCE_DEFINITION,
            "future_feature_prediction_form": "current_residual",
            "future_feature_predictor_kernel_size": 1,
            "future_motion_radius": 1,
            "future_motion_radius_units": "stage4_feature_cells",
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
        validation = validate_checkpoint(_checkpoint(), REVISION, STAGE)
        self.assertEqual(validation["history_mode"], "aligned_difference")
        self.assertEqual(validation["fusion_mode"], "none")
        self.assertEqual(validation["future_feature_stage"], 4)
        self.assertEqual(validation["target_flow_top_stage"], 5)

    def test_temporal_fusion_checkpoint_is_rejected(self):
        checkpoint = _checkpoint()
        checkpoint["config"]["future_feature_temporal_fusion_mode"] = (
            "two_frame_residual"
        )
        with self.assertRaisesRegex(ValueError, "none"):
            validate_checkpoint(checkpoint, REVISION, STAGE)

    def test_temporal_fusion_parameters_are_rejected(self):
        checkpoint = _checkpoint()
        checkpoint["config"]["optimized_parameter_names"] += (
            "temporal_fusion_module.0.weight",
        )
        with self.assertRaisesRegex(ValueError, "only the Future Predictor"):
            validate_checkpoint(checkpoint, REVISION, STAGE)

    def test_wrong_difference_definition_is_rejected(self):
        checkpoint = _checkpoint()
        checkpoint["config"]["future_feature_history_definition"] = (
            "use_aligned_previous_directly"
        )
        with self.assertRaisesRegex(ValueError, "D_current"):
            validate_checkpoint(checkpoint, REVISION, STAGE)

    def test_historical_future_warp_semantics_are_rejected(self):
        checkpoint = _checkpoint()
        checkpoint["config"]["future_feature_temporal_alignment"] = (
            "historical_warp_current_toward_future"
        )
        with self.assertRaisesRegex(ValueError, "target_coordinates"):
            validate_checkpoint(checkpoint, REVISION, STAGE)

    def test_checkpoint_for_a_different_prediction_stage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "future_feature_stage"):
            validate_checkpoint(_checkpoint(), REVISION, 5)

    def test_gate_requires_all_three_metrics_to_improve(self):
        passing_metrics = {
            "feature_mse": {"mean": 0.079},
            "feature_cosine": {"mean": 0.91},
            "normalized_feature_error": {"mean": 0.39},
            "copy_mse": {"mean": 0.08},
            "copy_cosine": {"mean": 0.90},
            "copy_normalized_feature_error": {"mean": 0.40},
        }
        passing_gate = evaluate_gate(passing_metrics)
        self.assertTrue(passing_gate["all_three_pass"])
        self.assertEqual(
            passing_gate["reference_scope"],
            "same_prediction_stage_same_frame_stream_copy_current",
        )
        passing_metrics["feature_cosine"]["mean"] = 0.90
        gate = evaluate_gate(passing_metrics)
        self.assertFalse(gate["all_three_pass"])
        self.assertFalse(gate["checks"]["feature_cosine_above_copy"])

    def test_same_drive_checkpoint_locks_chronological_test_policy(self):
        checkpoint = _checkpoint()
        checkpoint["config"].update(
            {
                "formal_split": False,
                "same_drive_split": False,
                "same_drive_three_way_split": True,
                "same_drive_drive": "drive_0005",
                "train_drives": ("drive_0005",),
                "val_drives": ("drive_0005",),
                "train_fraction": 0.6,
                "val_fraction": 0.2,
                "test_fraction": 0.2,
                "test_selection_role": (
                    "unseen_until_after_best_validation_checkpoint_selection"
                ),
                "shuffle_train_pairs": False,
                "shuffle_val_pairs": False,
                "same_drive_split_metadata": {
                    "split_name": "chronological_raw_frames_60_20_20",
                    "chronological_order": ("train", "val", "test"),
                    "shared_raw_frame_count": 0,
                    "shuffle": False,
                },
            }
        )
        validation = validate_same_drive_checkpoint(
            checkpoint,
            expected_revision=REVISION,
            expected_drive="drive_0005",
        )
        self.assertEqual(validation["future_feature_stage"], 4)

        checkpoint["config"]["test_selection_role"] = "used_for_epoch_selection"
        with self.assertRaisesRegex(ValueError, "test_selection_role"):
            validate_same_drive_checkpoint(
                checkpoint,
                expected_revision=REVISION,
                expected_drive="drive_0005",
            )

    def test_multidrive_checkpoint_uses_exact_train_val_and_no_test(self):
        checkpoint = _checkpoint()
        checkpoint["config"].update(
            {
                "formal_split": True,
                "same_drive_split": False,
                "same_drive_three_way_split": False,
                "stream_mode": True,
                "reset_each_frame": False,
                "shuffle_train_pairs": False,
                "shuffle_val_pairs": False,
                "train_drives": TRAIN_DRIVES,
                "val_drives": VAL_DRIVES,
            }
        )
        validation = validate_multidrive_checkpoint(checkpoint, REVISION)
        self.assertEqual(validation["future_feature_stage"], 4)

        checkpoint["config"]["test_drives"] = FROZEN_TEST_DRIVES
        with self.assertRaisesRegex(ValueError, "must not contain"):
            validate_multidrive_checkpoint(checkpoint, REVISION)

    def test_frozen_test_receipt_allows_only_one_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"fixed checkpoint")
            receipt, payload = claim_frozen_test_access(checkpoint, REVISION)
            self.assertTrue(receipt.is_file())
            self.assertEqual(payload["frozen_test_drives"], FROZEN_TEST_DRIVES)
            completed = finalize_frozen_test_receipt(
                receipt,
                payload,
                Path(directory) / "summary.json",
            )
            self.assertEqual(completed["status"], "completed")
            with self.assertRaisesRegex(RuntimeError, "already claimed"):
                claim_frozen_test_access(checkpoint, REVISION)

    def test_multidrive_replay_resets_at_every_valid_time_segment(self):
        class SegmentedDataset:
            valid_sample_segments = ((0, 1), (2, 3, 4))

            def __len__(self):
                return 5

            def __getitem__(self, index):
                return index

        def fake_evaluate_split(model, dataset, split, drive, fixed_dt_s):
            return [{"sample_index": index} for index in range(len(dataset))]

        with patch(
            "predify2021.mce_scores.evaluate_kitti_stage4_multidrive."
            "evaluate_split",
            side_effect=fake_evaluate_split,
        ) as replay:
            rows = evaluate_drive(
                model=object(),
                dataset=SegmentedDataset(),
                split="test",
                drive="drive_0051",
                fixed_dt_s=0.1035,
            )

        self.assertEqual(replay.call_count, 2)
        self.assertEqual([row["sample_index"] for row in rows], list(range(5)))
        self.assertEqual([row["segment_index"] for row in rows], [0, 0, 1, 1, 1])
        self.assertEqual(
            [row["segment_sample_index"] for row in rows],
            [0, 1, 0, 1, 2],
        )


if __name__ == "__main__":
    unittest.main()
