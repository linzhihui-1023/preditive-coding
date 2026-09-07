"""C-V7 semantic-first runtime corrections and fast-protocol wrapper.

中文：C-V7 语义优先效用目标与快速协议运行层。

This wrapper keeps the C-V7 architecture unchanged and corrects the training
objective so utility is aligned with the final hard semantic decision:

    A_k = I[history_k predicts GT] - I[current predicts GT]

Therefore A_k is exactly +1 (semantic rescue), 0 (no semantic net change), or
-1 (semantic damage).  RAFT temporal ranking is allowed only inside the same
semantic-utility level and can never override rescue/damage ordering.
"""

import json
from pathlib import Path

import torch

from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v6_error_centric_multihypothesis as c_v6,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v7_error_utility_estimator as c_v7,
)


DEV3 = ("0002", "0010", "0018")
SEMANTIC_TIE_DELTA = 0.10
UTILITY_TARGET_NAME = "I[history predicts GT] - I[current predicts GT]"


def build_semantic_correctness_utility_targets(
    current_logits,
    candidate_rows,
    history_length,
    current_gt_cpu,
    previous_cv3_logits,
    teacher_full,
):
    """Build {-1,0,+1} semantic utility plus RAFT tie-break evidence.

    +1: Current is wrong and this history hypothesis is correct.
     0: history does not change semantic correctness relative to Current.
    -1: Current is correct and this history hypothesis is wrong.

    Invalid history is excluded through valid_mask and receives zero target.
    """
    gt = current_gt_cpu.to(current_logits.device, non_blocking=True)
    valid_gt = gt != c_v5.IGNORE_LABEL
    current_pred = current_logits.argmax(1)[0]
    current_correct = current_pred == gt

    gains = []
    validities = []
    history_predictions = []
    for index in range(history_length):
        if index < len(candidate_rows):
            row = candidate_rows[index]
            valid = row["valid_full"][0].bool() & valid_gt
            history_pred = row["logits"].argmax(1)[0]
            history_correct = history_pred == gt
            gain = history_correct.to(torch.float32) - current_correct.to(torch.float32)
            gain = torch.where(valid, gain, torch.zeros_like(gain))
        else:
            valid = torch.zeros_like(valid_gt)
            gain = torch.zeros_like(current_pred, dtype=torch.float32)
            history_pred = current_pred
        gains.append(gain)
        validities.append(valid)
        history_predictions.append(history_pred)

    semantic_gains = torch.stack(gains, dim=0).unsqueeze(0)
    valid_mask = torch.stack(validities, dim=0).unsqueeze(0)

    # RAFT is training-only. It ranks candidates only after semantic utility
    # has tied, so it cannot turn a rescue (+1) into a neutral/damage choice.
    temporal_valid = torch.zeros_like(valid_gt).unsqueeze(0).unsqueeze(0)
    current_match = torch.zeros_like(valid_gt)
    history_matches = [torch.zeros_like(valid_gt) for _ in range(history_length)]
    if teacher_full is not None and previous_cv3_logits is not None:
        teacher_previous, teacher_valid = c_v5._warp_previous_prediction_with_raft(
            previous_cv3_logits,
            teacher_full,
        )
        teacher_previous = teacher_previous[0]
        teacher_valid = teacher_valid[0].bool() & valid_gt
        temporal_valid = teacher_valid.unsqueeze(0).unsqueeze(0)
        current_match = teacher_valid & (current_pred == teacher_previous)
        history_matches = [
            teacher_valid
            & validities[index]
            & (history_predictions[index] == teacher_previous)
            for index in range(history_length)
        ]

    temporal_matches = torch.stack(
        [current_match, *history_matches],
        dim=0,
    ).unsqueeze(0)
    return {
        "semantic_gains": semantic_gains,
        "valid_mask": valid_mask,
        "temporal_matches": temporal_matches,
        "temporal_valid": temporal_valid,
    }


def _pop_protocol(argv):
    argv = list(argv or [])
    protocol = "full9"
    if "--protocol" in argv:
        index = argv.index("--protocol")
        if index + 1 >= len(argv):
            raise ValueError("Missing value for --protocol")
        protocol = argv[index + 1].lower()
        del argv[index : index + 2]
    if protocol not in {"dev3", "full9"}:
        raise ValueError("--protocol must be dev3 or full9")
    return protocol, argv


def _inject_isolated_dev3_outputs(argv):
    argv = list(argv)
    if "--output" not in argv:
        argv.extend(["--output", c_v7.OUTPUT_DEFAULT + "_dev3"])
    if "--result-output" not in argv:
        argv.extend(["--result-output", c_v7.RESULT_DEFAULT + "_dev3"])
    return argv


def _rewrite_semantic_first_metadata(output_dir, result_dir, protocol):
    """Correct inherited C-V6/C-V7 metadata after a completed run."""
    result_dir = Path(result_dir)
    for path in [*sorted(result_dir.glob("epoch_*.json")), result_dir / "summary.json"]:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        payload["protocol"] = protocol
        rows = []
        if path.name == "summary.json":
            if isinstance(payload.get("best"), dict):
                rows.append(payload["best"])
            rows.extend(row for row in payload.get("history", []) if isinstance(row, dict))
            architecture = payload.setdefault("architecture", {})
            architecture["utility_target"] = UTILITY_TARGET_NAME
            architecture["utility_levels"] = [-1, 0, 1]
            architecture["semantic_tie_delta"] = SEMANTIC_TIE_DELTA
        else:
            rows.append(payload)
        for row in rows:
            train = row.get("train")
            if isinstance(train, dict):
                utility_training = train.get("utility_training")
                if isinstance(utility_training, dict):
                    utility_training["target"] = UTILITY_TARGET_NAME
                    utility_training["semantic_tie_delta"] = SEMANTIC_TIE_DELTA
                    utility_training["utility_levels"] = [-1, 0, 1]
        path.write_text(json.dumps(payload, indent=2))

    checkpoint_path = Path(output_dir) / "best.pt"
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu")
        payload["protocol"] = protocol
        architecture = payload.setdefault("architecture", {})
        architecture["utility_target"] = UTILITY_TARGET_NAME
        architecture["utility_levels"] = [-1, 0, 1]
        architecture["semantic_tie_delta"] = SEMANTIC_TIE_DELTA
        torch.save(payload, checkpoint_path)


def main(argv=None):
    protocol, forwarded = _pop_protocol(argv)
    if protocol == "dev3":
        forwarded = _inject_isolated_dev3_outputs(forwarded)

    output_dir = c_v7._arg_value(forwarded, "--output", c_v7.OUTPUT_DEFAULT)
    result_dir = c_v7._arg_value(forwarded, "--result-output", c_v7.RESULT_DEFAULT)

    original_target_builder = c_v7._build_utility_targets
    original_tie_delta = c_v7.SEMANTIC_TIE_DELTA
    original_full9 = c_v5.FULL9
    original_sequence_groups = c_v6.sequence_groups
    original_rewrite = c_v7._rewrite_artifacts

    def rewrite_with_correct_metadata(output, result):
        original_rewrite(output, result)
        _rewrite_semantic_first_metadata(output, result, protocol)

    if protocol == "dev3":
        def dev3_sequence_groups(dataset):
            groups = original_sequence_groups(dataset)
            missing = [sequence for sequence in DEV3 if sequence not in groups]
            if missing:
                available = sorted(groups.keys())
                raise RuntimeError(
                    "DEV3 requires KITTI-STEP sequences 0002/0010/0018 in this split; "
                    f"missing={missing}, available={available}"
                )
            return {sequence: groups[sequence] for sequence in DEV3}
    else:
        dev3_sequence_groups = original_sequence_groups

    try:
        c_v7._build_utility_targets = build_semantic_correctness_utility_targets
        c_v7.SEMANTIC_TIE_DELTA = SEMANTIC_TIE_DELTA
        c_v7._rewrite_artifacts = rewrite_with_correct_metadata
        c_v5.FULL9 = DEV3 if protocol == "dev3" else original_full9
        c_v6.sequence_groups = dev3_sequence_groups
        c_v7.main(forwarded)
    finally:
        c_v7._build_utility_targets = original_target_builder
        c_v7.SEMANTIC_TIE_DELTA = original_tie_delta
        c_v7._rewrite_artifacts = original_rewrite
        c_v5.FULL9 = original_full9
        c_v6.sequence_groups = original_sequence_groups
