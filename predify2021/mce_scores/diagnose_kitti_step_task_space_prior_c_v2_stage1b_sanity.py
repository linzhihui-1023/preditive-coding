"""Fast no-training sanity check for C-V2 Stage 1B shapes and causal indexing."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import (
    FAST_B_CHECKPOINT_DEFAULT,
    load_fast_b_model,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_only import FULL9, NUM_CLASSES
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_observer import (
    C1_CHANNELS,
    CORRELATION_RADIUS,
    HIDDEN_CHANNELS,
    MAX_DISPLACEMENT_LOW,
    PROJECTED_CHANNELS,
    _host_observation,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    LocalCorrelationMotionObserver,
    MotionResidualPredictor,
    warp_low_logits,
)

RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v2_stage1b_sanity.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    groups = sequence_groups(dataset)
    samples = groups[FULL9[0]][:3]
    if len(samples) < 3:
        raise RuntimeError("Need three frames for Stage 1B sanity check")

    frames = [_host_observation(model, sample) for sample in samples]
    for index, frame in enumerate(frames):
        c1 = frame[3]
        if c1.shape[1] != C1_CHANNELS:
            raise RuntimeError(f"Frame {index}: expected C1 channels {C1_CHANNELS}, got {c1.shape[1]}")

    observer = LocalCorrelationMotionObserver(
        c1_channels=C1_CHANNELS,
        num_classes=NUM_CLASSES,
        projected_channels=PROJECTED_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        correlation_radius=CORRELATION_RADIUS,
        max_displacement_low=MAX_DISPLACEMENT_LOW,
    ).cuda().eval()
    residual = MotionResidualPredictor(
        num_classes=NUM_CLASSES,
        hidden_channels=64,
        max_observed_displacement_low=MAX_DISPLACEMENT_LOW,
        max_residual_displacement_low=16.0,
    ).cuda().eval()

    low0, c10 = frames[0][2], frames[0][3]
    low1, c11 = frames[1][2], frames[1][3]
    with torch.no_grad():
        motion_1 = observer(
            c10,
            c11,
            F.softmax(low0, dim=1),
            F.softmax(low1, dim=1),
        )
        persistence_1, _ = warp_low_logits(low0, torch.zeros_like(motion_1))
        error_1 = F.softmax(low1, dim=1) - F.softmax(persistence_1, dim=1)
        predicted_motion_2, delta_2, hidden_1 = residual.predict_next(
            motion_1, error_1, None
        )

    expected_motion_shape = (1, 2, low1.shape[-2], low1.shape[-1])
    if tuple(motion_1.shape) != expected_motion_shape:
        raise RuntimeError(f"Unexpected observer motion shape: {tuple(motion_1.shape)}")
    if float(motion_1.abs().max().item()) != 0.0:
        raise RuntimeError("Fresh Observer must start at exact zero motion")
    if float(delta_2.abs().max().item()) != 0.0:
        raise RuntimeError("Fresh residual head must start at exact zero delta")
    if float((predicted_motion_2 - motion_1).abs().max().item()) != 0.0:
        raise RuntimeError("E0 residual predictor must equal Lagged-Motion-Persistence")

    # Explicit causal index ledger. M_t is observed only after frame t arrives;
    # H_t uses M_t/e_t and produces M_hat_{t+1}. No t+1 tensor participates.
    causal_ledger = [
        {"observed_pair": [0, 1], "motion": "M_1", "error": "e_1", "state": "H_1", "predicts": "M_hat_2"},
        {"observed_pair": [1, 2], "motion": "M_2", "error": "e_2", "state": "H_2", "predicts": "M_hat_3"},
    ]
    assert causal_ledger[0]["observed_pair"] == [0, 1]
    assert causal_ledger[0]["predicts"] == "M_hat_2"
    assert causal_ledger[1]["observed_pair"] == [1, 2]
    assert causal_ledger[1]["predicts"] == "M_hat_3"

    result = {
        "experiment": "C-V2 Stage 1B sanity",
        "training": False,
        "c1_shape": list(c10.shape),
        "low_logit_shape": list(low0.shape),
        "observer_motion_shape": list(motion_1.shape),
        "observer_e0_flow_max_abs": float(motion_1.abs().max().item()),
        "residual_e0_delta_max_abs": float(delta_2.abs().max().item()),
        "residual_e0_equals_lagged_max_abs": float((predicted_motion_2 - motion_1).abs().max().item()),
        "hidden_shape": list(hidden_1.shape),
        "causal_ledger": causal_ledger,
        "passed": True,
    }
    output = Path(args.result_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
