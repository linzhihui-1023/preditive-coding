"""Inference-only correction-scale backtrace for Stage C-Balance.

The Epoch-3 Stage C-Balance checkpoint is kept fixed.  Only the applied Z4
correction is scaled at inference time, while recurrent states and all other
evaluation logic remain unchanged.
"""
import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_raft_error_memory import FrozenRAFT
from predify2021.mce_scores.train_kitti_step_z4_only_stage_c import (
    DEV3,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STAGE_P_DEFAULT,
    FAST_B_DEFAULT,
    annotate_full9_deltas,
    evaluate,
    load_stage_c,
)

CHECKPOINT_DEFAULT = "/home/lin/predify/experiments/kitti_step_z4_only_stage_c_balance/best.pt"
RESULT_DEFAULT = "results/kitti_step_z4_only_stage_c_balance_scale_backtrace"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--stage-p-checkpoint", default=STAGE_P_DEFAULT)
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_DEFAULT)
    parser.add_argument("--dynamics-checkpoint", default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("Scale backtrace requires CUDA")

    load_args = Namespace(
        fast_b_checkpoint=args.fast_b_checkpoint,
        stage_p_checkpoint=args.stage_p_checkpoint,
        dynamics_checkpoint=args.dynamics_checkpoint,
    )
    model, predictor, _, _ = load_stage_c(load_args)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    predictor.load_state_dict(payload["model_state_dict"], strict=True)
    model.multi_layer_adapter.output_adapters[3].load_state_dict(
        payload["c4_output_adapter_state_dict"], strict=True
    )
    model.host_conditioned_writebacks["3"].load_state_dict(
        payload["c4_writeback_state_dict"], strict=True
    )

    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root), "val"
    )
    groups = sequence_groups(dataset)
    raft = FrozenRAFT()
    results = {}
    for alpha in ALPHAS:
        metrics = evaluate(model, predictor, groups, raft, correction_scale=alpha)
        results[str(alpha)] = annotate_full9_deltas(metrics)
        print(json.dumps({"alpha": alpha, "metrics": results[str(alpha)]}, sort_keys=True), flush=True)

    result = {
        "experiment": "Stage C-Balance correction scale backtrace",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": payload.get("epoch"),
        "inference_only": True,
        "alphas": list(ALPHAS),
        "sequences": list(groups),
        "dev3_reference_sequences": list(DEV3),
        "results": results,
    }
    output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text(
        "# Stage C-Balance correction scale backtrace\n\n"
        "Inference-only Full9 evaluation of fixed Epoch-3 Stage C-Balance. "
        "Only `Z4_corrected = Z4 + alpha * DeltaZ4` changes; alpha=0 is Host "
        "and alpha=1 is the unchanged Stage C-Balance path.\n"
    )
    print(json.dumps({"result": str(output / "summary.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
