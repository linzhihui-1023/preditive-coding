"""Formal Semantic V3 validation diagnostic.

It evaluates the trained V3 checkpoint on the published KITTI-STEP validation
split, preserving the existing blur/warm-up protocol and frozen writeback path.
The decoder sanity check is intentionally separate from the decoder-free fast
diagnostic, but both use the same recurrent causal order and frame-0 state.
"""

import argparse
import csv
import json
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.kitti_step_persistent_blur import BLUR_KERNEL_SIZE
from predify2021.mce_scores.role_separated_dynamic_error_correction import ADAPTER_CHECKPOINT_DEFAULT, STATIC_CHECKPOINT_DEFAULT, WRITEBACK_CHECKPOINT_DEFAULT, load_components
from predify2021.mce_scores.train_kitti_step_semantic_v3 import TRAIN_CONDITIONS, evaluate_condition
from predify2021.mce_scores.diagnose_kitti_step_semantic_v3_fast import gradients, run
from predify2021.model_factory.deeplabv3plus_resnet50 import ErrorRegulatedSemanticRestorationPredictor


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="/home/lin/predify/kitti_step"); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True); parser.add_argument("--gradient-steps", type=int, default=8); args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _ = load_components(STATIC_CHECKPOINT_DEFAULT, ADAPTER_CHECKPOINT_DEFAULT, payload["source_dynamics_checkpoint"], WRITEBACK_CHECKPOINT_DEFAULT)
    predictor = ErrorRegulatedSemanticRestorationPredictor().cuda(); predictor.load_state_dict(payload["model_state_dict"], strict=True); predictor.freeze_dynamics(); model.requires_grad_(False); model.eval(); predictor.eval()
    groups = sequence_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val"))
    fast, trace = run(model, predictor, groups, 10**9)
    formal = {c: evaluate_condition(model, predictor, groups, c) for c in TRAIN_CONDITIONS}
    for c in TRAIN_CONDITIONS:
        fast[c]["blur_mIoU"] = formal[c]["blur_mIoU"]; fast[c]["restored_mIoU"] = formal[c]["restored_mIoU"]; fast[c]["clean_mIoU"] = formal[c]["clean_mIoU"]; fast[c]["mIoU_gain_vs_blur"] = formal[c]["restored_mIoU"] - formal[c]["blur_mIoU"]
    train_groups = sequence_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")); gradient = gradients(model, predictor, train_groups, args.gradient_steps)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    summary = {"experiment": "kitti_step_semantic_v3_formal", "diagnostic_only": True, "parameters_updated": False, "checkpoint": args.checkpoint, "split": "val", "sequence_count": len(groups), "protocol": {"conditions": {"Blur-Mid": "sigma=2.25", "Blur-Max": "sigma=3.0"}, "kernel": [BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE], "warmup": "existing 10%", "states_start": "frame 0", "writeback": "existing frozen residual writeback"}, "results": fast, "gradient_norms": gradient, "stage_a": {c: "GO" if fast[c]["continuous_feature_recovery"] > 0 and fast[c]["state_recovery"] > 0 and fast[c]["continuous"]["direction"] > 0 and fast[c]["continuous"]["amplitude"] >= 0.08 and fast[c]["semantic_state"]["growth_ratio"] < 3 and fast[c]["restored_mIoU"] >= fast[c]["blur_mIoU"] else "NO-GO" for c in TRAIN_CONDITIONS}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); (output / "gradient_norms.json").write_text(json.dumps(gradient, indent=2, sort_keys=True) + "\n")
    rows = [{"condition": c, "stage_a": summary["stage_a"][c], **{k: v for k, v in fast[c].items() if not isinstance(v, (dict, list))}, **{f"continuous_{k}": v for k, v in fast[c]["continuous"].items()}, **{f"no_history_{k}": v for k, v in fast[c]["no_history"].items()}, **{f"zero_error_{k}": v for k, v in fast[c]["zero_error"].items()}, **{f"semantic_{k}": v for k, v in fast[c]["semantic_state"].items()}} for c in TRAIN_CONDITIONS]
    with (output / "condition_summary.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output / "comparison.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=["condition", "variant", "mIoU", "gain_vs_blur", "feature_recovery"]); writer.writeheader(); writer.writerows([{"condition": c, "variant": v, "mIoU": fast[c]["blur_mIoU"] if v == "blur" else fast[c]["restored_mIoU"] if v == "restored" else fast[c]["clean_mIoU"], "gain_vs_blur": 0. if v == "blur" else fast[c]["restored_mIoU"] - fast[c]["blur_mIoU"] if v == "restored" else fast[c]["clean_mIoU"] - fast[c]["blur_mIoU"], "feature_recovery": 0. if v == "blur" else fast[c]["continuous_feature_recovery"] if v == "restored" else None} for c in TRAIN_CONDITIONS for v in ("blur", "restored", "clean")])
    with (output / "temporal_trace.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=list(trace[0])); writer.writeheader(); writer.writerows(trace)
    print(json.dumps({"stage_a": summary["stage_a"]}, indent=2), flush=True)


if __name__ == "__main__": main()
