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
from predify2021.mce_scores.train_kitti_step_semantic_v3 import FAST_VALIDATION_SEQUENCES, TRAIN_CONDITIONS, evaluate_condition
from predify2021.mce_scores.diagnose_kitti_step_semantic_v3_fast import gradients, internal_diagnosis_row, run, write_csv
from predify2021.model_factory.deeplabv3plus_resnet50 import ErrorRegulatedSemanticRestorationPredictor

MAX_FAST_FRAMES_PER_SEQUENCE = 250


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="/home/lin/predify/kitti_step"); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True); parser.add_argument("--gradient-steps", type=int, default=8); args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _ = load_components(STATIC_CHECKPOINT_DEFAULT, ADAPTER_CHECKPOINT_DEFAULT, payload["source_dynamics_checkpoint"], WRITEBACK_CHECKPOINT_DEFAULT)
    predictor = ErrorRegulatedSemanticRestorationPredictor(use_error_temporal_stats=payload.get("use_error_temporal_stats", False)).cuda(); predictor.load_state_dict(payload["model_state_dict"], strict=True); predictor.freeze_dynamics(); model.requires_grad_(False); model.eval(); predictor.eval()
    groups_all = sequence_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")); missing = [sequence for sequence in FAST_VALIDATION_SEQUENCES if sequence not in groups_all]
    if missing: raise RuntimeError(f"Missing fixed fast-validation sequences: {missing}")
    groups = {sequence: groups_all[sequence] for sequence in FAST_VALIDATION_SEQUENCES}
    internal_results, trace = run(model, predictor, groups, MAX_FAST_FRAMES_PER_SEQUENCE)
    formal = {c: evaluate_condition(model, predictor, groups, c, max_frames=MAX_FAST_FRAMES_PER_SEQUENCE) for c in TRAIN_CONDITIONS}
    validation_results = {
        c: {
            "effective_frame_count": formal[c]["effective_frame_count"],
            "blur_mIoU": formal[c]["blur_mIoU"],
            "v3_mIoU": formal[c]["restored_mIoU"],
            "clean_mIoU": formal[c]["clean_mIoU"],
            "v3_minus_host": formal[c]["restored_mIoU"] - formal[c]["blur_mIoU"],
            "feature_recovery": formal[c]["feature_recovery"],
            "clean_stability_delta": formal[c]["restored_mIoU"] - formal[c]["clean_mIoU"],
        }
        for c in TRAIN_CONDITIONS
    }
    train_groups = sequence_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")); gradient = gradients(model, predictor, train_groups, args.gradient_steps)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    internal_stage_a = {c: "GO" if internal_results[c]["continuous_feature_recovery"] > 0 and internal_results[c]["semantic_state"]["state_recovery"] > 0 and internal_results[c]["continuous"]["direction"] > 0 and internal_results[c]["continuous"]["amplitude"] >= 0.08 and internal_results[c]["semantic_state"]["growth_ratio"] < 3 else "NO-GO" for c in TRAIN_CONDITIONS}
    validation_stage_a = {c: "GO" if validation_results[c]["v3_mIoU"] >= validation_results[c]["blur_mIoU"] else "NO-GO" for c in TRAIN_CONDITIONS}
    summary = {"experiment": "kitti_step_semantic_v3_formal", "diagnostic_only": True, "parameters_updated": False, "checkpoint": args.checkpoint, "split": "val", "sequence_count": len(groups), "protocol": {"conditions": {"Blur-Mid": "sigma=2.25", "Blur-Max": "sigma=3.0"}, "kernel": [BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE], "warmup": "existing 10%", "states_start": "frame 0", "max_post_warmup_frames_per_sequence": MAX_FAST_FRAMES_PER_SEQUENCE, "writeback": "existing frozen residual writeback"}, "internal_diagnosis": {"conditions": internal_results, "gradient_probe": gradient, "stage_a": internal_stage_a}, "fast_validation": {"conditions": validation_results, "stage_a": validation_stage_a}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); (output / "gradient_norms.json").write_text(json.dumps(gradient, indent=2, sort_keys=True) + "\n")
    write_csv(output / "internal_diagnosis.csv", [internal_diagnosis_row(c, internal_results[c], internal_stage_a[c]) for c in TRAIN_CONDITIONS])
    validation_rows = [{"condition": c, "stage_a": validation_stage_a[c], **validation_results[c]} for c in TRAIN_CONDITIONS]
    write_csv(output / "fast_validation.csv", validation_rows)
    with (output / "fast_validation_comparison.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=["condition", "variant", "mIoU", "gain_vs_blur", "feature_recovery"]); writer.writeheader(); writer.writerows([{"condition": c, "variant": v, "mIoU": validation_results[c]["blur_mIoU"] if v == "blur" else validation_results[c]["v3_mIoU"] if v == "restored" else validation_results[c]["clean_mIoU"], "gain_vs_blur": 0. if v == "blur" else validation_results[c]["v3_minus_host"] if v == "restored" else validation_results[c]["clean_mIoU"] - validation_results[c]["blur_mIoU"], "feature_recovery": 0. if v == "blur" else validation_results[c]["feature_recovery"] if v == "restored" else None} for c in TRAIN_CONDITIONS for v in ("blur", "restored", "clean")])
    with (output / "temporal_trace.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=list(trace[0])); writer.writeheader(); writer.writerows(trace)
    print(json.dumps({"internal_diagnosis_stage_a": internal_stage_a, "fast_validation_stage_a": validation_stage_a}, indent=2), flush=True)


if __name__ == "__main__": main()
