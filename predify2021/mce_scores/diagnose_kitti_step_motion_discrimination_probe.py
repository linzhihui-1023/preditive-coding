"""Linear-probe motion information in C4, P4, Z4 and Stage-T state.

For each Full9 adjacent-frame pair, the signed feature difference ΔF is used
as the input to a lightweight diagonal-ridge linear probe predicting RAFT
motion-vs-static labels.  Evaluation is leave-one-sequence-out, so adjacent
frames from one driving sequence never leak between fitting and testing.  The
probe is diagnostic only; no network weights are updated.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    load_components,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50 import AuxiliaryTemporalStateEncoder


FULL9 = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
FEATURES = ("C4", "P4", "Z4", "T")
FAST_B_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_semantic_v3_joint_c4_fast_ab_5754714/"
    "fast_b_joint_c4_weak_z4/best.pt"
)
STAGE_T_CHECKPOINT_DEFAULT = "/home/lin/predify/experiments/kitti_step_v2_auxiliary_stage_t/best.pt"
MOTION_THRESHOLD_PX = 1.0
MAX_SAMPLES_PER_CLASS_PER_SEQUENCE = 2000
RIDGE = 1e-2


def load_fast_b_model(path):
    model, _ = load_components(
        STATIC_CHECKPOINT_DEFAULT, ADAPTER_CHECKPOINT_DEFAULT,
        ROLE_PREDICTOR_CHECKPOINT_DEFAULT, WRITEBACK_CHECKPOINT_DEFAULT,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.output_adapters[3].load_state_dict(
        payload["c4_output_adapter_state_dict"], strict=True
    )
    model.host_conditioned_writebacks["3"].load_state_dict(
        payload["c4_writeback_state_dict"], strict=True
    )
    model.requires_grad_(False).eval()
    return model


def resize_mask(mask, size):
    return F.interpolate(mask.float()[None, None], size=size, mode="nearest")[0, 0].bool()


def auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64); labels = np.asarray(labels, dtype=np.int64)
    positive = labels == 1; n_pos = int(positive.sum()); n_neg = int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]; ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]: end += 1
        ranks[start:end] = (start + 1 + end) / 2.0; start = end
    return float((ranks[order][positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def sample_pair(features, labels, rng):
    labels = labels.cpu().numpy().astype(np.int64)
    values = features.permute(1, 2, 0).float().cpu().numpy()
    result_x = []; result_y = []
    for cls in (0, 1):
        indices = np.flatnonzero(labels == cls)
        if not len(indices): continue
        count = min(len(indices), MAX_SAMPLES_PER_CLASS_PER_SEQUENCE)
        chosen = rng.choice(indices, size=count, replace=False)
        result_x.append(values.reshape(-1, values.shape[-1])[chosen]); result_y.append(np.full(count, cls, dtype=np.int64))
    return (np.concatenate(result_x), np.concatenate(result_y)) if result_x else (None, None)


@torch.inference_mode()
def collect(model, encoder, groups, raft):
    output = {sequence: {name: {"x": [], "y": []} for name in FEATURES} for sequence in FULL9}
    rng = np.random.default_rng(0)
    for sequence in FULL9:
        previous_image = None; previous = None
        for sample in groups[sequence]:
            image = load_image(sample)
            raw = model.extract_backbone_features(image)
            adapter = model.multi_layer_adapter.input_adapters[3]
            p4 = adapter.projection(raw.c4); z4 = adapter.norm(p4); temporal = encoder(z4)
            current = {"C4": raw.c4[0], "P4": p4[0], "Z4": z4[0], "T": temporal[0]}
            if previous is not None:
                flow = raft.current_to_previous(image, previous_image)
                motion = flow.square().sum(dim=1).sqrt()[0] > MOTION_THRESHOLD_PX
                for name in FEATURES:
                    motion_feature = resize_mask(motion, current[name].shape[-2:])
                    x, y = sample_pair(current[name] - previous[name], motion_feature, rng)
                    if x is not None:
                        output[sequence][name]["x"].append(x); output[sequence][name]["y"].append(y)
            previous_image = image; previous = current
    return {
        sequence: {
            name: {"x": np.concatenate(values["x"]) if values["x"] else np.empty((0, 0)),
                   "y": np.concatenate(values["y"]) if values["y"] else np.empty(0, dtype=np.int64)}
            for name, values in per_feature.items()
        } for sequence, per_feature in output.items()
    }


def fit_diagonal_probe(train_x, train_y):
    # A diagonal-ridge linear probe: standardize each channel, then use the
    # closed-form ridge coefficient for each channel independently.  This is
    # intentionally small and deterministic for a diagnostic experiment.
    mean = train_x.mean(axis=0); scale = train_x.std(axis=0); scale[scale < 1e-6] = 1.0
    x = (train_x - mean) / scale
    target = (train_y.astype(np.float64) * 2.0 - 1.0)
    weight = (x * target[:, None]).mean(axis=0) / (1.0 + RIDGE)
    return mean, scale, weight


def evaluate(collected):
    folds = {name: [] for name in FEATURES}
    for held_out in FULL9:
        for name in FEATURES:
            train_x = np.concatenate([collected[s][name]["x"] for s in FULL9 if s != held_out])
            train_y = np.concatenate([collected[s][name]["y"] for s in FULL9 if s != held_out])
            test_x = collected[held_out][name]["x"]; test_y = collected[held_out][name]["y"]
            if not len(test_y): continue
            mean, scale, weight = fit_diagonal_probe(train_x, train_y)
            score = ((test_x - mean) / scale) @ weight
            folds[name].append({"held_out_sequence": held_out, "auroc": auc(score, test_y), "test_samples": int(len(test_y))})
    return {
        name: {"loso_auroc_mean": float(np.nanmean([row["auroc"] for row in rows])),
               "loso_auroc_std": float(np.nanstd([row["auroc"] for row in rows])),
               "folds": rows}
        for name, rows in folds.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--stage-t-checkpoint", default=STAGE_T_CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default="results/kitti_step_motion_discrimination_probe.json")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    model = load_fast_b_model(args.fast_b_checkpoint)
    payload = torch.load(args.stage_t_checkpoint, map_location="cpu", weights_only=False)
    encoder = AuxiliaryTemporalStateEncoder().cuda(); encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    encoder.requires_grad_(False).eval()
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    collected = collect(model, encoder, sequence_groups(dataset), FrozenRAFT())
    result = {"experiment": "C4/P4/Z4/T motion-discrimination linear probe", "full9": FULL9,
              "features": FEATURES, "motion_threshold_px": MOTION_THRESHOLD_PX,
              "max_samples_per_class_per_sequence": MAX_SAMPLES_PER_CLASS_PER_SEQUENCE,
              "ridge": RIDGE, "results": evaluate(collected),
              "sample_counts": {s: {name: int(len(collected[s][name]["y"])) for name in FEATURES} for s in FULL9}}
    output = Path(args.result_output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__": main()
