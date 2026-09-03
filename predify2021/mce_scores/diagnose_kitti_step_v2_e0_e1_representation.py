"""Full9 inference-only E0/E1 representation dynamics diagnosis for V2-Minimal."""

import argparse
import copy
import json
import math
import random
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.role_separated_dynamic_error_correction import ROLE_PREDICTOR_CHECKPOINT_DEFAULT
from predify2021.mce_scores.train_kitti_step_z4_only import FAST_B_DEFAULT
from predify2021.mce_scores.train_kitti_step_predictive_semantic_v2 import encode_host, load_v2

SEED = 0
FULL9 = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
E1_DEFAULT = "/home/lin/predify/experiments/kitti_step_predictive_semantic_v2/epoch_001.pt"
RESULT_DEFAULT = "results/kitti_step_v2_e0_e1_representation"


def reset_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def empty_stats():
    return {
        "frames": 0, "pairs": 0, "state_rms_sum": 0.0, "state_std_sum": 0.0,
        "copy_mse_sum": 0.0, "pred_mse_sum": 0.0, "pred_delta_mse_sum": 0.0,
    }


def finalize(stats):
    frames = max(stats["frames"], 1)
    pairs = max(stats["pairs"], 1)
    state_rms = stats["state_rms_sum"] / frames
    state_std = stats["state_std_sum"] / frames
    copy_mse = stats["copy_mse_sum"] / pairs
    pred_mse = stats["pred_mse_sum"] / pairs
    pred_delta_mse = stats["pred_delta_mse_sum"] / pairs
    true_delta_rms = math.sqrt(max(copy_mse, 0.0))
    pred_delta_rms = math.sqrt(max(pred_delta_mse, 0.0))
    error_rms = math.sqrt(max(pred_mse, 0.0))
    return {
        "frames": stats["frames"], "pairs": stats["pairs"],
        "state_rms": state_rms, "state_std": state_std,
        "copy_mse": copy_mse, "pred_mse": pred_mse,
        "Rpred": pred_mse / max(copy_mse, 1e-12),
        "true_delta_rms": true_delta_rms, "pred_delta_rms": pred_delta_rms,
        "error_rms": error_rms,
        "temporal_to_state_ratio": true_delta_rms / max(state_rms, 1e-12),
        "pred_motion_ratio": pred_delta_rms / max(true_delta_rms, 1e-12),
        "error_to_true_motion_ratio": error_rms / max(true_delta_rms, 1e-12),
    }


@torch.inference_mode()
def evaluate_pair(host, plugin_e0, plugin_e1, groups):
    plugins = {"E0": plugin_e0, "E1": plugin_e1}
    global_stats = {name: empty_stats() for name in plugins}
    per_sequence = {}
    for sequence in FULL9:
        samples = groups[sequence]
        seq_stats = {name: empty_stats() for name in plugins}
        previous_state, pending, hidden = {}, {}, {}
        for frame_index, sample in enumerate(samples):
            _, observation, _, _ = encode_host(host, sample)
            current_state = {name: plugin.encode(observation.z4) for name, plugin in plugins.items()}
            for name, state in current_state.items():
                for target in (global_stats[name], seq_stats[name]):
                    target["frames"] += 1
                    target["state_rms_sum"] += float(state.square().mean().sqrt().item())
                    target["state_std_sum"] += float(state.std().item())
            if frame_index == 0:
                for name, plugin in plugins.items():
                    pending[name], hidden[name] = plugin.predict_next(
                        current_state[name], torch.zeros_like(current_state[name]), None
                    )
                    previous_state[name] = current_state[name]
                continue
            for name, plugin in plugins.items():
                state, prev, pred = current_state[name], previous_state[name], pending[name]
                copy_mse = torch.mean((state - prev) ** 2).item()
                pred_mse = torch.mean((state - pred) ** 2).item()
                pred_delta_mse = torch.mean((pred - prev) ** 2).item()
                for target in (global_stats[name], seq_stats[name]):
                    target["pairs"] += 1
                    target["copy_mse_sum"] += copy_mse
                    target["pred_mse_sum"] += pred_mse
                    target["pred_delta_mse_sum"] += pred_delta_mse
                error = state - pred
                pending[name], hidden[name] = plugin.predict_next(state, error, hidden[name])
                previous_state[name] = state
        per_sequence[sequence] = {name: finalize(seq_stats[name]) for name in plugins}
    return {name: finalize(global_stats[name]) for name in plugins} | {"per_sequence": per_sequence}


def ratio(a, b):
    return a / max(b, 1e-12)


def classify(e0, e1):
    copy_ratio = ratio(e1["copy_mse"], e0["copy_mse"])
    rms_ratio = ratio(e1["state_rms"], e0["state_rms"])
    std_ratio = ratio(e1["state_std"], e0["state_std"])
    if copy_ratio < 0.5:
        temporal = "STRONG_TEMPORAL_FLATTENING"
    elif copy_ratio < 0.8:
        temporal = "MODERATE_TEMPORAL_FLATTENING"
    else:
        temporal = "NO_STRONG_TRAINING_INDUCED_FLATTENING"
    amplitude = "REPRESENTATION_AMPLITUDE_COLLAPSE" if rms_ratio < 0.5 or std_ratio < 0.5 else "NO_STRONG_AMPLITUDE_COLLAPSE"
    if e0["pred_motion_ratio"] > 10.0:
        predictor = "SEVERE_RANDOM_PREDICTOR_SCALE_MISMATCH"
    elif e0["pred_motion_ratio"] > 3.0:
        predictor = "PREDICTOR_SCALE_MISMATCH"
    else:
        predictor = "PREDICTOR_INITIAL_SCALE_REASONABLE"
    return {
        "copy_mse_E1_over_E0": copy_ratio,
        "state_rms_E1_over_E0": rms_ratio,
        "state_std_E1_over_E0": std_ratio,
        "temporal_representation_judgement": temporal,
        "amplitude_judgement": amplitude,
        "predictor_initialization_judgement": predictor,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--e1-checkpoint", default=E1_DEFAULT)
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_DEFAULT)
    parser.add_argument("--dynamics-checkpoint", default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    reset_seed()
    host, plugin_e0, _ = load_v2(args)
    plugin_e0.eval()
    plugin_e1 = copy.deepcopy(plugin_e0)
    payload = torch.load(args.e1_checkpoint, map_location="cpu", weights_only=False)
    plugin_e1.load_state_dict(payload["model_state_dict"], strict=True)
    plugin_e1.eval()
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    groups = sequence_groups(dataset)
    groups = {sequence: groups[sequence] for sequence in FULL9}
    metrics = evaluate_pair(host, plugin_e0, plugin_e1, groups)
    judgement = classify(metrics["E0"], metrics["E1"])
    result = {"experiment": "V2 E0 vs E1 Representation Dynamics Diagnosis", "inference_only": True, "seed": SEED, "e1_checkpoint": args.e1_checkpoint, "metrics": metrics, "judgement": judgement}
    output = Path(args.result_output); output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"; path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"E0": metrics["E0"], "E1": metrics["E1"], "judgement": judgement, "result": str(path)}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

