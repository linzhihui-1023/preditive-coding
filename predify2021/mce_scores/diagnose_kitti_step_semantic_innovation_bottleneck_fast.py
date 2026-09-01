"""Fast, decoder-free Semantic V2 innovation bottleneck diagnosis.

This is deliberately a separate entry point.  It evaluates three deterministic
KITTI-STEP validation sequences, batches clean/Blur-Mid/Blur-Max into one host
forward per frame, and writes only scalar traces.  No parameters are updated.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
    sequence_groups,
)
from predify2021.mce_scores.kitti_step_persistent_blur import (
    BLUR_KERNEL_SIZE,
    warmup_frame_count,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    error_state,
    load_components,
    zero_state,
)
from predify2021.mce_scores.train_kitti_step_convgru_decoupling_quick import (
    diagnostic_blur,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorGuidedSemanticRestorationPredictor,
    UnifiedFeatures,
)


CONDITIONS = ("Blur-Mid", "Blur-Max")
EPS = 1e-12
TOP_FRACTIONS = (0.05, 0.10, 0.20, 0.50)
CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_error_guided_semantic_restoration/"
    "best_error_guided_semantic_restoration.pt"
)


def slice_state(state, index):
    return UnifiedFeatures(*(value[index : index + 1] for value in state.as_tuple()))


def clone_state(state):
    return UnifiedFeatures(*(value.clone() for value in state.as_tuple()))


def rms(value):
    return float(value.detach().float().square().mean().sqrt().item())


def cosine(left, right):
    left = left.detach().float().flatten(1)
    right = right.detach().float().flatten(1)
    return float(F.cosine_similarity(left, right, dim=1, eps=EPS).mean().item())


def projection_scale(delta, oracle):
    delta = delta.detach().float().flatten(1)
    oracle = oracle.detach().float().flatten(1)
    return float(((delta * oracle).sum(1) / oracle.square().sum(1).clamp_min(EPS)).mean().item())


def amplitude_ratio(delta, oracle):
    return rms(delta) / max(rms(oracle), EPS)


def sampled_sparsity(oracle):
    energy = oracle.detach().float().square().flatten()
    total = energy.sum()
    if total.item() <= EPS:
        return {f"top_{int(f * 100)}pct_energy_fraction": 0.0 for f in TOP_FRACTIONS}
    cumulative = energy.sort(descending=True).values.cumsum(0)
    return {
        f"top_{int(f * 100)}pct_energy_fraction": float(
            (cumulative[max(1, math.ceil(energy.numel() * f)) - 1] / total).item()
        )
        for f in TOP_FRACTIONS
    }


def observed_batch(model, sample, frame, length):
    clean = load_image(sample)
    mid = diagnostic_blur(clean, frame, length, "Blur-Mid")
    maximum = diagnostic_blur(clean, frame, length, "Blur-Max")
    with torch.inference_mode():
        states = model.encode_backbone_features(
            model.extract_backbone_features(torch.cat((clean, mid, maximum), dim=0))
        )
    return tuple(slice_state(states, index) for index in range(3))


def empty_summary():
    return {
        "frames": 0,
        "obs_sse": 0.0,
        "pred_sse": 0.0,
        "feature_count": 0,
        "values": defaultdict(list),
        "tensor_sq": defaultdict(float),
        "tensor_count": defaultdict(int),
        "oracle_dot": 0.0,
        "oracle_sq": 0.0,
        "hidden_first": None,
        "hidden_last": None,
    }


def add_tensor(summary, name, value):
    value = value.detach().float()
    summary["tensor_sq"][name] += float(value.square().sum().item())
    summary["tensor_count"][name] += value.numel()


def add_variant(summary, prefix, observation, restored, clean):
    oracle = clean.z4 - observation.z4
    delta = restored.z4 - observation.z4
    obs_error = observation.z4.float() - clean.z4.float()
    restored_error = restored.z4.float() - clean.z4.float()
    summary.setdefault("variant_sse", defaultdict(float))[prefix] += float(restored_error.square().sum().item())
    values = summary["values"]
    values[f"{prefix}_direction_cosine"].append(cosine(delta, oracle))
    values[f"{prefix}_alpha"].append(projection_scale(delta, oracle))
    values[f"{prefix}_amplitude_ratio"].append(amplitude_ratio(delta, oracle))
    values[f"{prefix}_mse"].append(float(F.mse_loss(restored.z4, clean.z4).item()))


def measure_variant(summary, prefix, observation, restored, clean):
    oracle = clean.z4 - observation.z4
    delta = restored.z4 - observation.z4
    add_variant(summary, prefix, observation, restored, clean)
    add_tensor(summary, f"{prefix}_delta", delta)
    if prefix == "continuous":
        add_tensor(summary, "oracle_residual", oracle)


def run_condition(model, predictor, groups, condition, max_frames):
    summary = empty_summary()
    trace = []
    sampled = []
    for sequence, samples in groups.items():
        if len(samples) < 2:
            continue
        onset = warmup_frame_count(len(samples))
        clean0, mid0, max0 = observed_batch(model, samples[0], 0, len(samples))
        observations = {"Blur-Mid": mid0, "Blur-Max": max0}
        pending = {}
        for name, observation in observations.items():
            pending[name] = (*predictor.predict_next(observation, zero_state(observation), None, None), None)
        hidden = {name: predictor.initial_semantic_state() for name in CONDITIONS}
        count = 0
        for frame in range(1, len(samples)):
            if count >= max_frames:
                break
            clean, mid, maximum = observed_batch(model, samples[frame], frame, len(samples))
            observations = {"Blur-Mid": mid, "Blur-Max": maximum}
            if frame < onset:
                for name in CONDITIONS:
                    observation = observations[name]
                    prediction, h4, h1, _ = pending[name]
                    error = error_state(observation, prediction)
                    with torch.inference_mode():
                        _, hidden[name], _ = predictor.restore_current(observation, prediction, hidden[name])
                        pending[name] = (*predictor.predict_next(observation, error, h4, h1), None)
                continue
            row = {"sequence": sequence, "frame": frame, "condition": condition}
            # Run both conditions' independent states, while retaining one row for
            # the requested condition.  The other condition is updated as well so
            # its state is continuous when this function is called for it.
            for name in CONDITIONS:
                observation = observations[name]
                prediction, h4, h1, _ = pending[name]
                error = error_state(observation, prediction)
                with torch.inference_mode():
                    continuous, hidden[name], diagnostics = predictor.restore_current(
                        observation, prediction, hidden[name]
                    )
                    no_history, _, no_diag = predictor.restore_current(observation, prediction, None)
                selected = name == condition
                if selected:
                    oracle = clean.z4 - observation.z4
                    row.update(
                        {
                            "raw_error_rms": rms(diagnostics["prediction_error_z4"]),
                            "encoded_error_rms": rms(diagnostics["error_innovation"]),
                            "update_gain_mean": float(diagnostics["update_gain"].mean().item()),
                            "update_gain_std": float(diagnostics["update_gain"].std(unbiased=False).item()),
                            "state_innovation_rms": rms(diagnostics["state_innovation"]),
                            "hidden_rms": rms(hidden[name]),
                            "delta_z4_rms": rms(diagnostics["restoration_delta_z4"]),
                            "oracle_residual_rms": rms(oracle),
                            "direction_cosine": cosine(diagnostics["restoration_delta_z4"], oracle),
                            "alpha": projection_scale(diagnostics["restoration_delta_z4"], oracle),
                            "amplitude_ratio": amplitude_ratio(diagnostics["restoration_delta_z4"], oracle),
                            "hidden_error_cosine": cosine(hidden[name], diagnostics["error_innovation"]),
                            "state_projection_oracle_cosine": cosine(
                                torch.tanh(predictor.semantic_restoration_head.state_projection(hidden[name])), oracle
                            ),
                            "prediction_recovery_fraction": 1.0 - float(F.mse_loss(prediction.z4, clean.z4).item()) / max(float(F.mse_loss(observation.z4, clean.z4).item()), EPS),
                            "nohistory_direction_cosine": cosine(no_diag["restoration_delta_z4"], oracle),
                            "nohistory_alpha": projection_scale(no_diag["restoration_delta_z4"], oracle),
                            "nohistory_amplitude_ratio": amplitude_ratio(no_diag["restoration_delta_z4"], oracle),
                        }
                    )
                    summary["frames"] += 1
                    summary["hidden_first"] = summary["hidden_first"] or row["hidden_rms"]
                    summary["hidden_last"] = row["hidden_rms"]
                    add_variant(summary, "continuous", observation, continuous, clean)
                    add_variant(summary, "nohistory", observation, no_history, clean)
                    for name2, value in {
                        "raw_error": diagnostics["prediction_error_z4"],
                        "encoded_error": diagnostics["error_innovation"],
                        "state_innovation": diagnostics["state_innovation"],
                        "semantic_hidden": hidden[name],
                        "restoration_delta": diagnostics["restoration_delta_z4"],
                        "oracle_residual": oracle,
                    }.items():
                        add_tensor(summary, name2, value)
                    if (frame - onset) % 10 == 0:
                        sampled.append({**sampled_sparsity(oracle), "frame": frame, "sequence": sequence})
                    trace.append(row)
                with torch.inference_mode():
                    pending[name] = (*predictor.predict_next(observation, error, h4, h1), None)
                count += int(selected)
            if count >= max_frames:
                break
    # Counted values are frame means; RMS values are exact over all elements.
    element_count = max(summary["feature_count"], 1)
    observation_mse = summary["obs_sse"] / element_count
    prediction_mse = summary["pred_sse"] / element_count
    rms_values = {
        name: math.sqrt(summary["tensor_sq"][name] / max(summary["tensor_count"][name], 1))
        for name in summary["tensor_sq"]
    }
    values = summary["values"]
    means = {name: sum(items) / max(len(items), 1) for name, items in values.items()}
    result = {
        "condition": condition,
        "effective_frame_count": summary["frames"],
        "prediction_recovery": 1.0 - prediction_mse / max(observation_mse, EPS),
        "error_oracle_cosine": means.get("raw_error_oracle_cosine", None),
        "signal_rms": rms_values,
        "encoder_ratio": rms_values.get("encoded_error", 0.0) / max(rms_values.get("raw_error", 0.0), EPS),
        "state_ratio": rms_values.get("state_innovation", 0.0) / max(rms_values.get("encoded_error", 0.0), EPS),
        "restoration_amplitude": rms_values.get("restoration_delta", 0.0) / max(rms_values.get("oracle_residual", 0.0), EPS),
        "continuous_feature_recovery": 1.0 - means["continuous_mse"] / max(observation_mse, EPS),
        "no_history_feature_recovery": 1.0 - means["nohistory_mse"] / max(observation_mse, EPS),
        "temporal_feature_gain": means["nohistory_mse"] - means["continuous_mse"],
        "continuous": {key.removeprefix("continuous_"): value for key, value in means.items() if key.startswith("continuous_")},
        "no_history": {key.removeprefix("nohistory_"): value for key, value in means.items() if key.startswith("nohistory_")},
        "semantic_state": {
            "hidden_first_rms": summary["hidden_first"],
            "hidden_last_rms": summary["hidden_last"],
            "hidden_growth_ratio": summary["hidden_last"] / max(summary["hidden_first"], EPS),
            "hidden_encoded_error_cosine": means.get("hidden_error_cosine"),
            "state_projection_oracle_cosine": means.get("state_projection_oracle_cosine"),
        },
        "residual_sparsity_sample_count": len(sampled),
        "residual_sparsity_sample_mean": {
            key: sum(row[key] for row in sampled) / max(len(sampled), 1)
            for key in ("top_5pct_energy_fraction", "top_10pct_energy_fraction", "top_20pct_energy_fraction", "top_50pct_energy_fraction")
        },
    }
    return result, trace, sampled


def finalize_fast(summary, condition, sampled):
    count = max(summary["feature_count"], 1)
    obs_mse = summary["obs_sse"] / count
    values = summary["values"]
    means = {key: sum(items) / max(len(items), 1) for key, items in values.items()}
    rms_values = {key: math.sqrt(summary["tensor_sq"][key] / max(summary["tensor_count"][key], 1)) for key in summary["tensor_sq"]}
    continuous_mse = summary["variant_sse"]["continuous"] / count
    nohistory_mse = summary["variant_sse"]["nohistory"] / count
    return {
        "condition": condition, "effective_frame_count": summary["frames"], "observation_mse": obs_mse,
        "prediction_recovery": 1.0 - summary["pred_sse"] / count / max(obs_mse, EPS),
        "error_oracle_cosine": means.get("error_oracle_cosine"), "signal_rms": rms_values,
        "update_gain": {"mean": means.get("update_gain_mean"), "std": means.get("update_gain_std")},
        "encoder_ratio": rms_values.get("encoded_error", 0.0) / max(rms_values.get("raw_error", 0.0), EPS),
        "state_ratio": rms_values.get("state_innovation", 0.0) / max(rms_values.get("encoded_error", 0.0), EPS),
        "restoration_amplitude": rms_values.get("restoration_delta", 0.0) / max(rms_values.get("oracle_residual", 0.0), EPS),
        "continuous_feature_recovery": 1.0 - continuous_mse / max(obs_mse, EPS),
        "no_history_feature_recovery": 1.0 - nohistory_mse / max(obs_mse, EPS),
        "temporal_feature_gain": (nohistory_mse - continuous_mse) / max(obs_mse, EPS),
        "continuous": {key.removeprefix("continuous_"): value for key, value in means.items() if key.startswith("continuous_")},
        "no_history": {key.removeprefix("nohistory_"): value for key, value in means.items() if key.startswith("nohistory_")},
        "semantic_state": {"hidden_first_rms": summary["hidden_first"], "hidden_last_rms": summary["hidden_last"], "hidden_growth_ratio": summary["hidden_last"] / max(summary["hidden_first"], EPS), "hidden_encoded_error_cosine": means.get("hidden_error_cosine"), "state_projection_oracle_cosine": means.get("state_projection_oracle_cosine")},
        "residual_sparsity_sample_count": len(sampled),
        "residual_sparsity_sample_mean": {key: sum(row[key] for row in sampled) / max(len(sampled), 1) for key in ("top_5pct_energy_fraction", "top_10pct_energy_fraction", "top_20pct_energy_fraction", "top_50pct_energy_fraction")},
    }


def run_fast_shared(model, predictor, groups, max_frames):
    summaries = {condition: empty_summary() for condition in CONDITIONS}
    traces = []
    sampled = {condition: [] for condition in CONDITIONS}
    for sequence, samples in groups.items():
        if len(samples) < 2:
            continue
        onset = warmup_frame_count(len(samples))
        _, mid0, max0 = observed_batch(model, samples[0], 0, len(samples))
        observations = {"Blur-Mid": mid0, "Blur-Max": max0}
        pending = {name: (*predictor.predict_next(obs, zero_state(obs), None, None), None) for name, obs in observations.items()}
        hidden = {name: predictor.initial_semantic_state() for name in CONDITIONS}
        count = 0
        for frame in range(1, len(samples)):
            if count >= max_frames:
                break
            clean, mid, maximum = observed_batch(model, samples[frame], frame, len(samples))
            observations = {"Blur-Mid": mid, "Blur-Max": maximum}
            if frame < onset:
                for name in CONDITIONS:
                    obs = observations[name]; prediction, h4, h1, _ = pending[name]; error = error_state(obs, prediction)
                    with torch.inference_mode():
                        _, hidden[name], _ = predictor.restore_current(obs, prediction, hidden[name])
                        pending[name] = (*predictor.predict_next(obs, error, h4, h1), None)
                continue
            for name in CONDITIONS:
                obs = observations[name]; prediction, h4, h1, _ = pending[name]; error = error_state(obs, prediction)
                with torch.inference_mode():
                    continuous, hidden[name], diagnostics = predictor.restore_current(obs, prediction, hidden[name])
                    no_history, _, no_diag = predictor.restore_current(obs, prediction, None)
                oracle = clean.z4 - obs.z4; summary = summaries[name]
                summary["frames"] += 1
                obs_error = obs.z4.float() - clean.z4.float()
                summary["obs_sse"] += float(obs_error.square().sum().item()); summary["feature_count"] += obs_error.numel()
                summary["pred_sse"] += float((prediction.z4.float() - clean.z4.float()).square().sum().item())
                summary["values"]["error_oracle_cosine"].append(cosine(diagnostics["prediction_error_z4"], oracle))
                add_variant(summary, "continuous", obs, continuous, clean); add_variant(summary, "nohistory", obs, no_history, clean)
                for key, value in {"raw_error": diagnostics["prediction_error_z4"], "encoded_error": diagnostics["error_innovation"], "state_innovation": diagnostics["state_innovation"], "semantic_hidden": hidden[name], "restoration_delta": diagnostics["restoration_delta_z4"], "oracle_residual": oracle}.items():
                    add_tensor(summary, key, value)
                with torch.no_grad():
                    projection = torch.tanh(predictor.semantic_restoration_head.state_projection(hidden[name]))
                row = {"sequence": sequence, "condition": name, "frame": frame, "raw_error_rms": rms(diagnostics["prediction_error_z4"]), "encoded_error_rms": rms(diagnostics["error_innovation"]), "update_gain_mean": float(diagnostics["update_gain"].mean().item()), "update_gain_std": float(diagnostics["update_gain"].std(unbiased=False).item()), "state_innovation_rms": rms(diagnostics["state_innovation"]), "hidden_rms": rms(hidden[name]), "delta_z4_rms": rms(diagnostics["restoration_delta_z4"]), "oracle_residual_rms": rms(oracle), "direction_cosine": cosine(diagnostics["restoration_delta_z4"], oracle), "alpha": projection_scale(diagnostics["restoration_delta_z4"], oracle), "amplitude_ratio": amplitude_ratio(diagnostics["restoration_delta_z4"], oracle), "hidden_error_cosine": cosine(hidden[name], diagnostics["error_innovation"]), "state_projection_oracle_cosine": cosine(projection, oracle), "prediction_recovery_fraction": 1.0 - float(F.mse_loss(prediction.z4, clean.z4).item()) / max(float(F.mse_loss(obs.z4, clean.z4).item()), EPS), "nohistory_direction_cosine": cosine(no_diag["restoration_delta_z4"], oracle), "nohistory_alpha": projection_scale(no_diag["restoration_delta_z4"], oracle), "nohistory_amplitude_ratio": amplitude_ratio(no_diag["restoration_delta_z4"], oracle)}
                summary["values"]["update_gain_mean"].append(row["update_gain_mean"])
                summary["values"]["update_gain_std"].append(row["update_gain_std"])
                summary["hidden_first"] = summary["hidden_first"] or row["hidden_rms"]; summary["hidden_last"] = row["hidden_rms"]
                summary["values"]["continuous_direction_cosine"].append(row["direction_cosine"]); summary["values"]["continuous_alpha"].append(row["alpha"]); summary["values"]["continuous_amplitude_ratio"].append(row["amplitude_ratio"]); summary["values"]["continuous_mse"].append(float(F.mse_loss(continuous.z4, clean.z4).item())); summary["values"]["nohistory_direction_cosine"].append(row["nohistory_direction_cosine"]); summary["values"]["nohistory_alpha"].append(row["nohistory_alpha"]); summary["values"]["nohistory_amplitude_ratio"].append(row["nohistory_amplitude_ratio"]); summary["values"]["nohistory_mse"].append(float(F.mse_loss(no_history.z4, clean.z4).item())); summary["values"]["hidden_error_cosine"].append(row["hidden_error_cosine"]); summary["values"]["state_projection_oracle_cosine"].append(row["state_projection_oracle_cosine"])
                if (frame - onset) % 10 == 0:
                    sampled[name].append(sampled_sparsity(oracle))
                traces.append(row)
                with torch.inference_mode():
                    pending[name] = (*predictor.predict_next(obs, error, h4, h1), None)
            count += 1
    return {condition: finalize_fast(summaries[condition], condition, sampled[condition]) for condition in CONDITIONS}, traces


def grad_norm(module):
    total = sum(float(p.grad.detach().float().square().sum().item()) for p in module.parameters() if p.grad is not None)
    return math.sqrt(total)


def gradient_check(model, predictor, groups, steps):
    sequence, samples = next(iter(groups.items()))
    onset = warmup_frame_count(len(samples))
    if onset + steps >= len(samples):
        raise RuntimeError("training sequence is too short for requested gradient chunk")
    predictor.train()
    predictor.zero_grad(set_to_none=True)
    _, mid, maximum = (clone_state(value) for value in observed_batch(model, samples[0], 0, len(samples)))
    with torch.no_grad():
        pending = {}
        for name, observation in (("Blur-Mid", mid), ("Blur-Max", maximum)):
            pending[name] = (*predictor.predict_next(observation, zero_state(observation), None, None), None)
    hidden = {name: predictor.initial_semantic_state() for name in CONDITIONS}
    for frame in range(1, onset):
        _, mid, maximum = (clone_state(value) for value in observed_batch(model, samples[frame], frame, len(samples)))
        for name, observation in (("Blur-Mid", mid), ("Blur-Max", maximum)):
            prediction, h4, h1, _ = pending[name]
            prediction = clone_state(prediction)
            error = error_state(observation, prediction)
            with torch.no_grad():
                _, hidden[name], _ = predictor.restore_current(observation, prediction, hidden[name])
                pending[name] = (*predictor.predict_next(observation, error, h4, h1), None)
    hidden = {name: value.clone() for name, value in hidden.items()}
    losses = []
    for frame in range(onset, onset + steps):
        clean, mid, maximum = (clone_state(value) for value in observed_batch(model, samples[frame], frame, len(samples)))
        for name, observation in (("Blur-Mid", mid), ("Blur-Max", maximum)):
            prediction, h4, h1, _ = pending[name]
            prediction = clone_state(prediction)
            error = error_state(observation, prediction)
            restored, hidden[name], _ = predictor.restore_current(observation, prediction, hidden[name])
            losses.append(F.smooth_l1_loss(restored.z4, clean.z4.detach()))
            with torch.no_grad():
                pending[name] = (*predictor.predict_next(observation, error, h4, h1), None)
    loss = torch.stack(losses).mean()
    loss.backward()
    modules = {
        "semantic_error_encoder": predictor.semantic_error_encoder,
        "semantic_recurrent.update_gain": predictor.semantic_recurrent.update_gain,
        "semantic_recurrent.error_drive": predictor.semantic_recurrent.error_drive,
        "semantic_recurrent.context_modulation": predictor.semantic_recurrent.context_modulation,
        "semantic_restoration_head.state_projection": predictor.semantic_restoration_head.state_projection,
        "semantic_restoration_head.observation_modulation": predictor.semantic_restoration_head.observation_modulation,
        "semantic_restoration_head.output": predictor.semantic_restoration_head.output,
    }
    result = {"sequence": sequence, "start_frame": onset, "step_count": steps, "loss": float(loss.item()), "optimizer_step_performed": False, "gradient_norms": {name: grad_norm(module) for name, module in modules.items()}}
    predictor.zero_grad(set_to_none=True)
    predictor.eval()
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_csv_row(condition, result, bottleneck):
    row = {"condition": condition, "primary_bottleneck": bottleneck}
    row.update({key: value for key, value in result.items() if not isinstance(value, (dict, list))})
    row.update({f"continuous_{key}": value for key, value in result["continuous"].items()})
    row.update({f"no_history_{key}": value for key, value in result["no_history"].items()})
    row.update({f"semantic_{key}": value for key, value in result["semantic_state"].items()})
    row.update({f"residual_{key}": value for key, value in result["residual_sparsity_sample_mean"].items()})
    row.update({f"update_gain_{key}": value for key, value in result["update_gain"].items()})
    return row


def classify(result):
    if result["prediction_recovery"] <= 0 and abs(result["error_oracle_cosine"] or 0.0) < 0.1:
        return "PREDICTION_SOURCE"
    if result["encoder_ratio"] < 0.1:
        return "ERROR_ENCODER"
    if result["state_ratio"] < 0.1:
        return "RECURRENT_UPDATE"
    if result["continuous"]["direction_cosine"] > 0.3 and abs(result["continuous"]["alpha"]) < 0.1:
        return "RESTORATION_HEAD"
    # A moderate positive direction with <10% amplitude is still an output
    # collapse signal; the strict >0.3 rule above remains the paper-facing
    # diagnostic threshold.
    if result["continuous"]["direction_cosine"] >= 0.1 and result["restoration_amplitude"] < 0.1:
        return "RESTORATION_HEAD"
    if result["temporal_feature_gain"] <= 0:
        return "TEMPORAL_MEMORY"
    if result["residual_sparsity_sample_mean"]["top_10pct_energy_fraction"] > 0.8:
        return "TRAINING_TARGET"
    return "UNRESOLVED"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default="results/kitti_step_semantic_innovation_bottleneck_fast")
    parser.add_argument("--max-frames-per-sequence", type=int, default=250)
    parser.add_argument("--gradient-steps", type=int, default=8)
    parser.add_argument("--skip-gradient-check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, _ = load_components(STATIC_CHECKPOINT_DEFAULT, ADAPTER_CHECKPOINT_DEFAULT, payload["source_dynamics_checkpoint"], WRITEBACK_CHECKPOINT_DEFAULT)
    predictor = ErrorGuidedSemanticRestorationPredictor().cuda()
    predictor.load_state_dict(payload["model_state_dict"], strict=True)
    model.requires_grad_(False); model.eval(); predictor.freeze_dynamics(); predictor.eval()
    groups = sequence_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val"))
    ids = sorted(groups)
    selected = [ids[0], ids[len(ids) // 2], ids[-1]]
    if args.smoke:
        selected = selected[:1]
    groups = {key: groups[key] for key in selected}
    max_frames = 8 if args.smoke else args.max_frames_per_sequence
    output = Path(args.output) / "smoke" if args.smoke else Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results, trace = run_fast_shared(model, predictor, groups, max_frames)
    for condition, result in results.items():
        print(json.dumps({"condition": condition, **result}, sort_keys=True), flush=True)
    gradients = None
    if not args.skip_gradient_check:
        train = sequence_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train"))
        gradients = gradient_check(model, predictor, train, args.gradient_steps)
        (output / "gradient_norms.json").write_text(json.dumps(gradients, indent=2, sort_keys=True) + "\n")
    summary = {
        "experiment": "kitti_step_semantic_innovation_bottleneck_fast",
        "diagnostic_only": True, "parameters_updated": False, "checkpoint": str(checkpoint),
        "split": "val", "selected_sequences": selected, "max_frames_per_sequence": max_frames,
        "protocol": {"batch": ["clean", "Blur-Mid", "Blur-Max"], "sigma": {"Blur-Mid": 2.25, "Blur-Max": 3.0}, "kernel": [BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE], "warmup": "existing 10%", "evaluation": "post-warmup only; states start at frame 0"},
        "cosine_sign_note": "e=blur-prediction and R*=clean-blur; useful prediction therefore yields a negative raw cos(e,R*).",
        "results": results, "gradient_norms": gradients,
        "primary_bottleneck": {condition: classify(result) for condition, result in results.items()},
        "thresholds": {"prediction_cosine_weak": 0.1, "encoder_or_state_ratio": 0.1, "direction_learned": 0.3, "amplitude_collapsed": 0.1, "top10_sparse": 0.8},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(output / "condition_summary.csv", [condition_csv_row(condition, result, summary["primary_bottleneck"][condition]) for condition, result in results.items()])
    write_csv(output / "temporal_trace.csv", trace)
    print(json.dumps({"primary_bottleneck": summary["primary_bottleneck"]}, indent=2), flush=True)
    print(f"results={output}", flush=True)


if __name__ == "__main__":
    main()
