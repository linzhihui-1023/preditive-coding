"""Locate the signal bottleneck in Semantic V2 without updating parameters.

The diagnostic follows the established KITTI-STEP Blur-Mid/Blur-Max protocol,
measures every stage from dynamics prediction error to the restored Z4 residual,
and performs one read-only backward pass on a short training chunk.
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
from predify2021.mce_scores.diagnose_kitti_step_error_guided_semantic_restoration import (
    CHECKPOINT_DEFAULT,
    slice_state,
)
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
)


CONDITIONS = ("Blur-Mid", "Blur-Max")
TOP_ENERGY_FRACTIONS = (0.05, 0.10, 0.20, 0.50)
EPSILON = 1e-12


def rms(value):
    return float(value.detach().float().square().mean().sqrt().item())


def flat_cosine(left, right):
    left = left.detach().float().flatten(1)
    right = right.detach().float().flatten(1)
    return float(F.cosine_similarity(left, right, dim=1, eps=EPSILON).mean().item())


def oracle_projection_scale(correction, oracle):
    correction = correction.detach().float().flatten(1)
    oracle = oracle.detach().float().flatten(1)
    numerator = (correction * oracle).sum(dim=1)
    denominator = oracle.square().sum(dim=1).clamp_min(EPSILON)
    return float((numerator / denominator).mean().item())


def residual_energy_fractions(residual):
    energy = residual.detach().float().square().flatten()
    total = float(energy.sum().item())
    if total <= EPSILON:
        return {f"top_{int(fraction * 100)}pct_energy_fraction": 0.0 for fraction in TOP_ENERGY_FRACTIONS}
    sorted_energy = energy.sort(descending=True).values
    cumulative = sorted_energy.cumsum(0)
    result = {}
    for fraction in TOP_ENERGY_FRACTIONS:
        count = max(1, math.ceil(energy.numel() * fraction))
        result[f"top_{int(fraction * 100)}pct_energy_fraction"] = float(
            (cumulative[count - 1] / total).item()
        )
    return result


def mean(values):
    return sum(values) / max(len(values), 1)


class ConditionAggregate:
    def __init__(self):
        self.frame_count = 0
        self.tensor_square_sum = defaultdict(float)
        self.tensor_element_count = defaultdict(int)
        self.scalar_values = defaultdict(list)
        self.observation_sse = 0.0
        self.prediction_sse = 0.0
        self.feature_elements = 0
        self.correction_oracle_dot = 0.0
        self.oracle_square_sum = 0.0

    def add_tensor(self, name, value):
        value = value.detach().float()
        self.tensor_square_sum[name] += float(value.square().sum().item())
        self.tensor_element_count[name] += value.numel()

    def add_scalar(self, name, value):
        self.scalar_values[name].append(float(value))

    def add_prediction(self, observation, prediction, clean):
        observation_error = observation.detach().float() - clean.detach().float()
        prediction_error = prediction.detach().float() - clean.detach().float()
        self.observation_sse += float(observation_error.square().sum().item())
        self.prediction_sse += float(prediction_error.square().sum().item())
        self.feature_elements += observation_error.numel()

    def add_projection(self, correction, oracle):
        correction = correction.detach().float()
        oracle = oracle.detach().float()
        self.correction_oracle_dot += float((correction * oracle).sum().item())
        self.oracle_square_sum += float(oracle.square().sum().item())

    def finish(self):
        signal_rms = {
            name: math.sqrt(
                self.tensor_square_sum[name]
                / max(self.tensor_element_count[name], 1)
            )
            for name in self.tensor_square_sum
        }
        observation_mse = self.observation_sse / max(self.feature_elements, 1)
        prediction_mse = self.prediction_sse / max(self.feature_elements, 1)
        scalar_means = {
            name: mean(values) for name, values in self.scalar_values.items()
        }
        return {
            "effective_frame_count": self.frame_count,
            "dynamics_prior": {
                "observation_mse_to_clean_z4": observation_mse,
                "prediction_mse_to_clean_z4": prediction_mse,
                "prediction_recovery_fraction": (
                    1.0 - prediction_mse / max(observation_mse, EPSILON)
                ),
                "cos_raw_prediction_error_oracle": scalar_means[
                    "cos_raw_prediction_error_oracle"
                ],
                "cos_negated_prediction_error_oracle": scalar_means[
                    "cos_negated_prediction_error_oracle"
                ],
            },
            "signal_rms": signal_rms,
            "signal_ratios": {
                "rho_encoder": signal_rms["encoded_error"]
                / max(signal_rms["prediction_error"], EPSILON),
                "rho_state": signal_rms["state_innovation"]
                / max(signal_rms["encoded_error"], EPSILON),
                "rho_restore": signal_rms["restoration_delta"]
                / max(signal_rms["oracle_residual"], EPSILON),
            },
            "update_gain": {
                name.removeprefix("update_gain_"): value
                for name, value in scalar_means.items()
                if name.startswith("update_gain_")
            },
            "direction_and_scale": {
                "correction_direction_cosine": scalar_means[
                    "correction_direction_cosine"
                ],
                "mean_frame_oracle_projection_scale": scalar_means[
                    "oracle_projection_scale"
                ],
                "global_oracle_projection_scale": self.correction_oracle_dot
                / max(self.oracle_square_sum, EPSILON),
            },
            "semantic_state": {
                "hidden_encoded_error_cosine": scalar_means[
                    "hidden_encoded_error_cosine"
                ],
                "state_projection_oracle_cosine": scalar_means[
                    "state_projection_oracle_cosine"
                ],
            },
            "residual_sparsity": {
                name: value
                for name, value in scalar_means.items()
                if name.endswith("energy_fraction")
            },
        }


def encode_clean_and_observation(model, sample, condition, frame_index, length):
    clean_image = load_image(sample)
    observed_image = diagnostic_blur(
        clean_image, frame_index, length, condition
    )
    with torch.no_grad():
        states = model.encode_backbone_features(
            model.extract_backbone_features(
                torch.cat((clean_image, observed_image), dim=0)
            )
        )
    return slice_state(states, 0), slice_state(states, 1)


def frame_metrics(
    condition,
    sequence,
    frame_index,
    observation,
    clean,
    prediction,
    semantic_hidden,
    diagnostics,
    predictor,
):
    prediction_error = diagnostics["prediction_error_z4"]
    encoded_error = diagnostics["error_innovation"]
    state_innovation = diagnostics["state_innovation"]
    restoration_delta = diagnostics["restoration_delta_z4"]
    oracle_residual = clean.z4 - observation.z4
    state_projection = torch.tanh(
        predictor.semantic_restoration_head.state_projection(semantic_hidden)
    )
    gain = diagnostics["update_gain"].detach().float()
    gain_flat = gain.flatten()

    row = {
        "condition": condition,
        "sequence": sequence,
        "frame_index": frame_index,
        "observation_mse_to_clean_z4": float(
            F.mse_loss(observation.z4.float(), clean.z4.float()).item()
        ),
        "prediction_mse_to_clean_z4": float(
            F.mse_loss(prediction.z4.float(), clean.z4.float()).item()
        ),
        "cos_raw_prediction_error_oracle": flat_cosine(
            prediction_error, oracle_residual
        ),
        "cos_negated_prediction_error_oracle": flat_cosine(
            -prediction_error, oracle_residual
        ),
        "prediction_error_rms": rms(prediction_error),
        "encoded_error_rms": rms(encoded_error),
        "update_gain_mean": float(gain.mean().item()),
        "update_gain_std": float(gain.std(unbiased=False).item()),
        "update_gain_p05": float(torch.quantile(gain_flat, 0.05).item()),
        "update_gain_p95": float(torch.quantile(gain_flat, 0.95).item()),
        "error_drive_rms": rms(diagnostics["error_drive"]),
        "context_modulation_rms": rms(diagnostics["context_modulation"]),
        "state_innovation_rms": rms(state_innovation),
        "semantic_hidden_rms": rms(semantic_hidden),
        "restoration_delta_rms": rms(restoration_delta),
        "oracle_residual_rms": rms(oracle_residual),
        "correction_direction_cosine": flat_cosine(
            restoration_delta, oracle_residual
        ),
        "oracle_projection_scale": oracle_projection_scale(
            restoration_delta, oracle_residual
        ),
        "hidden_encoded_error_cosine": flat_cosine(
            semantic_hidden, encoded_error
        ),
        "state_projection_oracle_cosine": flat_cosine(
            state_projection, oracle_residual
        ),
    }
    row["prediction_recovery_fraction"] = 1.0 - row[
        "prediction_mse_to_clean_z4"
    ] / max(row["observation_mse_to_clean_z4"], EPSILON)
    row["rho_encoder"] = row["encoded_error_rms"] / max(
        row["prediction_error_rms"], EPSILON
    )
    row["rho_state"] = row["state_innovation_rms"] / max(
        row["encoded_error_rms"], EPSILON
    )
    row["rho_restore"] = row["restoration_delta_rms"] / max(
        row["oracle_residual_rms"], EPSILON
    )
    row.update(residual_energy_fractions(oracle_residual))
    return row, {
        "prediction_error": prediction_error,
        "encoded_error": encoded_error,
        "error_drive": diagnostics["error_drive"],
        "context_modulation": diagnostics["context_modulation"],
        "state_innovation": state_innovation,
        "semantic_hidden": semantic_hidden,
        "restoration_delta": restoration_delta,
        "oracle_residual": oracle_residual,
    }


def evaluate_condition(
    model, predictor, groups, condition, max_effective_frames=0
):
    aggregate = ConditionAggregate()
    rows = []
    profile = defaultdict(lambda: defaultdict(list))

    predictor.eval()
    with torch.inference_mode():
        for sequence, samples in groups.items():
            if len(samples) < 2:
                continue
            onset = warmup_frame_count(len(samples))
            _, first_observation = encode_clean_and_observation(
                model, samples[0], condition, 0, len(samples)
            )
            pending_prediction, h4_dyn, h1_dyn = predictor.predict_next(
                first_observation,
                zero_state(first_observation),
                None,
                None,
            )
            semantic_hidden = predictor.initial_semantic_state()

            for frame_index in range(1, len(samples)):
                if max_effective_frames and aggregate.frame_count >= max_effective_frames:
                    break
                clean, observation = encode_clean_and_observation(
                    model,
                    samples[frame_index],
                    condition,
                    frame_index,
                    len(samples),
                )
                current_prediction = pending_prediction
                prediction_error = error_state(observation, current_prediction)
                _, semantic_hidden, diagnostics = predictor.restore_current(
                    observation, current_prediction, semantic_hidden
                )

                if frame_index >= onset:
                    row, tensors = frame_metrics(
                        condition,
                        sequence,
                        frame_index,
                        observation,
                        clean,
                        current_prediction,
                        semantic_hidden,
                        diagnostics,
                        predictor,
                    )
                    rows.append(row)
                    aggregate.frame_count += 1
                    aggregate.add_prediction(
                        observation.z4, current_prediction.z4, clean.z4
                    )
                    aggregate.add_projection(
                        tensors["restoration_delta"], tensors["oracle_residual"]
                    )
                    for name, tensor in tensors.items():
                        aggregate.add_tensor(name, tensor)
                    for name in (
                        "cos_raw_prediction_error_oracle",
                        "cos_negated_prediction_error_oracle",
                        "update_gain_mean",
                        "update_gain_std",
                        "update_gain_p05",
                        "update_gain_p95",
                        "correction_direction_cosine",
                        "oracle_projection_scale",
                        "hidden_encoded_error_cosine",
                        "state_projection_oracle_cosine",
                        "top_5pct_energy_fraction",
                        "top_10pct_energy_fraction",
                        "top_20pct_energy_fraction",
                        "top_50pct_energy_fraction",
                    ):
                        aggregate.add_scalar(name, row[name])
                    for name in (
                        "semantic_hidden_rms",
                        "state_innovation_rms",
                        "restoration_delta_rms",
                    ):
                        profile[frame_index][name].append(row[name])

                pending_prediction, h4_dyn, h1_dyn = predictor.predict_next(
                    observation,
                    prediction_error,
                    h4_dyn,
                    h1_dyn,
                )

            if max_effective_frames and aggregate.frame_count >= max_effective_frames:
                break

    profile_rows = []
    for frame_index in sorted(profile):
        values = profile[frame_index]
        profile_rows.append(
            {
                "condition": condition,
                "frame_index": frame_index,
                "sequence_count": len(values["semantic_hidden_rms"]),
                **{name: mean(items) for name, items in values.items()},
            }
        )
    return aggregate.finish(), rows, profile_rows


def module_gradient_norm(module):
    square_sum = 0.0
    parameter_count = 0
    parameters_with_gradient = 0
    for parameter in module.parameters():
        parameter_count += parameter.numel()
        if parameter.grad is not None:
            parameters_with_gradient += parameter.numel()
            square_sum += float(parameter.grad.detach().float().square().sum().item())
    return {
        "gradient_norm": math.sqrt(square_sum),
        "parameter_count": parameter_count,
        "parameters_with_gradient": parameters_with_gradient,
    }


def gradient_diagnostic(model, predictor, groups, steps):
    sequence, samples = next(iter(groups.items()))
    onset = warmup_frame_count(len(samples))
    if onset + steps >= len(samples):
        raise RuntimeError("Selected training sequence is too short for gradient chunk")

    predictor.train()
    predictor.zero_grad(set_to_none=True)
    _, observation = encode_clean_and_observation(
        model, samples[0], "Blur-Max", 0, len(samples)
    )
    with torch.no_grad():
        pending_prediction, h4_dyn, h1_dyn = predictor.predict_next(
            observation, zero_state(observation), None, None
        )
    semantic_hidden = predictor.initial_semantic_state()

    # Run the clean warm-up without constructing a training graph, then detach.
    for frame_index in range(1, onset):
        _, observation = encode_clean_and_observation(
            model, samples[frame_index], "Blur-Max", frame_index, len(samples)
        )
        prediction_error = error_state(observation, pending_prediction)
        with torch.no_grad():
            _, semantic_hidden, _ = predictor.restore_current(
                observation, pending_prediction, semantic_hidden
            )
            pending_prediction, h4_dyn, h1_dyn = predictor.predict_next(
                observation, prediction_error, h4_dyn, h1_dyn
            )

    losses = []
    for frame_index in range(onset, onset + steps):
        clean, observation = encode_clean_and_observation(
            model, samples[frame_index], "Blur-Max", frame_index, len(samples)
        )
        prediction_error = error_state(observation, pending_prediction)
        restored, semantic_hidden, _ = predictor.restore_current(
            observation, pending_prediction, semantic_hidden
        )
        losses.append(F.smooth_l1_loss(restored.z4, clean.z4.detach()))
        with torch.no_grad():
            pending_prediction, h4_dyn, h1_dyn = predictor.predict_next(
                observation, prediction_error, h4_dyn, h1_dyn
            )

    loss = torch.stack(losses).mean()
    loss.backward()
    modules = {
        "semantic_error_encoder": predictor.semantic_error_encoder,
        "semantic_recurrent.update_gain": predictor.semantic_recurrent.update_gain,
        "semantic_recurrent.error_drive": predictor.semantic_recurrent.error_drive,
        "semantic_recurrent.context_modulation": (
            predictor.semantic_recurrent.context_modulation
        ),
        "semantic_restoration_head.state_projection": (
            predictor.semantic_restoration_head.state_projection
        ),
        "semantic_restoration_head.observation_modulation": (
            predictor.semantic_restoration_head.observation_modulation
        ),
        "semantic_restoration_head.output": predictor.semantic_restoration_head.output,
    }
    result = {
        "split": "train",
        "condition": "Blur-Max",
        "sequence": sequence,
        "start_frame": onset,
        "step_count": steps,
        "loss": float(loss.detach().item()),
        "optimizer_step_performed": False,
        "modules": {name: module_gradient_norm(module) for name, module in modules.items()},
    }
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


def infer_findings(results):
    findings = {}
    for condition, result in results.items():
        prior = result["dynamics_prior"]
        ratios = result["signal_ratios"]
        direction = result["direction_and_scale"]
        state = result["semantic_state"]
        findings[condition] = {
            "dynamics_prior_has_recovery": prior["prediction_recovery_fraction"] > 0.0,
            "raw_error_has_directional_information": abs(
                prior["cos_raw_prediction_error_oracle"]
            ) >= 0.05,
            "encoder_severe_attenuation": ratios["rho_encoder"] < 0.1,
            "state_innovation_severe_attenuation": ratios["rho_state"] < 0.1,
            "correction_direction_learned": direction[
                "correction_direction_cosine"
            ] >= 0.2,
            "correction_amplitude_collapsed": (
                direction["correction_direction_cosine"] >= 0.2
                and abs(direction["global_oracle_projection_scale"]) < 0.1
            ),
            "state_more_aligned_with_error_than_oracle": abs(
                state["hidden_encoded_error_cosine"]
            ) > abs(state["state_projection_oracle_cosine"]),
        }
    return findings


def load_models(checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, _ = load_components(
        STATIC_CHECKPOINT_DEFAULT,
        ADAPTER_CHECKPOINT_DEFAULT,
        payload["source_dynamics_checkpoint"],
        WRITEBACK_CHECKPOINT_DEFAULT,
    )
    predictor = ErrorGuidedSemanticRestorationPredictor().cuda()
    predictor.load_state_dict(payload["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    predictor.freeze_dynamics()
    predictor.eval()
    return model, predictor, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument(
        "--output",
        default="results/kitti_step_semantic_innovation_bottleneck",
    )
    parser.add_argument("--sequence-limit", type=int, default=0)
    parser.add_argument("--max-effective-frames", type=int, default=0)
    parser.add_argument("--gradient-steps", type=int, default=8)
    parser.add_argument("--skip-gradient-check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Semantic innovation bottleneck diagnostic requires CUDA")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model, predictor, payload = load_models(checkpoint)

    val_groups = sequence_groups(
        KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    )
    sequence_limit = args.sequence_limit
    max_effective_frames = args.max_effective_frames
    output = Path(args.output)
    if args.smoke:
        sequence_limit = 1
        max_effective_frames = 8
        output = output / "smoke"
    if sequence_limit:
        val_groups = dict(list(val_groups.items())[:sequence_limit])
    output.mkdir(parents=True, exist_ok=True)

    results = {}
    all_frame_rows = []
    all_profile_rows = []
    for condition in CONDITIONS:
        result, frame_rows, profile_rows = evaluate_condition(
            model,
            predictor,
            val_groups,
            condition,
            max_effective_frames=max_effective_frames,
        )
        results[condition] = result
        all_frame_rows.extend(frame_rows)
        all_profile_rows.extend(profile_rows)
        print(json.dumps({"condition": condition, **result}, sort_keys=True), flush=True)

    gradient = None
    if not args.skip_gradient_check:
        train_groups = sequence_groups(
            KITTISTEPSegmentationDataset.from_kitti_step_root(
                Path(args.root), "train"
            )
        )
        gradient = gradient_diagnostic(
            model, predictor, train_groups, args.gradient_steps
        )
        print(json.dumps({"gradient_diagnostic": gradient}, sort_keys=True), flush=True)

    summary = {
        "experiment": "kitti_step_semantic_innovation_bottleneck",
        "diagnostic_only": True,
        "parameters_updated": False,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "split": "val",
        "sequence_count": len(val_groups),
        "protocol": {
            "conditions": {
                "Blur-Mid": "Gaussian blur sigma=2.25",
                "Blur-Max": "Gaussian blur sigma=3.0",
            },
            "kernel_size": [BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE],
            "warmup": "first floor(10% * sequence length) frames remain clean",
            "evaluation": "post-warmup frames only; recurrent states run from sequence start",
        },
        "cosine_sign_note": (
            "The requested raw error is e=Z4_blur-Z4_prediction while the oracle "
            "residual is R*=Z4_clean-Z4_blur. A useful clean prediction therefore "
            "makes cos(e,R*) negative. cos(-e,R*) is also reported as the "
            "restoration-aligned form."
        ),
        "results": results,
        "gradient_diagnostic": gradient,
        "decision_thresholds": {
            "near_zero_absolute_cosine": 0.05,
            "severe_signal_attenuation_ratio": 0.1,
            "learned_correction_direction_cosine": 0.2,
            "collapsed_absolute_projection_scale": 0.1,
        },
    }
    summary["findings"] = infer_findings(results)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_csv(output / "frame_metrics.csv", all_frame_rows)
    write_csv(output / "frame_index_profile.csv", all_profile_rows)
    print(json.dumps({"findings": summary["findings"]}, indent=2), flush=True)
    print(f"results={output}", flush=True)


if __name__ == "__main__":
    main()
