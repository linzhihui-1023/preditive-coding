import json
import os
import random
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_adaptive_dynamic_correction import detach_state
from predify2021.mce_scores.evaluate_kitti_step_closed_loop_dynamic_correction import add_frame_noise
from predify2021.mce_scores.evaluate_kitti_step_context_residual_ablation import split_state
from predify2021.mce_scores.evaluate_kitti_step_context_residual_correction import (
    LEGACY_CORRECTION_CHECKPOINT_DEFAULT,
    SIGMA,
    context_outputs,
    legacy_gains,
    load_components,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
    predict_current,
    sequence_groups,
    update_dynamic_error,
)
from predify2021.mce_scores.evaluate_kitti_step_error_correction import corrected_host_feature
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import BackboneFeatures, UnifiedFeatures


SEED = 0
EXPECTED_MIOU = 0.3157625378
EXPECTED_SEQUENCES = 9
EXPECTED_TOTAL_FRAMES = 2981
EXPECTED_EVALUATED_FRAMES = 2963
GAIN_EPSILON = 1e-6
CONTEXT_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_context_residual_ablation_working_tree/best_gain_dynamic_instant.pt"
)


def split_backbone_features(features, index):
    return BackboneFeatures(*(value[index:index + 1] for value in features.as_tuple()))


def encode_frame(model, image):
    raw_features = model.extract_backbone_features(image)
    state = model.encode_backbone_features(raw_features)
    return state, raw_features


def solve_oracle_gain(predicted, observation, clean):
    gains = []
    for predicted_value, observed_value, clean_value in (
        (predicted.z1, observation.z1, clean.z1),
        (predicted.z4, observation.z4, clean.z4),
    ):
        error = observed_value - predicted_value
        target_delta = clean_value - predicted_value
        denominator = error.square()
        numerator = target_delta * error
        gain = torch.where(
            denominator > 0,
            numerator / denominator,
            torch.zeros_like(denominator),
        )
        gains.append(gain.clamp(0.0, 1.0))
    return tuple(gains)


def oracle_posterior(predicted, observation, gains):
    return UnifiedFeatures(
        predicted.z1 + gains[0] * (observation.z1 - predicted.z1),
        observation.z2,
        observation.z3,
        predicted.z4 + gains[1] * (observation.z4 - predicted.z4),
    )


def update_confusion(confusion, model, host_feature, mask):
    logits = model.decode_from_host_feature(host_feature)
    prediction = logits.argmax(dim=1).squeeze(0).cpu().to(torch.int64)
    update_confusion_matrix(confusion, prediction, mask)
    return logits


def add_metrics(confusion):
    iou = compute_iou(confusion)
    return float(torch.nanmean(iou).item())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Oracle upper-bound diagnostic requires CUDA.")
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_ORACLE_OUTPUT_DIR",
            "results/kitti_step_oracle_upper_bound",
        )
    )
    context_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_CONTEXT_CHECKPOINT", CONTEXT_CHECKPOINT_DEFAULT)
    )

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model, predictor, legacy, context, checkpoints = load_components(root)
    context_payload = torch.load(context_checkpoint, map_location="cpu", weights_only=False)
    context.load_state_dict(context_payload["context_state_dict"], strict=True)
    model.eval()
    predictor.eval()
    legacy.eval()
    context.eval()
    modules = (model, predictor, legacy, context)
    parameter_snapshot = {
        name: parameter.detach().cpu().clone()
        for module in modules
        for name, parameter in module.named_parameters()
    }

    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    sequence_count = len(groups)
    total_frame_count = len(dataset.samples)
    if (
        sequence_count != EXPECTED_SEQUENCES
        or total_frame_count != EXPECTED_TOTAL_FRAMES
    ):
        raise RuntimeError(
            "Dataset protocol mismatch: expected 9 sequences and 2981 total frames, "
            f"got {sequence_count} sequences and {total_frame_count} frames."
        )

    confusion = {
        name: torch.zeros((19, 19), dtype=torch.int64)
        for name in (
            "clean_static",
            "noisy_static",
            "learned_context",
            "clean_state_injection",
            "oracle_clipped_gain_closed_loop",
        )
    }
    state_sums = {
        name: {"prediction": 0.0, "observation": 0.0, "posterior": 0.0}
        for name in ("z1", "z4")
    }
    gain_stats = {
        "min": float("inf"),
        "max": float("-inf"),
        "sum": 0.0,
        "count": 0,
        "near_zero_count": 0,
        "near_one_count": 0,
    }
    finite = True
    gains_in_range = True
    injection_exact = True
    evaluated_frame_count = 0
    learned_previous_previous = learned_previous = None
    oracle_previous_previous = oracle_previous = None
    learned_dynamic_error = oracle_dynamic_error = None

    with torch.inference_mode():
        for samples in groups.values():
            learned_previous_previous = learned_previous = None
            oracle_previous_previous = oracle_previous = None
            learned_dynamic_error = oracle_dynamic_error = None
            for sample in samples:
                clean_image = load_image(sample)
                noisy_image = add_frame_noise(clean_image, SIGMA)
                clean_state, _ = encode_frame(model, clean_image)
                observation, noisy_features = encode_frame(model, noisy_image)
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])

                clean_logits = model(clean_image)
                noisy_logits = model(noisy_image)
                update_confusion_matrix(
                    confusion["clean_static"],
                    clean_logits.argmax(dim=1).squeeze(0).cpu().to(torch.int64),
                    mask,
                )
                update_confusion_matrix(
                    confusion["noisy_static"],
                    noisy_logits.argmax(dim=1).squeeze(0).cpu().to(torch.int64),
                    mask,
                )

                if learned_previous is None:
                    learned_previous = observation
                    oracle_previous = observation
                    continue
                if learned_previous_previous is None:
                    learned_previous_previous = learned_previous
                    learned_previous = observation
                    oracle_previous_previous = oracle_previous
                    oracle_previous = observation
                    continue

                learned_predicted, learned_instant = predict_current(
                    predictor,
                    learned_previous_previous,
                    learned_previous,
                    observation,
                )
                learned_dynamic_error = update_dynamic_error(
                    learned_instant, learned_dynamic_error
                )
                learned_gain = legacy_gains(learned_dynamic_error, legacy)
                learned_posterior, _, _ = context_outputs(
                    learned_predicted,
                    observation,
                    learned_instant,
                    learned_dynamic_error,
                    learned_gain,
                    context,
                )

                oracle_predicted, oracle_instant = predict_current(
                    predictor,
                    oracle_previous_previous,
                    oracle_previous,
                    observation,
                )
                oracle_dynamic_error = update_dynamic_error(
                    oracle_instant, oracle_dynamic_error
                )
                oracle_legacy_gain = legacy_gains(oracle_dynamic_error, legacy)
                oracle_gains = solve_oracle_gain(oracle_predicted, observation, clean_state)
                oracle_posterior = oracle_posterior(
                    oracle_predicted, observation, oracle_gains
                )

                output_size = tuple(clean_image.shape[-2:])
                learned_host = corrected_host_feature(
                    model,
                    noisy_features,
                    observation,
                    learned_posterior,
                    output_size,
                )
                injection_target = UnifiedFeatures(
                    clean_state.z1.clone(),
                    observation.z2,
                    observation.z3,
                    clean_state.z4.clone(),
                )
                injection_exact = injection_exact and torch.equal(
                    injection_target.z1, clean_state.z1
                ) and torch.equal(injection_target.z4, clean_state.z4)
                injection_host = corrected_host_feature(
                    model,
                    noisy_features,
                    observation,
                    injection_target,
                    output_size,
                )
                oracle_host = corrected_host_feature(
                    model,
                    noisy_features,
                    observation,
                    oracle_posterior,
                    output_size,
                )
                update_confusion(
                    confusion["learned_context"], model, learned_host, mask
                )
                update_confusion(
                    confusion["clean_state_injection"], model, injection_host, mask
                )
                oracle_logits = update_confusion(
                    confusion["oracle_clipped_gain_closed_loop"],
                    model,
                    oracle_host,
                    mask,
                )

                for name, predicted_value, observed_value, posterior_value, clean_value in (
                    ("z1", oracle_predicted.z1, observation.z1, oracle_posterior.z1, clean_state.z1),
                    ("z4", oracle_predicted.z4, observation.z4, oracle_posterior.z4, clean_state.z4),
                ):
                    state_sums[name]["prediction"] += (
                        torch.mean((predicted_value - clean_value).square()).item()
                    )
                    state_sums[name]["observation"] += (
                        torch.mean((observed_value - clean_value).square()).item()
                    )
                    state_sums[name]["posterior"] += (
                        torch.mean((posterior_value - clean_value).square()).item()
                    )

                for gain in oracle_gains:
                    values = gain.flatten()
                    gain_stats["min"] = min(gain_stats["min"], values.min().item())
                    gain_stats["max"] = max(gain_stats["max"], values.max().item())
                    gain_stats["sum"] += values.sum().item()
                    gain_stats["count"] += values.numel()
                    gain_stats["near_zero_count"] += int((values <= GAIN_EPSILON).sum().item())
                    gain_stats["near_one_count"] += int(
                        (values >= 1.0 - GAIN_EPSILON).sum().item()
                    )
                finite = finite and all(
                    torch.isfinite(value).all().item()
                    for value in (
                        clean_state.z1,
                        clean_state.z4,
                        observation.z1,
                        observation.z4,
                        learned_posterior.z1,
                        learned_posterior.z4,
                        oracle_predicted.z1,
                        oracle_predicted.z4,
                        oracle_posterior.z1,
                        oracle_posterior.z4,
                        oracle_gains[0],
                        oracle_gains[1],
                        oracle_logits,
                    )
                )
                gains_in_range = gains_in_range and all(
                    bool(((gain >= 0.0) & (gain <= 1.0)).all())
                    for gain in oracle_gains
                )
                evaluated_frame_count += 1
                learned_dynamic_error = detach_state(learned_dynamic_error)
                oracle_dynamic_error = detach_state(oracle_dynamic_error)
                learned_previous_previous = learned_previous
                learned_previous = detach_state(learned_posterior)
                oracle_previous_previous = oracle_previous
                oracle_previous = detach_state(oracle_posterior)

    if evaluated_frame_count != EXPECTED_EVALUATED_FRAMES:
        raise RuntimeError(
            f"Evaluation protocol mismatch: expected {EXPECTED_EVALUATED_FRAMES}, "
            f"got {evaluated_frame_count}."
        )
    mious = {name: add_metrics(value) for name, value in confusion.items()}
    recovery = (
        (mious["clean_state_injection"] - mious["noisy_static"])
        / (mious["clean_static"] - mious["noisy_static"])
    )
    headroom = (
        mious["oracle_clipped_gain_closed_loop"] - mious["learned_context"]
    )
    posterior_mse = {
        name: {
            key: value[key] / evaluated_frame_count
            for key in ("prediction", "observation", "posterior")
        }
        for name, value in state_sums.items()
    }
    parameter_unchanged = all(
        torch.equal(parameter.detach().cpu(), parameter_snapshot[name])
        for module in modules
        for name, parameter in module.named_parameters()
    )
    gates = {
        "learned_context_reproduction": {
            "expected": EXPECTED_MIOU,
            "actual": mious["learned_context"],
            "absolute_error": abs(mious["learned_context"] - EXPECTED_MIOU),
            "pass": abs(mious["learned_context"] - EXPECTED_MIOU) <= 1e-6,
        },
        "dataset_protocol": {
            "sequence_count": sequence_count,
            "total_frame_count": total_frame_count,
            "evaluated_frame_count": evaluated_frame_count,
            "pass": True,
        },
        "oracle_gain_range": {"pass": gains_in_range},
        "oracle_mse_dominance": {
            name: {
                "posterior_not_above_prediction": values["posterior"] <= values["prediction"] + 1e-10,
                "posterior_not_above_observation": values["posterior"] <= values["observation"] + 1e-10,
            }
            for name, values in posterior_mse.items()
        },
        "injection_exact_before_decode": {"pass": injection_exact},
        "no_training": {
            "optimizer_created": False,
            "backward_called": False,
            "parameters_unchanged": parameter_unchanged,
        },
        "finite": {"pass": finite},
    }
    gates["oracle_mse_dominance"]["pass"] = all(
        value["posterior_not_above_prediction"] and value["posterior_not_above_observation"]
        for name, value in gates["oracle_mse_dominance"].items()
        if name in ("z1", "z4")
    )
    gates["all_pass"] = all(
        gate.get("pass", False) for name, gate in gates.items()
        if name not in ("learned_context_reproduction", "oracle_mse_dominance")
    ) and gates["learned_context_reproduction"]["pass"] and gates["oracle_mse_dominance"]["pass"]

    if not gates["learned_context_reproduction"]["pass"]:
        diagnosis = "REPRODUCTION_FAILED"
    elif recovery < 0.80:
        diagnosis = "ADAPTER_WRITEBACK_BOTTLENECK"
    elif headroom >= 0.020:
        diagnosis = "GAIN_LEARNING_OBJECTIVE_BOTTLENECK"
    elif headroom < 0.005:
        diagnosis = "INTERPOLATION_OR_PREDICTOR_ENDPOINT_BOTTLENECK"
    else:
        diagnosis = "LIMITED_HEADROOM"

    summary = {
        "experiment": "kitti_step_oracle_upper_bound_diagnostic",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "protocol": {
            "sigma": SIGMA,
            "seed": SEED,
            "sequence_count": sequence_count,
            "total_frame_count": total_frame_count,
            "evaluated_frame_count": evaluated_frame_count,
            "paired_noisy_stream": True,
        },
        "checkpoints": {
            **{name: str(path) for name, path in checkpoints.items()},
            "learned_context": str(context_checkpoint),
        },
        "mIoU": mious,
        "recovery": {"clean_state_injection_relative_to_clean_noisy": recovery},
        "headroom": {
            "oracle_clipped_gain_closed_loop_minus_learned_context": headroom
        },
        "oracle_state_mse": posterior_mse,
        "oracle_gain": {
            "min": gain_stats["min"],
            "max": gain_stats["max"],
            "mean": gain_stats["sum"] / gain_stats["count"],
            "near_zero_fraction": gain_stats["near_zero_count"] / gain_stats["count"],
            "near_one_fraction": gain_stats["near_one_count"] / gain_stats["count"],
            "near_threshold": GAIN_EPSILON,
        },
        "gates": gates,
        "finite_is_not_stability": True,
        "diagnosis": diagnosis,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
