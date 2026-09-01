"""Diagnose semantic-reference quality and long-horizon semantic-state drift.

No parameters are updated. The frozen RoleSeparatedRecurrentPredictor is evaluated
under the same Clean / Blur-Mid / Blur-Max diagnostic conditions used by the
current quick screens.

For each current frame t, the pending semantic reference M_t^sem produced at
t-1 is compared against the clean current unified state Z_t^clean.

Two recurrence modes are evaluated with identical weights:
  continuous: semantic ConvGRU states run for the full sequence.
  reset8: only h4_sem/h1_sem are reset every 8 input frames; dynamics states
          remain continuous. This is a diagnostic control for the original
          8-frame semantic-predictor training horizon.

Outputs:
  summary.json  overall + position-binned metrics and drift decisions
  binned.csv    compact table for plotting/comparison
  frames.csv    per-frame latent metrics
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
    sequence_groups,
)
from predify2021.mce_scores.evaluate_kitti_step_semantic_recurrent_predictor import (
    decode_state,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.role_separated_direct_state_correction import (
    load_role_components,
    make_paths,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    error_state,
    zero_state,
)
from predify2021.mce_scores.train_kitti_step_convgru_decoupling_quick import (
    diagnostic_blur,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature

SEED = 0
CONDITIONS = ("Clean", "Blur-Mid", "Blur-Max")
MODES = ("continuous", "reset8")
BINS = (
    ("0-31", 0, 32),
    ("32-63", 32, 64),
    ("64-127", 64, 128),
    ("128+", 128, None),
)


def position_bin(frame_index):
    for name, start, end in BINS:
        if frame_index >= start and (end is None or frame_index < end):
            return name
    raise RuntimeError(f"unhandled frame index {frame_index}")


def latent_metrics(reference, target):
    result = {}
    for stage in ("z1", "z4"):
        ref = getattr(reference, stage)
        clean = getattr(target, stage)
        result[f"mse_{stage}"] = float(F.mse_loss(ref, clean).item())
        result[f"cos_{stage}"] = float(
            F.cosine_similarity(ref, clean, dim=1).mean().item()
        )
    result["mse_mean"] = 0.5 * (result["mse_z1"] + result["mse_z4"])
    result["cos_mean"] = 0.5 * (result["cos_z1"] + result["cos_z4"])
    return result


def new_aggregate():
    return {
        "count": 0,
        "semantic_mse_z1": 0.0,
        "semantic_mse_z4": 0.0,
        "semantic_mse_mean": 0.0,
        "semantic_cos_z1": 0.0,
        "semantic_cos_z4": 0.0,
        "semantic_cos_mean": 0.0,
        "observation_mse_z1": 0.0,
        "observation_mse_z4": 0.0,
        "observation_mse_mean": 0.0,
        "observation_cos_z1": 0.0,
        "observation_cos_z4": 0.0,
        "observation_cos_mean": 0.0,
    }


def add_aggregate(aggregate, semantic, observation):
    aggregate["count"] += 1
    for prefix, values in (("semantic", semantic), ("observation", observation)):
        for metric in ("mse_z1", "mse_z4", "mse_mean", "cos_z1", "cos_z4", "cos_mean"):
            aggregate[f"{prefix}_{metric}"] += values[metric]


def finalize_aggregate(aggregate):
    count = max(aggregate["count"], 1)
    result = {"count": aggregate["count"]}
    for key, value in aggregate.items():
        if key == "count":
            continue
        result[key] = value / count
    result["semantic_mse_ratio_vs_observation"] = (
        result["semantic_mse_mean"] / max(result["observation_mse_mean"], 1e-12)
    )
    result["semantic_cos_gain_vs_observation"] = (
        result["semantic_cos_mean"] - result["observation_cos_mean"]
    )
    return result


def decode_host(model, raw, output_size):
    return model.decode_from_host_feature(
        HostFeature(raw.c4, raw.c1, output_size)
    )


def evaluate_condition(model, predictor, groups, condition, reset_period):
    aggregates = {
        mode: defaultdict(new_aggregate)
        for mode in MODES
    }
    reference_confusion = {
        mode: {
            "overall": torch.zeros((19, 19), dtype=torch.int64),
            **{
                name: torch.zeros((19, 19), dtype=torch.int64)
                for name, _, _ in BINS
            },
        }
        for mode in MODES
    }
    host_confusion = {
        "clean": {
            "overall": torch.zeros((19, 19), dtype=torch.int64),
            **{
                name: torch.zeros((19, 19), dtype=torch.int64)
                for name, _, _ in BINS
            },
        },
        "corrupted": {
            "overall": torch.zeros((19, 19), dtype=torch.int64),
            **{
                name: torch.zeros((19, 19), dtype=torch.int64)
                for name, _, _ in BINS
            },
        },
    }
    frame_rows = []

    with torch.inference_mode():
        for sequence, samples in groups.items():
            states = {
                mode: {
                    "hidden": predictor.initial_state(),
                    "pending_dynamics": None,
                    "pending_semantic": None,
                }
                for mode in MODES
            }

            for frame_index, sample in enumerate(samples):
                clean_image = load_image(sample)
                noisy_image = diagnostic_blur(
                    clean_image, frame_index, len(samples), condition
                )

                clean_raw = model.extract_backbone_features(clean_image)
                clean_state = model.encode_backbone_features(clean_raw)
                noisy_raw = model.extract_backbone_features(noisy_image)
                observation = model.encode_backbone_features(noisy_raw)
                output_size = tuple(clean_image.shape[-2:])
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])

                clean_prediction = decode_host(
                    model, clean_raw, output_size
                ).argmax(1).squeeze(0).cpu()
                corrupted_prediction = decode_host(
                    model, noisy_raw, output_size
                ).argmax(1).squeeze(0).cpu()

                bin_name = position_bin(frame_index)

                update_confusion_matrix(
                    host_confusion["clean"]["overall"], clean_prediction, mask
                )
                update_confusion_matrix(
                    host_confusion["clean"][bin_name], clean_prediction, mask
                )
                update_confusion_matrix(
                    host_confusion["corrupted"]["overall"], corrupted_prediction, mask
                )
                update_confusion_matrix(
                    host_confusion["corrupted"][bin_name], corrupted_prediction, mask
                )

                if frame_index == 0:
                    zero = zero_state(observation)
                    for mode in MODES:
                        dynamics, semantic, *hidden = predictor.step(
                            observation, zero, *states[mode]["hidden"]
                        )
                        states[mode]["pending_dynamics"] = dynamics
                        states[mode]["pending_semantic"] = semantic
                        states[mode]["hidden"] = tuple(hidden)
                    continue

                observation_quality = latent_metrics(observation, clean_state)

                for mode in MODES:
                    state = states[mode]
                    semantic_reference = state["pending_semantic"]
                    semantic_quality = latent_metrics(
                        semantic_reference, clean_state
                    )
                    semantic_prediction = decode_state(
                        model, semantic_reference, output_size
                    ).argmax(1).squeeze(0).cpu()

                    update_confusion_matrix(
                        reference_confusion[mode]["overall"],
                        semantic_prediction,
                        mask,
                    )
                    update_confusion_matrix(
                        reference_confusion[mode][bin_name],
                        semantic_prediction,
                        mask,
                    )

                    add_aggregate(
                        aggregates[mode]["overall"],
                        semantic_quality,
                        observation_quality,
                    )
                    add_aggregate(
                        aggregates[mode][bin_name],
                        semantic_quality,
                        observation_quality,
                    )

                    frame_rows.append(
                        {
                            "condition": condition,
                            "mode": mode,
                            "sequence": sequence,
                            "frame_index": frame_index,
                            "position_bin": bin_name,
                            **{
                                f"semantic_{key}": value
                                for key, value in semantic_quality.items()
                            },
                            **{
                                f"observation_{key}": value
                                for key, value in observation_quality.items()
                            },
                        }
                    )

                # Advance both diagnostic streams to predict frame t+1.
                for mode in MODES:
                    state = states[mode]
                    current_error = error_state(
                        observation, state["pending_dynamics"]
                    )

                    h4_dyn, h4_sem, h1_dyn, h1_sem = state["hidden"]
                    if mode == "reset8" and frame_index % reset_period == 0:
                        h4_sem = None
                        h1_sem = None

                    dynamics, semantic, *hidden = predictor.step(
                        observation,
                        current_error,
                        h4_dyn,
                        h4_sem,
                        h1_dyn,
                        h1_sem,
                    )
                    state["pending_dynamics"] = dynamics
                    state["pending_semantic"] = semantic
                    state["hidden"] = tuple(hidden)

    summary = {
        "condition": condition,
        "modes": {},
        "host": {},
    }

    for mode in MODES:
        summary["modes"][mode] = {}
        for key, aggregate in aggregates[mode].items():
            metrics = finalize_aggregate(aggregate)
            metrics["decoded_mIoU"] = float(
                torch.nanmean(compute_iou(reference_confusion[mode][key])).item()
            )
            summary["modes"][mode][key] = metrics

    for host_name in ("clean", "corrupted"):
        summary["host"][host_name] = {}
        for key, confusion in host_confusion[host_name].items():
            summary["host"][host_name][key] = {
                "mIoU": float(torch.nanmean(compute_iou(confusion)).item())
            }

    return summary, frame_rows


def drift_decision(condition_summary):
    continuous = condition_summary["modes"]["continuous"]
    reset8 = condition_summary["modes"]["reset8"]

    early = continuous.get("0-31")
    late = continuous.get("128+")
    if early is None or late is None or late["count"] == 0:
        return {
            "available": False,
            "reason": "sequence has no 128+ bin",
        }

    continuous_mse_growth = (
        late["semantic_mse_mean"] / max(early["semantic_mse_mean"], 1e-12)
    )
    continuous_cos_drop = (
        early["semantic_cos_mean"] - late["semantic_cos_mean"]
    )
    reset8_late_mse_improvement = (
        1.0
        - reset8["128+"]["semantic_mse_mean"]
        / max(late["semantic_mse_mean"], 1e-12)
    )
    reset8_late_miou_gain = (
        reset8["128+"]["decoded_mIoU"] - late["decoded_mIoU"]
    )

    supported = (
        continuous_mse_growth >= 1.10
        and (
            reset8_late_mse_improvement >= 0.05
            or reset8_late_miou_gain >= 0.005
        )
    )

    return {
        "available": True,
        "continuous_late_vs_early_mse_ratio": continuous_mse_growth,
        "continuous_early_to_late_cos_drop": continuous_cos_drop,
        "reset8_late_mse_improvement": reset8_late_mse_improvement,
        "reset8_late_decoded_mIoU_gain": reset8_late_miou_gain,
        "SEMANTIC_STATE_DRIFT": "SUPPORTED" if supported else "NOT SUPPORTED",
        "note": (
            "Diagnostic criterion only. Main claim must rely on formal task metrics."
        ),
    }


def write_binned_csv(path, condition_summaries):
    rows = []
    for condition, summary in condition_summaries.items():
        for mode in MODES:
            for bin_name in ("overall",) + tuple(name for name, _, _ in BINS):
                if bin_name not in summary["modes"][mode]:
                    continue
                rows.append(
                    {
                        "condition": condition,
                        "mode": mode,
                        "position_bin": bin_name,
                        **summary["modes"][mode][bin_name],
                    }
                )

    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=sorted({key for row in rows for key in row}),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/lin/predify/kitti_step",
    )
    parser.add_argument(
        "--output",
        default="results/kitti_step_semantic_reference_quality_diagnostic",
    )
    parser.add_argument(
        "--sequence-limit",
        type=int,
        default=0,
        help="0 evaluates all validation sequences.",
    )
    parser.add_argument(
        "--reset-period",
        type=int,
        default=8,
        help="Semantic-state-only reset period for the diagnostic control.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    paths = make_paths()
    model, predictor = load_role_components(
        paths["static"],
        paths["adapter"],
        paths["predictor"],
        paths["writeback"],
    )
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    model.eval()
    predictor.eval()

    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root), "val"
    )
    groups = sequence_groups(dataset)
    if args.sequence_limit:
        groups = dict(list(groups.items())[: args.sequence_limit])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    condition_summaries = {}
    all_frame_rows = []

    for condition in CONDITIONS:
        summary, frame_rows = evaluate_condition(
            model,
            predictor,
            groups,
            condition,
            args.reset_period,
        )
        condition_summaries[condition] = summary
        all_frame_rows.extend(frame_rows)
        print(
            json.dumps(
                {
                    "condition": condition,
                    "continuous": summary["modes"]["continuous"]["overall"],
                    "reset8": summary["modes"]["reset8"]["overall"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    decisions = {
        condition: drift_decision(summary)
        for condition, summary in condition_summaries.items()
    }

    result = {
        "experiment": "semantic_reference_quality_diagnostic",
        "parameters_updated": False,
        "protocol": {
            "split": "val",
            "sequence_names": list(groups.keys()),
            "sequence_count": len(groups),
            "conditions": {
                "Clean": "none",
                "Blur-Mid": "diagnostic sigma=2.25 with existing warmup",
                "Blur-Max": "diagnostic sigma=3.0 with existing warmup",
            },
            "semantic_state_control": (
                f"reset only h4_sem/h1_sem every {args.reset_period} frames; "
                "dynamics recurrent states remain continuous"
            ),
            "position_bins": [name for name, _, _ in BINS],
        },
        "conditions": condition_summaries,
        "decisions": decisions,
    }

    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    write_binned_csv(output / "binned.csv", condition_summaries)

    with (output / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=sorted(
                {key for row in all_frame_rows for key in row}
            ),
        )
        writer.writeheader()
        writer.writerows(all_frame_rows)


if __name__ == "__main__":
    main()
