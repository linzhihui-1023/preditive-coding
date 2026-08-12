import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from predify2021.mce_scores.kitti_pairs import KITTIMultiHorizonFrameDataset
from predify2021.model_factory.get_model import get_model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-12
EXPECTED_CONDITIONS = {
    "copy_current": {
        "history_mode": "copy_current",
        "fusion_mode": "none",
        "optimizer_created": False,
    },
    "current_only": {
        "history_mode": "none",
        "fusion_mode": "none",
        "optimizer_created": True,
    },
    "temporal_fusion": {
        "history_mode": "none",
        "fusion_mode": "two_frame_residual",
        "optimizer_created": True,
    },
}


def _parse_checkpoints(value):
    checkpoints = {}
    for item in value.split(","):
        if not item.strip():
            continue
        label, separator, path = item.partition("=")
        if not separator or not label.strip() or not path.strip():
            raise ValueError(f"Invalid checkpoint mapping: {item!r}.")
        checkpoints[label.strip()] = Path(path.strip())
    if set(checkpoints) != set(EXPECTED_CONDITIONS):
        raise ValueError(
            f"Expected checkpoint labels {sorted(EXPECTED_CONDITIONS)}, "
            f"got {sorted(checkpoints)}."
        )
    return checkpoints


def _validate_trainable_parameters(label, config):
    names = tuple(config.get("optimized_parameter_names", ()))
    expected_optimizer = EXPECTED_CONDITIONS[label]["optimizer_created"]
    if bool(config.get("optimizer_created")) != expected_optimizer:
        raise ValueError(f"{label} has an unexpected optimizer configuration.")
    if label == "copy_current":
        if names:
            raise ValueError("copy_current must have no optimized parameters.")
        return names
    allowed_prefixes = ["future_feature_predictor."]
    if label == "temporal_fusion":
        allowed_prefixes.append("temporal_fusion_module.")
    if not names or not all(
        any(name.startswith(prefix) for prefix in allowed_prefixes) for name in names
    ):
        raise ValueError(f"{label} contains an unexpected trainable parameter family.")
    for prefix in allowed_prefixes:
        if not any(name.startswith(prefix) for name in names):
            raise ValueError(f"{label} does not optimize {prefix}")
    return names


def validate_checkpoint(label, checkpoint, expected_revision):
    config = checkpoint.get("config")
    if not isinstance(config, dict) or config.get("prediction_task") != "future_feature":
        raise ValueError(f"{label} is not a future-feature checkpoint.")
    expected = EXPECTED_CONDITIONS[label]
    observed = (
        config.get("future_feature_history_mode"),
        config.get("future_feature_temporal_fusion_mode", "none"),
    )
    required = (expected["history_mode"], expected["fusion_mode"])
    if observed != required:
        raise ValueError(f"{label} expected history/fusion {required}, got {observed}.")
    if config.get("future_feature_prediction_form") != "current_residual":
        raise ValueError(f"{label} must use current_residual prediction.")
    if config.get("git_revision") != expected_revision:
        raise ValueError(
            f"{label} checkpoint revision {config.get('git_revision')!r} "
            f"!= {expected_revision!r}."
        )
    if checkpoint.get("checkpoint_kind") != "best_val_future_feature_mse":
        raise ValueError(f"{label} is not a best-validation future-feature checkpoint.")
    frozen_contract = {
        "pretrained": True,
        "train_backbone": False,
        "feedback_decoder_trainable": False,
        "top_target_source": "student_self",
        "stream_mode": True,
        "reset_each_frame": False,
        "shuffle_train_pairs": False,
        "shuffle_val_pairs": False,
    }
    for key, required_value in frozen_contract.items():
        if config.get(key) != required_value:
            raise ValueError(
                f"{label} expected {key}={required_value!r}, got {config.get(key)!r}."
            )
    if float(config.get("local_reconstruction_weight", -1.0)) != 0.0:
        raise ValueError(f"{label} must disable local reconstruction training.")
    trainable_names = _validate_trainable_parameters(label, config)
    return {
        "history_mode": observed[0],
        "fusion_mode": observed[1],
        "predictor_kernel_size": int(config["future_feature_predictor_kernel_size"]),
        "selected_epoch": checkpoint.get("selected_epoch"),
        "checkpoint_revision": config["git_revision"],
        "optimized_parameter_names": trainable_names,
    }


def build_model(checkpoint, validation):
    config = checkpoint["config"]
    model = get_model(
        "pvgg_tf",
        pretrained=False,
        target_flow_mode=config.get("target_flow_mode", "recursive"),
        temporal_target_mode=config.get("temporal_target_mode", "next_top"),
        temporal_horizons=tuple(config.get("temporal_horizons", (1,))),
        dynamic_error=config.get("dynamic_error", True),
        error_state_mode=config.get("error_state_mode", "ema"),
        local_loss_error_source=config.get("local_loss_error_source", "instant"),
        error_sample_time=config.get("error_sample_time", 0.1035),
        error_time_constant=config.get("error_time_constant", 0.5),
        error_gain=config.get("error_gain", 1.0),
        temporal_error_sample_time=config.get("temporal_error_sample_time", 0.1035),
        temporal_error_time_constant=config.get("temporal_error_time_constant", 0.5),
        temporal_error_gain=config.get("temporal_error_gain", 1.0),
        task="future_feature",
        future_feature_history_mode=validation["history_mode"],
        future_feature_temporal_fusion_mode=validation["fusion_mode"],
        future_feature_predictor_kernel_size=validation["predictor_kernel_size"],
        future_feature_prediction_form="current_residual",
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model.to(DEVICE)


def _distribution(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    quantiles = torch.quantile(
        tensor,
        torch.tensor([0.05, 0.5, 0.9, 0.95], dtype=torch.float64),
    )
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "minimum": tensor.min().item(),
        "p05": quantiles[0].item(),
        "p50": quantiles[1].item(),
        "p90": quantiles[2].item(),
        "p95": quantiles[3].item(),
        "maximum": tensor.max().item(),
    }


def evaluate_split(model, dataset, label, split, drive, fixed_dt_s):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.reset()
    rows = []
    with torch.inference_mode():
        for sample_index, (current, futures, current_names, future_names) in enumerate(
            tqdm(loader, desc=f"{label}:{split}")
        ):
            current = current.to(DEVICE)
            future = futures[:, 0].to(DEVICE)
            model.step_frame(
                current,
                top_target_provider=lambda: model.extract_top_forward_feature(
                    future,
                    detach=True,
                ),
            )
            outputs = model.future_prediction_outputs
            predicted = outputs["predicted_future_top"].float()
            target = outputs["future_top_target"].float()
            current_top = outputs["current_top"].float()
            fused_top = outputs["fused_top"].float()
            fusion_residual = outputs["fusion_residual_top"].float()
            predicted_residual = outputs["predicted_residual_top"].float()
            predicted_flat = predicted.flatten(1)
            target_flat = target.flatten(1)
            raw_start = int(dataset.valid_start_indices[sample_index])
            rows.append(
                {
                    "condition": label,
                    "split": split,
                    "drive": drive,
                    "sample_index": sample_index,
                    "current_raw_frame_index": raw_start,
                    "future_raw_frame_index": raw_start + 1,
                    "current_frame_name": current_names[0],
                    "future_frame_name": future_names[0][0],
                    "horizon_frames": 1,
                    "horizon_seconds": fixed_dt_s,
                    "temporal_fusion_applied": bool(outputs["temporal_fusion_applied"]),
                    "feature_mse": float((predicted - target).square().mean().item()),
                    "copy_mse": float((current_top - target).square().mean().item()),
                    "feature_cosine": float(
                        F.cosine_similarity(predicted_flat, target_flat, dim=1).mean().item()
                    ),
                    "copy_cosine": float(
                        F.cosine_similarity(
                            current_top.flatten(1), target_flat, dim=1
                        ).mean().item()
                    ),
                    "normalized_feature_error": float(
                        (
                            torch.linalg.vector_norm(predicted_flat - target_flat, dim=1)
                            / torch.linalg.vector_norm(target_flat, dim=1).clamp_min(EPS)
                        ).mean().item()
                    ),
                    "copy_normalized_feature_error": float(
                        (
                            torch.linalg.vector_norm(
                                current_top.flatten(1) - target_flat, dim=1
                            )
                            / torch.linalg.vector_norm(target_flat, dim=1).clamp_min(EPS)
                        ).mean().item()
                    ),
                    "fusion_residual_rms": math.sqrt(
                        float(fusion_residual.square().mean().item())
                    ),
                    "fused_change_rms": math.sqrt(
                        float((fused_top - current_top).square().mean().item())
                    ),
                    "predicted_residual_rms": math.sqrt(
                        float(predicted_residual.square().mean().item())
                    ),
                    "current_feature_rms": math.sqrt(
                        float(current_top.square().mean().item())
                    ),
                    "target_feature_rms": math.sqrt(float(target.square().mean().item())),
                }
            )
    return rows


def summarize_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["split"])].append(row)
    metric_names = (
        "feature_mse",
        "copy_mse",
        "feature_cosine",
        "copy_cosine",
        "normalized_feature_error",
        "copy_normalized_feature_error",
        "fusion_residual_rms",
        "fused_change_rms",
        "predicted_residual_rms",
        "current_feature_rms",
        "target_feature_rms",
    )
    summary = {}
    for (condition, split), group in sorted(grouped.items()):
        summary[f"{condition}/{split}"] = {
            "condition": condition,
            "split": split,
            "sample_count": len(group),
            "temporal_fusion_applied_count": sum(
                bool(row["temporal_fusion_applied"]) for row in group
            ),
            "metrics": {
                metric: _distribution([float(row[metric]) for row in group])
                for metric in metric_names
            },
        }
    return summary


def _relative_reduction(baseline, candidate):
    return (baseline - candidate) / baseline


def compare_conditions(matrix):
    comparisons = {}
    for split in ("train", "val"):
        means = {
            condition: {
                metric: matrix[f"{condition}/{split}"]["metrics"][metric]["mean"]
                for metric in (
                    "feature_mse",
                    "feature_cosine",
                    "normalized_feature_error",
                )
            }
            for condition in EXPECTED_CONDITIONS
        }
        split_comparisons = {}
        for candidate, baseline in (
            ("current_only", "copy_current"),
            ("temporal_fusion", "copy_current"),
            ("temporal_fusion", "current_only"),
        ):
            split_comparisons[f"{candidate}_vs_{baseline}"] = {
                "feature_mse_reduction_fraction": _relative_reduction(
                    means[baseline]["feature_mse"], means[candidate]["feature_mse"]
                ),
                "feature_cosine_increase": (
                    means[candidate]["feature_cosine"]
                    - means[baseline]["feature_cosine"]
                ),
                "normalized_error_reduction_fraction": _relative_reduction(
                    means[baseline]["normalized_feature_error"],
                    means[candidate]["normalized_feature_error"],
                ),
            }
        comparisons[split] = split_comparisons
    return comparisons


def main():
    output_dir = Path(os.environ["PREDIFY_TEMPORAL_FUSION_EVAL_OUTPUT_DIR"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_revision = os.environ["PREDIFY_GIT_REVISION"]
    checkpoints = _parse_checkpoints(
        os.environ["PREDIFY_TEMPORAL_FUSION_CHECKPOINTS"]
    )
    root = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
    train_drive = os.environ["PREDIFY_TRAIN_DRIVES"]
    val_drive = os.environ["PREDIFY_VAL_DRIVES"]
    datasets = {
        "train": KITTIMultiHorizonFrameDataset(
            root,
            train_drive,
            camera=camera,
            horizons=(1,),
            fixed_dt_s=fixed_dt_s,
            dt_tolerance_s=tolerance,
        ),
        "val": KITTIMultiHorizonFrameDataset(
            root,
            val_drive,
            camera=camera,
            horizons=(1,),
            fixed_dt_s=fixed_dt_s,
            dt_tolerance_s=tolerance,
        ),
    }

    rows = []
    checkpoint_records = {}
    for label, checkpoint_path in checkpoints.items():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validation = validate_checkpoint(label, checkpoint, expected_revision)
        model = build_model(checkpoint, validation)
        checkpoint_records[label] = {"path": str(checkpoint_path), **validation}
        for split, drive in (("train", train_drive), ("val", val_drive)):
            rows.extend(
                evaluate_split(
                    model,
                    datasets[split],
                    label=label,
                    split=split,
                    drive=drive,
                    fixed_dt_s=fixed_dt_s,
                )
            )
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = output_dir / "per_frame.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    matrix = summarize_rows(rows)
    comparisons = compare_conditions(matrix)
    result = {
        "experiment": "kitti_stage5_two_frame_temporal_fusion_matrix",
        "git_revision": expected_revision,
        "device": str(DEVICE),
        "camera": camera,
        "fixed_dt_s": fixed_dt_s,
        "fixed_dt_tolerance_s": tolerance,
        "train_drive": train_drive,
        "val_drive": val_drive,
        "conditions": EXPECTED_CONDITIONS,
        "checkpoints": checkpoint_records,
        "matrix": matrix,
        "comparisons": comparisons,
        "per_frame_csv": "per_frame.csv",
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "row_count": len(rows),
                "comparisons": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
