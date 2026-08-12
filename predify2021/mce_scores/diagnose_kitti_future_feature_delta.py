import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from predify2021.mce_scores.kitti_pairs import build_kitti_multi_horizon_dataset
from predify2021.model_factory.get_model import get_model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-12


def _parse_drives(value):
    drives = tuple(item.strip() for item in value.split(",") if item.strip())
    if not drives:
        raise ValueError("Expected at least one drive.")
    return drives


def _population_moments(total, squared_total, count):
    mean = total / count
    variance = max(0.0, squared_total / count - mean * mean)
    return {"mean": mean, "std": math.sqrt(variance), "element_count": count}


def _distribution(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    quantiles = torch.quantile(
        tensor,
        torch.tensor([0.05, 0.5, 0.9, 0.95], dtype=torch.float64),
    )
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "p05": quantiles[0].item(),
        "p50": quantiles[1].item(),
        "p90": quantiles[2].item(),
        "p95": quantiles[3].item(),
    }


def _build_model(checkpoint):
    config = checkpoint.get("config", {})
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
        task="future_feature",
        future_feature_history_mode="none",
        future_feature_predictor_kernel_size=config.get(
            "future_feature_predictor_kernel_size",
            1,
        ),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model.to(DEVICE)


def diagnose_split(model, dataset, split):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(os.environ.get("PREDIFY_DIAGNOSTIC_NUM_WORKERS", "0")),
        pin_memory=DEVICE.type == "cuda",
    )
    target_sum = 0.0
    target_squared_sum = 0.0
    predicted_sum = 0.0
    predicted_squared_sum = 0.0
    element_count = 0
    frame_metrics = {
        name: []
        for name in (
            "target_delta_l2",
            "predicted_delta_l2",
            "target_delta_rms",
            "predicted_delta_rms",
            "predicted_to_target_norm_ratio",
            "delta_cosine",
            "target_projection_coefficient",
            "learned_feature_mse",
            "copy_current_feature_mse",
        )
    }

    with torch.inference_mode():
        for current_frames, future_frames, _, _ in tqdm(loader, desc=split):
            current_frames = current_frames.to(DEVICE, non_blocking=True)
            next_frames = future_frames[:, 0].to(DEVICE, non_blocking=True)
            paired_top = model.extract_top_forward_feature(
                torch.cat([current_frames, next_frames], dim=0),
                detach=True,
            )
            current_top, next_top = paired_top.chunk(2, dim=0)
            target_delta = next_top - current_top
            predictor_input = torch.cat(
                [current_top, torch.zeros_like(current_top)],
                dim=1,
            )
            predicted_delta = model.future_feature_predictor(predictor_input)

            target_flat = target_delta.flatten(1)
            predicted_flat = predicted_delta.flatten(1)
            target_norm = torch.linalg.vector_norm(target_flat, dim=1)
            predicted_norm = torch.linalg.vector_norm(predicted_flat, dim=1)
            dimensions = target_flat.shape[1]

            target_double = target_delta.double()
            predicted_double = predicted_delta.double()
            target_sum += target_double.sum().item()
            target_squared_sum += target_double.square().sum().item()
            predicted_sum += predicted_double.sum().item()
            predicted_squared_sum += predicted_double.square().sum().item()
            element_count += target_delta.numel()

            frame_metrics["target_delta_l2"].extend(target_norm.cpu().tolist())
            frame_metrics["predicted_delta_l2"].extend(predicted_norm.cpu().tolist())
            frame_metrics["target_delta_rms"].extend(
                (target_norm / math.sqrt(dimensions)).cpu().tolist()
            )
            frame_metrics["predicted_delta_rms"].extend(
                (predicted_norm / math.sqrt(dimensions)).cpu().tolist()
            )
            frame_metrics["predicted_to_target_norm_ratio"].extend(
                (predicted_norm / target_norm.clamp_min(EPS)).cpu().tolist()
            )
            frame_metrics["delta_cosine"].extend(
                F.cosine_similarity(predicted_flat, target_flat, dim=1).cpu().tolist()
            )
            projection = (predicted_flat * target_flat).sum(dim=1) / (
                target_flat.square().sum(dim=1).clamp_min(EPS)
            )
            frame_metrics["target_projection_coefficient"].extend(
                projection.cpu().tolist()
            )
            frame_metrics["learned_feature_mse"].extend(
                (predicted_flat - target_flat).square().mean(dim=1).cpu().tolist()
            )
            frame_metrics["copy_current_feature_mse"].extend(
                target_flat.square().mean(dim=1).cpu().tolist()
            )

    distributions = {
        name: _distribution(values) for name, values in frame_metrics.items()
    }
    learned_mse = distributions["learned_feature_mse"]["mean"]
    copy_mse = distributions["copy_current_feature_mse"]["mean"]
    return {
        "split": split,
        "num_pairs": len(dataset),
        "target_delta_elements": _population_moments(
            target_sum,
            target_squared_sum,
            element_count,
        ),
        "predicted_delta_elements": _population_moments(
            predicted_sum,
            predicted_squared_sum,
            element_count,
        ),
        **distributions,
        "learned_minus_copy_mse": learned_mse - copy_mse,
        "learned_vs_copy_percent": 100.0 * (learned_mse / copy_mse - 1.0),
    }


def main():
    checkpoint_path = Path(os.environ["PREDIFY_DIAGNOSTIC_CHECKPOINT"])
    output_path = Path(
        os.environ.get(
            "PREDIFY_DIAGNOSTIC_OUTPUT_PATH",
            checkpoint_path.with_suffix(".delta_diagnostics.json"),
        )
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = _build_model(checkpoint)
    root = os.environ.get("PREDIFY_KITTI_ROOT", "/home/lin/predify/kitti_raw")
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    fixed_dt = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))
    train_drives = _parse_drives(os.environ["PREDIFY_TRAIN_DRIVES"])
    val_drives = _parse_drives(os.environ["PREDIFY_VAL_DRIVES"])
    train_dataset = build_kitti_multi_horizon_dataset(
        root,
        train_drives,
        camera=camera,
        horizons=(1,),
        fixed_dt_s=fixed_dt,
        dt_tolerance_s=tolerance,
    )
    val_dataset = build_kitti_multi_horizon_dataset(
        root,
        val_drives,
        camera=camera,
        horizons=(1,),
        fixed_dt_s=fixed_dt,
        dt_tolerance_s=tolerance,
    )

    selected_epoch = checkpoint.get("selected_epoch")
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_kind": checkpoint.get("checkpoint_kind"),
        "checkpoint_config": checkpoint.get("config", {}),
        "selected_epoch": selected_epoch,
        "device": str(DEVICE),
        "diagnostic_definition": {
            "target_delta": "F_(t+1)-F_t",
            "predicted_delta": "P(F_t, zeros)",
            "prediction": "F_t+predicted_delta",
            "copy_current": "F_t",
            "l2_quantiles": "per-frame flattened stage-5 feature delta",
            "rms_quantiles": "per-frame L2/sqrt(num_stage5_elements)",
        },
        "train": diagnose_split(model, train_dataset, "train"),
        "val": diagnose_split(model, val_dataset, "val"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({"output_path": str(output_path), "train": result["train"], "val": result["val"]}, indent=2))


if __name__ == "__main__":
    main()
