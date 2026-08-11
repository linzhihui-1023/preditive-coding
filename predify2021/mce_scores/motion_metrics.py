import torch


def _component_metrics(absolute_error, squared_error, prefix, unit_suffix):
    return {
        f"{prefix}_mse_{unit_suffix}2": float(squared_error.mean().item()),
        f"{prefix}_rmse_{unit_suffix}": float(squared_error.mean().sqrt().item()),
        f"{prefix}_mae_{unit_suffix}": float(absolute_error.mean().item()),
        f"{prefix}_median_ae_{unit_suffix}": float(
            torch.quantile(absolute_error, 0.5).item()
        ),
        f"{prefix}_p95_ae_{unit_suffix}": float(
            torch.quantile(absolute_error, 0.95).item()
        ),
    }


def compute_motion_diagnostics(predictions, targets, target_mean=None, target_std=None):
    predictions = torch.as_tensor(predictions).detach().cpu().float().reshape(-1, 2)
    targets = torch.as_tensor(targets).detach().cpu().float().reshape(-1, 2)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {predictions.shape} versus {targets.shape}."
        )
    if predictions.shape[0] == 0:
        raise ValueError("Cannot compute motion diagnostics for an empty sample set.")

    error = predictions - targets
    absolute_error = error.abs()
    squared_error = error.square()
    metrics = {
        "sample_count": int(predictions.shape[0]),
        **_component_metrics(absolute_error[:, 0], squared_error[:, 0], "forward", "m"),
        **_component_metrics(absolute_error[:, 1], squared_error[:, 1], "yaw", "rad"),
    }

    if target_mean is not None or target_std is not None:
        if target_mean is None or target_std is None:
            raise ValueError("target_mean and target_std must be provided together.")
        mean = torch.as_tensor(target_mean).detach().cpu().float().reshape(1, 2)
        std = torch.as_tensor(target_std).detach().cpu().float().reshape(1, 2)
        if torch.any(std <= 0):
            raise ValueError(f"Target standard deviations must be positive: {std.tolist()}.")
        standardized_error = error / std
        metrics["standardized_joint_mse"] = float(standardized_error.square().mean().item())
        metrics["forward_standardized_mse"] = float(
            standardized_error[:, 0].square().mean().item()
        )
        metrics["yaw_standardized_mse"] = float(
            standardized_error[:, 1].square().mean().item()
        )

    return metrics
