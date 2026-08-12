import torch
import torch.nn.functional as F


def candidate_shifts(radius):
    radius = int(radius)
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}.")
    shifts = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    return tuple(
        sorted(
            shifts,
            key=lambda shift: (
                shift[0] * shift[0] + shift[1] * shift[1],
                abs(shift[0]) + abs(shift[1]),
                shift[0],
                shift[1],
            ),
        )
    )


def translate_feature(feature, dy, dx):
    """Translate a BCHW feature map with zero fill and no wraparound."""
    if feature.ndim != 4:
        raise ValueError(f"Expected BCHW feature tensor, got shape {tuple(feature.shape)}.")
    dy = int(dy)
    dx = int(dx)
    height, width = feature.shape[-2:]
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError(
            f"Shift {(dy, dx)} must be smaller than feature shape {(height, width)}."
        )

    translated = torch.zeros_like(feature)
    source_y_start = max(0, -dy)
    source_y_stop = min(height, height - dy)
    source_x_start = max(0, -dx)
    source_x_stop = min(width, width - dx)
    target_y_start = max(0, dy)
    target_y_stop = min(height, height + dy)
    target_x_start = max(0, dx)
    target_x_stop = min(width, width + dx)
    translated[..., target_y_start:target_y_stop, target_x_start:target_x_stop] = (
        feature[..., source_y_start:source_y_stop, source_x_start:source_x_stop]
    )
    return translated


def patch_descriptors(feature, patch_size):
    patch_size = int(patch_size)
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be a positive odd integer, got {patch_size}.")
    batch_size, channels, height, width = feature.shape
    unfolded = F.unfold(
        feature,
        kernel_size=patch_size,
        padding=patch_size // 2,
    )
    return unfolded.reshape(batch_size, channels * patch_size * patch_size, height, width)


def estimate_local_displacement(source, target, radius, patch_size):
    """Find a source-to-target displacement at every target location."""
    if source.shape != target.shape or source.ndim != 4:
        raise ValueError("source and target must be matching BCHW tensors.")
    source_descriptors = patch_descriptors(source, patch_size)
    target_descriptors = patch_descriptors(target, patch_size)
    batch_size, _, height, width = source.shape
    valid_source = torch.ones(
        batch_size,
        1,
        height,
        width,
        dtype=source.dtype,
        device=source.device,
    )
    best_cost = torch.full(
        (batch_size, height, width),
        float("inf"),
        dtype=source.dtype,
        device=source.device,
    )
    best_dy = torch.zeros_like(best_cost, dtype=torch.int64)
    best_dx = torch.zeros_like(best_cost, dtype=torch.int64)
    matched_source = torch.zeros_like(source)

    for dy, dx in candidate_shifts(radius):
        if abs(dy) >= height or abs(dx) >= width:
            continue
        candidate_descriptors = translate_feature(source_descriptors, dy=dy, dx=dx)
        candidate_source = translate_feature(source, dy=dy, dx=dx)
        valid = translate_feature(valid_source, dy=dy, dx=dx).squeeze(1) > 0.5
        cost = (candidate_descriptors - target_descriptors).square().mean(dim=1)
        cost = torch.where(valid, cost, torch.full_like(cost, float("inf")))
        improved = cost < best_cost
        best_cost = torch.where(improved, cost, best_cost)
        best_dy = torch.where(improved, dy, best_dy)
        best_dx = torch.where(improved, dx, best_dx)
        matched_source = torch.where(improved.unsqueeze(1), candidate_source, matched_source)

    return {
        "dy": best_dy,
        "dx": best_dx,
        "patch_matching_cost": best_cost,
        "matched_source": matched_source,
    }


def align_source_to_target(source, target, radius, patch_size):
    """Place source features in target coordinates using causal local matching."""
    motion = estimate_local_displacement(
        source,
        target,
        radius=radius,
        patch_size=patch_size,
    )
    return {
        "aligned_source": motion["matched_source"],
        "dy": motion["dy"],
        "dx": motion["dx"],
        "patch_matching_cost": motion["patch_matching_cost"],
    }


def forward_splat_discrete(feature, dy_field, dx_field, radius):
    """Move current features forward; average collisions and copy-fill holes."""
    if feature.ndim != 4 or dy_field.shape != feature.shape[:1] + feature.shape[-2:]:
        raise ValueError("Motion fields must have shape BHW matching the BCHW feature.")
    if dx_field.shape != dy_field.shape:
        raise ValueError("dy_field and dx_field must have the same shape.")

    accumulated = torch.zeros_like(feature)
    weights = torch.zeros(
        feature.shape[0],
        1,
        feature.shape[2],
        feature.shape[3],
        dtype=feature.dtype,
        device=feature.device,
    )
    for dy, dx in candidate_shifts(radius):
        if abs(dy) >= feature.shape[2] or abs(dx) >= feature.shape[3]:
            continue
        mask = ((dy_field == dy) & (dx_field == dx)).unsqueeze(1).to(feature.dtype)
        accumulated = accumulated + translate_feature(feature * mask, dy=dy, dx=dx)
        weights = weights + translate_feature(mask, dy=dy, dx=dx)

    covered = weights > 0
    warped = torch.where(covered, accumulated / weights.clamp_min(1.0), feature)
    return {
        "warped": warped,
        "coverage_fraction": covered.flatten(1).float().mean(dim=1),
        "collision_fraction": (weights > 1).flatten(1).float().mean(dim=1),
    }
