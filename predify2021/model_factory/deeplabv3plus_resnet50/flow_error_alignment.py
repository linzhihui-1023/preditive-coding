import torch
from torch.nn import functional as F


def backward_flow_to_state_warp(previous_error, backward_flow):
    """Warp prior error with current-to-previous pixel correspondence flow."""
    _, _, height, width = previous_error.shape
    flow = F.interpolate(
        backward_flow,
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    )
    flow_x = flow[:, 0] * (width / backward_flow.shape[-1])
    flow_y = flow[:, 1] * (height / backward_flow.shape[-2])
    y, x = torch.meshgrid(
        torch.arange(height, device=previous_error.device, dtype=flow.dtype),
        torch.arange(width, device=previous_error.device, dtype=flow.dtype),
        indexing="ij",
    )
    source_x = x.unsqueeze(0) + flow_x
    source_y = y.unsqueeze(0) + flow_y
    valid = (
        (source_x >= 0)
        & (source_x <= width - 1)
        & (source_y >= 0)
        & (source_y <= height - 1)
    )
    grid = torch.stack(
        (
            2.0 * source_x / (width - 1) - 1.0,
            2.0 * source_y / (height - 1) - 1.0,
        ),
        dim=-1,
    )
    warped = F.grid_sample(
        previous_error,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped * valid.unsqueeze(1), valid
