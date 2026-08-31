import inspect
from functools import wraps

import numpy as np
import torch


CITYSCAPES_C_COMMON_CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)
CITYSCAPES_C_SEVERITIES = (1, 2, 3, 4, 5)


def _imagecorruptions():
    from imagecorruptions import corrupt
    from imagecorruptions import corruptions as imagecorruptions_impl

    gaussian = imagecorruptions_impl.gaussian
    if not getattr(gaussian, "_predify_channel_axis_compat", False):
        if "multichannel" not in inspect.signature(gaussian).parameters:
            @wraps(gaussian)
            def gaussian_compat(*args, **kwargs):
                if kwargs.pop("multichannel", False):
                    kwargs["channel_axis"] = -1
                return gaussian(*args, **kwargs)

            gaussian_compat._predify_channel_axis_compat = True
            imagecorruptions_impl.gaussian = gaussian_compat
    return corrupt


def apply_cityscapes_c_corruption_uint8(image, corruption, severity, seed=None):
    """Apply the published common corruption to an HWC uint8 RGB NumPy image."""
    if corruption not in CITYSCAPES_C_COMMON_CORRUPTIONS:
        raise ValueError(f"Unsupported Cityscapes-C corruption: {corruption}")
    if severity not in CITYSCAPES_C_SEVERITIES:
        raise ValueError(f"Severity must be one of {CITYSCAPES_C_SEVERITIES}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected HWC uint8 RGB image")

    if seed is not None:
        np.random.seed(int(seed) % (2**32 - 1))

    corrupted = _imagecorruptions()(
        np.ascontiguousarray(image),
        corruption_name=corruption,
        severity=int(severity),
    )
    return np.ascontiguousarray(corrupted, dtype=np.uint8)


def apply_cityscapes_c_corruption(image, corruption, severity, seed=None):
    """Compatibility wrapper for BCHW unit-range tensors."""
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("Expected one RGB image with shape [1, 3, H, W]")

    source = (
        image.detach()
        .squeeze(0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    corrupted = apply_cityscapes_c_corruption_uint8(
        source,
        corruption,
        severity,
        seed=seed,
    )
    output = torch.from_numpy(corrupted).permute(2, 0, 1)
    return output.to(device=image.device, dtype=image.dtype).div(255.0).unsqueeze(0)
