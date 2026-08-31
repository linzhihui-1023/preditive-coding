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


def apply_cityscapes_c_corruption(image, corruption, severity):
    """Apply an ImageNet-C common corruption to a BCHW unit-range RGB tensor."""
    if corruption not in CITYSCAPES_C_COMMON_CORRUPTIONS:
        raise ValueError(f"Unsupported Cityscapes-C corruption: {corruption}")
    if severity not in CITYSCAPES_C_SEVERITIES:
        raise ValueError(f"Severity must be one of {CITYSCAPES_C_SEVERITIES}")
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("Expected one RGB image with shape [1, 3, H, W]")

    from imagecorruptions import corrupt
    from imagecorruptions import corruptions as imagecorruptions_impl

    # imagecorruptions releases before the scikit-image channel_axis rename
    # still call gaussian(..., multichannel=True). Keep the published
    # corruption implementation and translate only that renamed keyword.
    gaussian = imagecorruptions_impl.gaussian
    if not getattr(gaussian, "_predify_channel_axis_compat", False):
        import inspect

        if "multichannel" not in inspect.signature(gaussian).parameters:
            from functools import wraps

            @wraps(gaussian)
            def gaussian_compat(*args, **kwargs):
                if kwargs.pop("multichannel", False):
                    kwargs["channel_axis"] = -1
                return gaussian(*args, **kwargs)

            gaussian_compat._predify_channel_axis_compat = True
            imagecorruptions_impl.gaussian = gaussian_compat

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
    corrupted = corrupt(
        np.ascontiguousarray(source),
        corruption_name=corruption,
        severity=int(severity),
    )
    output = torch.from_numpy(np.ascontiguousarray(corrupted)).permute(2, 0, 1)
    return output.to(device=image.device, dtype=image.dtype).div(255.0).unsqueeze(0)
