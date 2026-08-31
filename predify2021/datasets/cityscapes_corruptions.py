"""Online access to the published ImageNet-C-derived Cityscapes transforms."""

import json
from pathlib import Path

import numpy as np


PROTOCOL_PATH = Path(__file__).with_name("cityscapes_corruption_protocol.json")
PUBLISHED_IMAGE_CORRUPTIONS = (
    "motion_blur",
    "defocus_blur",
    "glass_blur",
    "gaussian_blur",
    "gaussian_noise",
    "impulse_noise",
    "shot_noise",
    "speckle_noise",
    "snow",
    "spatter",
    "fog",
    "frost",
    "brightness",
    "contrast",
    "saturate",
    "jpeg_compression",
)


def _imagecorruptions_corrupt(image, corruption_name, severity):
    import imagecorruptions.corruptions as implementation

    # imagecorruptions 1.1.2 uses the removed skimage `multichannel` keyword.
    from skimage.filters import gaussian as skimage_gaussian

    def gaussian_compat(array, sigma, multichannel=False, **kwargs):
        channel_axis = -1 if multichannel else None
        return skimage_gaussian(array, sigma=sigma, channel_axis=channel_axis, **kwargs)

    implementation.gaussian = gaussian_compat
    from imagecorruptions import corrupt

    return corrupt(image, corruption_name=corruption_name, severity=severity)


def apply_published_corruption(image, corruption_name, severity):
    """Apply one published transform online; return uint8 RGB with unchanged size."""
    if corruption_name not in PUBLISHED_IMAGE_CORRUPTIONS:
        raise ValueError(
            f"{corruption_name!r} is not one of the 16 executable ImageNet-C-derived transforms"
        )
    if int(severity) not in range(1, 6):
        raise ValueError("severity must be an integer in [1, 5]")
    image = np.asarray(image, dtype=np.uint8)
    return np.asarray(
        _imagecorruptions_corrupt(image, corruption_name, int(severity)),
        dtype=np.uint8,
    )


def load_published_protocol():
    return json.loads(PROTOCOL_PATH.read_text())
