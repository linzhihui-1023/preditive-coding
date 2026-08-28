import torch
import torchvision.transforms.functional as TF


BLUR_KERNEL_SIZE = 11
BLUR_SIGMA_MAX = 3.0
BLUR_SIGMA_LEVELS = (0.75, 1.5, 2.25, 3.0)


def persistent_gaussian_blur(image, frame_index, total_frames, warmup_fraction=1 / 3):
    onset = int(total_frames * warmup_fraction)
    if frame_index < onset:
        return image
    sigma = BLUR_SIGMA_LEVELS[min(frame_index - onset, len(BLUR_SIGMA_LEVELS) - 1)]
    return TF.gaussian_blur(
        image,
        [BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE],
        [sigma, sigma],
    )
