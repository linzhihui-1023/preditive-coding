import torch
import torchvision.transforms.functional as TF


BLUR_KERNEL_SIZE = 11
BLUR_SIGMA_MAX = 3.0
BLUR_SIGMA_LEVELS = (0.75, 1.5, 2.25, 3.0)
BLUR_WARMUP_FRACTION = 0.10
IMAGENET_C_GAUSSIAN_BLUR_SIGMAS = (1.0, 2.0, 3.0, 4.0, 6.0)


def warmup_frame_count(total_frames):
    return total_frames // 10


def persistent_gaussian_blur(image, frame_index, total_frames):
    onset = warmup_frame_count(total_frames)
    if frame_index < onset:
        return image
    sigma = BLUR_SIGMA_LEVELS[min(frame_index - onset, len(BLUR_SIGMA_LEVELS) - 1)]
    return TF.gaussian_blur(
        image,
        [BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE],
        [sigma, sigma],
    )


def imagenet_c_gaussian_blur(image, sigma):
    sigma = float(sigma)
    kernel_size = 2 * int(4.0 * sigma + 0.999999) + 1
    return TF.gaussian_blur(
        image,
        [kernel_size, kernel_size],
        [sigma, sigma],
    )
