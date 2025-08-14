from functools import partial
from typing import NamedTuple

import torch
from torch import Tensor, device

from u2fold.math.convolution import convolve


def _compute_centered_gaussian_kernel(
    standard_deviations: torch.Tensor,  # (B, C, 1, 1)
    kernel_size: int,
) -> torch.Tensor:
    center = kernel_size // 2

    x = (
        torch.arange(
            kernel_size,
            dtype=standard_deviations.dtype,
        )
        - center
    ).reshape(kernel_size, 1).to(standard_deviations.device)
    y = x.transpose(0, 1)

    distances_squared = (x**2 + y**2).unsqueeze(0).unsqueeze(0)

    exponentials = torch.exp(-distances_squared / (2 * standard_deviations**2))

    return exponentials / exponentials.sum(dim=(-1, -2), keepdim=True)


def _compute_data_range(input: Tensor, enhanced: Tensor) -> Tensor:
    batch_size, channels = input.shape[:2]
    unrolled_input = input.reshape(batch_size, channels, -1)
    unrolled_enhanced = enhanced.reshape(batch_size, channels, -1)

    return (
        torch.max(
            unrolled_input.max(dim=-1).values,
            unrolled_enhanced.max(dim=-1).values,
        )
        - torch.min(
            unrolled_input.min(dim=-1).values,
            unrolled_enhanced.min(dim=-1).values,
        )
    ).reshape(batch_size, channels, 1, 1)


def _get_gaussian_kernel(device: device) -> Tensor:
    return _compute_centered_gaussian_kernel(
        standard_deviations=torch.full((1, 1, 1, 1), 1.5).to(device), kernel_size=11
    )


class LocalMetrics(NamedTuple):
    patch_means: tuple[Tensor, Tensor]
    patch_variances: tuple[Tensor, Tensor]
    patch_covariance: Tensor


def _compute_local_metrics(input: Tensor, enhanced: Tensor) -> LocalMetrics:
    mean_filter = partial(convolve, kernel=_get_gaussian_kernel(input.device))

    input_patch_mean = mean_filter(input=input)
    enhanced_patch_mean = mean_filter(input=enhanced)

    input_patch_variance = mean_filter(input=input**2) - input_patch_mean**2
    enhanced_patch_variance = mean_filter(input=enhanced**2) - enhanced_patch_mean**2

    patch_covariance = (
        mean_filter(input=input * enhanced) - input_patch_mean * enhanced_patch_mean
    )

    return LocalMetrics(
        patch_means=(input_patch_mean, enhanced_patch_mean),
        patch_variances=(input_patch_variance, enhanced_patch_variance),
        patch_covariance=patch_covariance,
    )


def ssim(input: Tensor, enhanced: Tensor) -> Tensor:
    data_range = _compute_data_range(input, enhanced)
    c_1 = (0.01 * data_range) ** 2
    c_2 = (0.03 * data_range) ** 2

    local_metrics = _compute_local_metrics(input, enhanced)

    mu_x, mu_y = local_metrics.patch_means
    sigma_sq_x, sigma_sq_y = local_metrics.patch_variances

    sigma_xy = local_metrics.patch_covariance
    local_ssim = (
        (2 * mu_x * mu_y + c_1)
        * (2 * sigma_xy + c_2)
        / ((mu_x**2 + mu_y**2 + c_1) * (sigma_sq_x + sigma_sq_y + c_2))
    )

    return local_ssim.mean()


def dssim(input: Tensor, enhanced: Tensor) -> Tensor:
    return (1 - ssim(input, enhanced)) / 2

def dssim_calibrated(input: Tensor, enhanced: Tensor) -> Tensor:
    uieb_average = 0.11068128794431686
    return dssim(input, enhanced) / uieb_average
