from functools import partial

import torch
from torch import Tensor


def mean_filter(image: Tensor, patch_radius: int) -> Tensor:
    """Computes a shape-preserving mean filter with a square window for a batch
    of images.

    Args:
        image: The image whom the mean filter is to be applied to.
            Shape (B, C, H, W)
        patch_radius: Number of pixels from the center of the patch (i.e.
            the filter) to its border, without counting the center itself.

    Returns:
        filtered_image: The image after passing the filter through it.
            Shape (B, C, H, W)
    """
    return torch.nn.functional.avg_pool2d(
        image,
        kernel_size=2 * patch_radius + 1,
        stride=1,
        padding=patch_radius,
        count_include_pad=False,  # borders would be darker otherwise
    )


def cross_multiply_channels(
    a: Tensor,  # (B, C, H, W)
    b: Tensor,  # (B, D, H, W)
) -> Tensor:  # (B, C, D, H, W)
    return a.unsqueeze(2) * b.unsqueeze(1)


def identity(C: int) -> Tensor:  # (1, C, C, 1, 1)
    return torch.eye(C).reshape(1, C, C, 1, 1)


def guided_filter(
    guide: Tensor,
    input: Tensor,
    patch_radius: int,
    regularization_coefficient: float,
) -> Tensor:
    """Computes the guided filter for the given (batched) input and guide.

    As defined in https://doi.org/10.1109/TPAMI.2012.213.

    Args:
        guide: The batch of guided images. ("I" in the paper).
            Shape (B, C, H, W)
        input: The batch of input images ("p" in the paper).
            Shape (B, D, H, W)
        patch_radius: Number of pixels from the center of the patch (i.e.
            the filter) to its border, without counting the center itself.
            ("r" in the paper).
        regularization_coef: Coefficient of the L^2 regularization used
            for the input patch means ("\\varepsilon" in the paper).mean_filter(
        (guide - guide_patch_means) ** 2, patch_radius
    )

    Returns:
        output: The batch of output images ("q" in the paper).
            Shape (B, D, H, W)
    """

    B, C, H, W = guide.shape
    _, D, _, _ = input.shape

    mf = partial(mean_filter, patch_radius=patch_radius)
    meanI = mf(guide)  # (B, C, H, W)
    meanp = mf(input)  # (B, D, H, W)

    corrI = mf(
        cross_multiply_channels(guide, guide).reshape(B, -1, H, W)
    ).reshape(B, C, C, H, W)
    corrIp = mf(
        cross_multiply_channels(guide, input).reshape(B, -1, H, W)
    ).reshape(B, C, D, H, W)

    covI = corrI - cross_multiply_channels(meanI, meanI)  # (B, C, C, H, W)
    covIp = corrIp - cross_multiply_channels(meanI, meanp)  # (B, C, D, H, W)

    id_matrix = identity(C).to(input.device)
    a: Tensor = (
        torch.linalg.solve(
            (
                (covI + regularization_coefficient * id_matrix).permute(
                    0, 3, 4, 1, 2
                )  # (B, H, W, C, C)
            ),
            covIp.permute(0, 3, 4, 1, 2),  # (B, H, W, C, D)
        ).permute(0, 3, 4, 1, 2)  # (B, H, W, C, D)  # (B, C, D, H, W)
    )

    b = meanp - (a * meanI.unsqueeze(2)).sum(dim=1)

    a_smoothed = mf(a.reshape(B, -1, H, W)).reshape(B, C, D, H, W)
    b_smoothed = mf(b)

    q = (a_smoothed * guide.unsqueeze(2)).sum(dim=1) + b_smoothed

    return q
