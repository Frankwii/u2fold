from typing import cast

import torch
from torch import Tensor


def __assign_scores(patch: Tensor) -> Tensor:
    """Returns the average minus standard deviation along the image pixels.

    Args:
        patch: Shape (*dims, C, H, W).

    Returns
        scores: Shape (*dims,)
    """
    avg, std = torch.std_mean(patch, dim=(-3, -2, -1))

    return avg - std


def __linear_search_background_light(
    patch: Tensor,
) -> Tensor:
    square_distances_to_white = torch.sum((1 - patch) ** 2, dim=-3)

    maximal_pixel_coordinates = torch.unravel_index(
        square_distances_to_white.argmax(), patch.shape[-2:]
    )

    return patch[..., :, *maximal_pixel_coordinates]


def __estimate_background_light_switching_dimensions(
    image: Tensor, split_horizontal: bool
) -> tuple[float, float, float]:
    if image.numel() <= 48:  # 48=3*4*4; 4x4 patch is the largest allowed
        background_light = __linear_search_background_light(image)

        return cast(
            tuple[float, float, float], tuple(background_light.flatten())
        )

    split_dimension = -(1 + split_horizontal)
    dimension_size = image.size(split_dimension)
    half_size = dimension_size // 2
    split_sizes = [half_size, dimension_size - half_size]

    half1, half2 = torch.split(image, split_sizes, dim=split_dimension)

    if __assign_scores(half1) < __assign_scores(half2):
        return __estimate_background_light_switching_dimensions(
            half2, not split_horizontal
        )

    return __estimate_background_light_switching_dimensions(
        half1, not split_horizontal
    )


def estimate_background_light_switching_dimensions(
    image: Tensor,
) -> tuple[float, float, float]:
    return __estimate_background_light_switching_dimensions(image, False)


def __MAX_ALLOWED_SIZE(channels: int) -> int:
    return channels * 16

def __split_image_batch_into_quadrants(
    images: Tensor, # Shape(B, C, H, W)
    height: int, # = H
    width: int # = W
) -> Tensor: # Shape (4, B, C, H//2, W//2)

    if height % 2 == 0 and width % 2 == 0:
        return (
            images.reshape(
                -1,
                -1,
                2,
                height // 2,
                2,
                width // 2
            ) # (B, C, 2, H//2, 2, W//2)
            .permute(2, 4, 0, 1, 3, 5) # (2, 2, B, C, H//2, W//2)
            .reshape(4, -1, -1, -1, -1) # (4, B, C, H//2, W//2)
        )

    else:
        half_height = height // 2
        half_width = width // 2

        left, right, *_ = torch.split(images, half_width, dim=-1)
        upper_left, lower_left, *_ = torch.split(left, half_height, dim=-2)
        upper_right, lower_right, *_ = torch.split(right, half_height, dim=-2)

        # (4, B, C, H//2, W//2)
        return torch.stack(
            [upper_right, upper_left, lower_left, lower_right]
        )

def __batched_estimate_background_light_by_squares(
    images: Tensor,  # Shape (B, C, H, W)
) -> Tensor:  # Shape (B, C)
    batch_size, channels, height, width = images.shape

    if images.numel() <= batch_size * __MAX_ALLOWED_SIZE(channels):
        return __linear_search_background_light(images)

    quadrants = __split_image_batch_into_quadrants(images, height, width)

    # (4, B)
    scores = __assign_scores(quadrants)

    best_score_indices = (
        scores.argmax(dim=0)  # (B,)
        .view(1, batch_size, 1, 1, 1)  # (1, B, 1, 1, 1)
        .expand(1, batch_size, channels, height, width)  # (1, B, C, H//2, W//2)
    )

    best_sections = torch.gather(
        quadrants, dim=0, index=best_score_indices
    ).squeeze(0)  # (B, C, H//2, W//2)

    return __batched_estimate_background_light_by_squares(best_sections)


def estimate_background_light_by_squares(
    images: Tensor,
) -> Tensor:
    """Estimate background light as in https://doi.org/10.1109/TCSVT.2021.3115791.

    Args:
        image: Tensor of shape (B, C, H, W).

    Returns
        background_light: Tensor of shape (B, C).
    """
    return __batched_estimate_background_light_by_squares(images)
