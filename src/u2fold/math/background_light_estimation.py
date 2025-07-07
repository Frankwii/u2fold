import torch
from torch import Tensor


@torch.compile
def _assign_scores(patches: Tensor) -> Tensor:
    """Returns the average minus standard deviation along the image pixels.

    Args:
        patches: Shape (*dims, C, H, W).

    Returns:
        scores: Shape (*dims,).
    """
    std, avg = torch.std_mean(patches, dim=(-2, -1))

    return (avg - std).sum(dim=-1)


@torch.compile
def _linear_search_background_light(
    patches: Tensor,
) -> Tensor:
    """Returns the value of the pixels closest to white.

    The pixel "closest to white" is selected, meaning that the euclidean
    distance between the vector formed by the pixel intensities across the C
    channels of the image and the vector formed by C ones is minimal among
    the set of pixels of the image.

    This operation is performed separately for each patch in the batch.

    Args:
        patches: Shape (B, C, H, W).

    Returns:
        pixels: Shape (B, C, 1, 1).
    """
    batch_size, channels, height, width = patches.shape

    # (B, H, W)
    square_distances_to_white = torch.sum((patches - 1) ** 2, dim=-3)

    ravelled_closest_pixels_to_white = (
        torch.argmin(
            input=square_distances_to_white.reshape(batch_size, -1),  # (B, H*W)
            dim=1,
        )  # (B,)
        .reshape(batch_size, 1, 1)  # (B, 1, 1)
        .expand(batch_size, channels, 1)  # (B, C, 1)
    )

    flattened_patches = patches.reshape(
        -1, channels, height * width
    )  # (B, C, H*W)

    closest_to_white_values = torch.gather(
        flattened_patches, dim=2, index=ravelled_closest_pixels_to_white
    )  # (B, C, 1)

    return closest_to_white_values.reshape(
        batch_size, channels, 1, 1
    )  # (B, C, 1, 1)


@torch.compile
def _MAX_ALLOWED_SIZE(batch_size: int, channels: int) -> int:
    return batch_size * channels * 16  # 4 x 4 patches are the biggest allowed.


@torch.compile
def _split_image_batch_into_quadrants(
    images: Tensor,  # Shape(B, C, H, W)
    channels: int,
    height: int,  # := H
    width: int,  # := W
) -> Tensor:  # Shape (4, B, C, H//2, W//2)
    half_height = height // 2
    half_width = width // 2

    if height % 2 == 0 and width % 2 == 0:
        return (
            images.reshape(
                -1, channels, 2, half_height, 2, half_width
            )  # (B, C, 2, H//2, 2, W//2)
            .permute(2, 4, 0, 1, 3, 5)  # (2, 2, B, C, H//2, W//2)
            .reshape(
                4, -1, channels, half_height, half_width
            )  # (4, B, C, H//2, W//2)
        )
    else:
        left, right, *_ = images.split(half_width, dim=-1)
        upper_left, lower_left, *_ = left.split(half_height, dim=-2)
        upper_right, lower_right, *_ = right.split(half_height, dim=-2)

        # (4, B, C, H//2, W//2)
        return torch.stack(
            [upper_left, upper_right, lower_left, lower_right]
        )


@torch.compile
def estimate_background_light(
    images: Tensor,
) -> Tensor:
    """Estimate background light as in https://doi.org/10.1109/TCSVT.2021.3115791.

    Args:
        image: Tensor of shape (B, C, H, W).

    Returns
        background_light: Tensor of shape (B, C, 1, 1).
    """

    batch_size, channels, height, width = images.shape

    img = images.detach().clone()

    while img.numel() > _MAX_ALLOWED_SIZE(batch_size, channels):
        # images: Shape (B, C, H, W)
        quadrants = _split_image_batch_into_quadrants(
            img, channels, height, width
        )

        # (4, B)
        scores = _assign_scores(quadrants)

        height //= 2
        width //= 2

        best_score_indices = (
            scores.argmax(dim=0)  # (B,)
            .reshape(1, batch_size, 1, 1, 1)  # (1, B, 1, 1, 1)
            .expand(
                1, batch_size, channels, height, width
            )  # (1, B, C, H//2, W//2)
        )

        img = torch.gather(
            quadrants, dim=0, index=best_score_indices
        ).squeeze(0)  # (B, C, H//2, W//2)
        # Update notation: H = H //2, W = W // 2

    return _linear_search_background_light(img)
