import torch
from torch import Tensor

from .background_light_estimation import estimate_background_light
from .transmission_map_estimation import estimate_transmission_map


@torch.compile
def compute_I1(images: Tensor) -> Tensor:
    batch_size = images.size(0)

    background_light_estimations = estimate_background_light(images)  # (B, C)

    transmission_map_estimation = estimate_transmission_map(
        images, background_light_estimations
    )  # (B, 1, H, W)

    return images - (
        background_light_estimations.reshape(batch_size, -1, 1, 1) ** 2
        * transmission_map_estimation
    )

@torch.compile
def compute_I2(images: Tensor) -> Tensor:
    batch_size = images.size(0)
    background_light_estimations = estimate_background_light(images)  # (B, C)

    transmission_map_estimation = estimate_transmission_map(
        images, background_light_estimations
    )  # (B, 1, H, W)

    return (
        background_light_estimations.reshape(batch_size, -1, 1, 1) ** 2
        * transmission_map_estimation
    )
