import torch
from torch import Tensor

from u2fold.config_parsing.config_dataclasses import (
    TransmissionMapEstimationConfig,
)
from u2fold.math import background_light_estimation

from .background_light_estimation import estimate_background_light
from .transmission_map_estimation import estimate_transmission_map


@torch.compile
def estimate_radiance_and_transmission_map(
    images: Tensor,  # I; (B, C, H, W)
    background_light: Tensor,  # B; (B, C, 1, 1)
    config: TransmissionMapEstimationConfig,
) -> tuple[Tensor, Tensor]:  # (J_0, t); ((B, C, H, W), (B, 1, H, W))
    transmission_map_estimation = estimate_transmission_map(
        images,
        background_light,
        config.patch_radius,
        config.saturation_coefficient,
        config.regularization_coefficient,
    )  # t; (B, 1, H, W)

    scene_radiance_estimation = (images - background_light) / torch.max(
        transmission_map_estimation, torch.tensor(0.1)
    ) + (1 - background_light) * background_light  # J_0; (B, C, H, W)

    return scene_radiance_estimation, transmission_map_estimation


# batch_size = images.size(0)
#
# background_light = estimate_background_light(
#     images
# ).reshape(batch_size, -1, 1, 1)  # (B, C, 1, 1)


@torch.compile
def compute_I2(
    images: Tensor,  # I; (B, C, H, W)
    scene_radiance: Tensor,  # J; (B, C, H, W)
    transmission_map: Tensor,  # t; (B, 1, H, W)
    background_light: Tensor,  # B; (B, C, 1, 1)
) -> Tensor:  # I2; (B, C, H, W)
    return images - (
        scene_radiance * transmission_map
        + background_light * (1 - transmission_map)
    )
