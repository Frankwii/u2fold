import torch
from torch import Tensor

from .transmission_map_estimation import estimate_transmission_map


@torch.compile
def estimate_fidelity_and_transmission_map(
    images: Tensor,  # I; (B, C, H, W)
    background_light: Tensor,  # B; (B, C, 1, 1)
    patch_radius: int,
    saturation_coefficient: float,
    regularization_coefficient: float,
) -> tuple[Tensor, Tensor]:  # (J_0, t); ((B, C, H, W), (B, 1, H, W))
    transmission_map_estimation = estimate_transmission_map(
        images,
        background_light,
        patch_radius,
        saturation_coefficient,
        regularization_coefficient,
    )  # t; (B, 1, H, W)

    fidelity = images - (1 - transmission_map_estimation) * background_light

    return fidelity, transmission_map_estimation


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
