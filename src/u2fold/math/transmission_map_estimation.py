import torch
from torch import Tensor

from u2fold.math.guided_filter import gray_guided_filter


@torch.compile
def compute_saturation_map(images: Tensor) -> Tensor:
    """Computes the saturation map for a batch of images.

    The saturation for a RGB image I of shape (3, H, W) is defined as
    \\[
        \\text{Sat}(x, y) = 1 -
        \\frac{\\text{min}_{c\\in \\{r, g, b\\}}I_c(x,y))}
        {\\text{max}_{c\\in \\{r, g, b\\}}I_c(x,y))}
    \\]

    Args:
        images: Shape (B, C, H, W).

    Returns:
        saturation_maps: Tensor of shape (B, 1, H, W) containing the saturation
            map as defined above (for C channels intead of 3; the generalization
            is immediate).
    """
    min_intensities = images.min(dim=1, keepdim=True).values
    max_instensities = images.max(dim=1, keepdim=True).values

    intensity_is_constant = max_instensities == min_intensities

    return torch.where(
        intensity_is_constant, 0.0, 1.0 - min_intensities / max_instensities
    )


@torch.compile
def estimate_coarse_transmission_map(
    images: Tensor,
    background_lights: Tensor,
    patch_radius: int,
    saturation_coef: float,
) -> Tensor:
    """Estimates the coarse red transmission map as in https://doi.org/10.1109/TCSVT.2021.

    Args:
        images: Shape (B, 3, H, W)
        background_lights: Shape (B, 3)
        saturation_coef: Constant (should be between 0 and 1,
            but this is not enforced) to weigh the saturation map with.
        patch_radius: Number of pixels from the center of the patch to its
            border, without counting the center itself.

            For example, a 3x3 patch should have a radius of 1; a 5x5 one, a
            radius of 2; a 7x7 one, 3, and so on.

            Note: this is geometrically closer to the apothem of the square than
            to its radius, but that name might confuse some.
    Returns:
        coarse_red_map: Shape (B, 1, H, W)
    """
    batch_size = images.size(0)

    saturation_map = compute_saturation_map(images)

    all_data = torch.cat((images, saturation_map), dim=1)  # (B, 4, H, W)

    all_data[:, 0, :, :] = 1 - all_data[:, 0, :, :]

    patch_minima = -torch.nn.functional.max_pool2d(
        -all_data,
        kernel_size=2 * patch_radius + 1,
        stride=1,
        padding=patch_radius,
    )  # (B, 4, H, W)

    background_lights[:, 0] = 1 - background_lights[:, 0]
    background_lights = torch.max(
        input=background_lights, other=torch.tensor(1e-3)
    )

    background_lights = 1 / background_lights

    coefficients = torch.cat(
        (
            background_lights,  # (B, 3)
            torch.tensor([saturation_coef])  # CPU
            .to(background_lights.device)  # GPU
            .view(1, 1)
            .expand(batch_size, 1),  # (B, 1)
        ),
        dim=1,  # (B, 4)
    ).view(batch_size, 4, 1, 1)  # (B, 4, 1, 1)

    overall_minima = torch.min(
        patch_minima * coefficients, dim=1, keepdim=True
    ).values  # (B, 1, H, W)

    return 1 - overall_minima


@torch.compile
def estimate_transmission_map(
    images: Tensor,
    background_light: Tensor,
    patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
) -> Tensor:
    """Estimates the fine red transmission map as in https://doi.org/10.1109/TCSVT.2021.

    Args:
        images: Shape (B, 3, H, W)
        background_lights: Shape (B, 3)
        patch_radius: Number of pixels from the center of the patch to its
            border, without counting the center itself.

            For example, a 3x3 patch should have a radius of 1; a 5x5 one, a
            radius of 2; a 7x7 one, 3, and so on.
        saturation_coef: Constant (should be between 0 and 1, but this is
           not enforced) to weigh the saturation map with.
        regularization_coef: Constant (should be positive, but this is not
            enforced) to be used as the coefficient of the L^2 regularization
            for the guided filter.
    Returns:
        fine_red_map: Shape (B, 1, H, W)
    """
    coarse_transmission_map = estimate_coarse_transmission_map(
        images=images,
        background_lights=background_light,
        saturation_coef=saturation_coef,
        patch_radius=patch_radius,
    )

    return gray_guided_filter(
        guide=images[:, 0, :, :],
        input=coarse_transmission_map,
        patch_radius=patch_radius,
        regularization_coef=regularization_coef,
    )
