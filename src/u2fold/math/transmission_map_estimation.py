import torch
from torch import Tensor

from u2fold.math.guided_filter import guided_filter


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


def estimate_coarse_red_transmission_map(
    images: Tensor,
    background_lights: Tensor,
    patch_radius: int,
    saturation_coefficient: float,
) -> Tensor:
    """Estimates the coarse red transmission map as in https://doi.org/10.1109/TCSVT.2021.

    Args:
        images: Shape (B, 3, H, W)
        background_lights: Shape (B, 3, 1, 1)
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

    copy_background_lights = background_lights.detach().clone()
    copy_background_lights[:, 0, 0, 0] = 1 - background_lights[:, 0, 0, 0]
    copy_background_lights = 1 / copy_background_lights.clamp(min=1e-3)

    coefficients = torch.cat(
        (
            copy_background_lights,  # (B, 3, 1, 1)
            torch.tensor([saturation_coefficient])  # CPU
            .to(copy_background_lights.device)  # GPU
            .view(1, 1, 1, 1)
            .expand(batch_size, 1, 1, 1),  # (B, 1, 1, 1)
        ),
        dim=1,  # (B, 4)
    ).view(batch_size, 4, 1, 1)  # (B, 4, 1, 1)

    overall_minima = torch.min(
        patch_minima * coefficients, dim=1, keepdim=True
    ).values  # (B, 1, H, W)

    return 1 - overall_minima


def estimate_transmission_map(
    images: Tensor,
    background_light: Tensor,
    guided_filter_patch_radius: int,
    transmission_map_patch_radius: int,
    saturation_coefficient: float,
    regularization_coefficient: float,
) -> Tensor:
    """Estimates the fine red transmission map as in https://doi.org/10.1109/TCSVT.2021 and the other two channels as in https://doi.org/10.1109/TIP.2016.2612882.

    Args:
        images: Shape (B, 3, H, W)
        background_lights: Shape (B, 3, 1, 1)
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
        fine_transmission_map: Shape (B, 3, H, W)
    """
    coarse_red_transmission_map = estimate_coarse_red_transmission_map(
        images=images,
        background_lights=background_light,
        saturation_coefficient=saturation_coefficient,
        patch_radius=transmission_map_patch_radius,
    )

    fine_red_transmission_map = guided_filter(
        guide=images,
        input=coarse_red_transmission_map,
        patch_radius=guided_filter_patch_radius,
        regularization_coefficient=regularization_coefficient,
    )


    # As specified in https://doi.org/10.1109/TIP.2016.2612882
    wavelength_coefficient = -0.00113
    wavelength_bias = 1.62517
    ## in nanometers, for red, green, blue
    channel_wavelengths = torch.Tensor([620, 540, 450]).reshape(1, 3, 1, 1)

    coefficients = (
        wavelength_coefficient * channel_wavelengths + wavelength_bias
    ).to(background_light.device)

    divisions = coefficients / (background_light + 1e-4)

    exponents = divisions / divisions[:, 0:1, :, :]

    return torch.pow(
        input=fine_red_transmission_map.clamp(0.1),
        exponent=exponents
    )


