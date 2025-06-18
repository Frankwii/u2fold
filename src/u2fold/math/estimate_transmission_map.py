import torch
from torch import Tensor


def compute_saturation_map(images: Tensor) -> Tensor:
    """Computes the saturation map for a batch of images.

    The saturation for a RGB image I of shape (3, H, W) is defined as
    \\[
        \\text{Sat}(x, y) = 1 -
        \\frac{\\text{min}_{c\\in \\{r, g, b\\}}I_c(x,y))}
        {\\text{max}_{c\\in \\{r, g, b\\}}I_c(x,y))}
    \\]

    Args:
        images: Shape (B, 3, H, W)

    Returns:
        saturation_maps: Tensor of shape (B, 1, H, W) containing
            saturation as defined above.
    """
    B, _, H, W = images.shape
    min_intensities = images.min(dim=1).values
    max_instensities = images.max(dim=1).values

    intensity_is_constant = max_instensities == min_intensities

    return torch.where(
        intensity_is_constant, 0.0, 1.0 - min_intensities / max_instensities
    ).reshape(B, 1, H, W)

def estimate_transmission_map(
    image: Tensor, background_light: tuple[float, float, float]
) -> Tensor: ...
