import torch
from torch import Tensor


@torch.compile
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


@torch.compile
def gray_guided_filter(
    guide: Tensor, input: Tensor, patch_radius: int, regularization_coef: float
) -> Tensor:
    """Computes the (1D) guided filter for the given (batched) input and guide.

    As defined in https://doi.org/10.1109/TPAMI.2012.213, for the grayscale
    (1D) case.

    Args:
        guide: The batch of guided images. ("I" in the paper).
            Shape (B, 1, H, W)
        input: The batch of input images ("p" in the paper).
            Shape (B, 1, H, W)
        patch_radius: Number of pixels from the center of the patch (i.e.
            the filter) to its border, without counting the center itself.
            ("r" in the paper).
        regularization_coef: Coefficient of the L^2 regularization used
            for the input patch means ("\\varepsilon" in the paper).mean_filter(
        (guide - guide_patch_means) ** 2, patch_radius
    )

    Returns:
        output: The batch of output images ("q" in the paper).
            Shape (B, 1, H, W)
    """

    guide_patch_means = mean_filter(guide, patch_radius)  # "\mu" in the paper
    guide_patch_second_moments = mean_filter(guide ** 2, patch_radius)
    guide_patch_variances = (
        guide_patch_second_moments - guide_patch_means ** 2
    )  # "\sigma^2" in the paper

    input_patch_means = mean_filter(input, patch_radius) # \bar{p} in the paper

    product_patch_means = mean_filter(guide * input, patch_radius)

    input_coefficients = (
        product_patch_means - guide_patch_means * input_patch_means
    ) / (guide_patch_variances + regularization_coef)  # "a" in the paper

    independent_terms = (
        input_patch_means - input_coefficients * guide_patch_means
    )  # "b" in the paper

    smoothed_input_coefficients = mean_filter(input_coefficients, patch_radius)
    smoothed_independent_terms = mean_filter(independent_terms, patch_radius)

    return smoothed_input_coefficients * guide + smoothed_independent_terms
