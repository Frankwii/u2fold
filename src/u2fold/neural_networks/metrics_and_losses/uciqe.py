from .colorspace_utilities import rgb_to_lab
import torch
from torch import Tensor

# As defined in https://doi.org/10.1109/TIP.2015.2491020
def uciqe(input: Tensor) -> Tensor:
    batch_size = input.size(0)
    lab_input = rgb_to_lab(input)

    luminance_contrast = (
        torch.quantile(
            input=lab_input[:, 0, :, :].reshape(batch_size, -1),
            q=torch.Tensor([0.01, 0.99]).to(input.device),
            dim=-1,
        )
        .diff(dim=0)
        .reshape(batch_size, 1, 1, 1)
    )

    # sqrt(a ** 2 + b ** 2)
    chroma = (lab_input[:, 1:, :, :] ** 2).sum(dim=1).sqrt()

    chroma_std = chroma.std(dim=(-2, -1,)).reshape_as(luminance_contrast)

    saturation_mean = (
        (chroma / lab_input[:, :1, :, :].clamp(0.1))
        .mean(dim=(-3, -2, -1))
        .reshape_as(luminance_contrast)
    )

    submetrics = torch.stack(
        (chroma_std, luminance_contrast, saturation_mean), dim=1
    )

    coefficients = torch.Tensor([0.468, 0.2745, 0.2576]).reshape(1, 3, 1, 1, 1).to(input.device)

    return torch.sum(coefficients * submetrics, dim = 1).mean()

def uciqe_minimizable(input: Tensor) -> Tensor:
    """One over the UCIQE metric for the input"""
    return 1 / uciqe(input).clamp(0.1)

def uciqe_minimizable_calibrated(input: Tensor) -> Tensor:
    uieb_average = 0.03628702089190483

    return uciqe_minimizable(input) / uieb_average
