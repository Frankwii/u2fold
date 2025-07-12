import colorspace_utilities
import torch
from torch import Tensor


def uciqe(input: Tensor) -> Tensor:
    batch_size = input.size(0)
    lab_input = colorspace_utilities.rgb_to_cielab(input)

    luminance_contrast = (
        torch.quantile(
            input=lab_input[:, 0, :, :].reshape(batch_size, -1),
            q=torch.Tensor([0.01, 0.99]),
            dim=-1,
        )
        .diff(dim=0)
        .reshape(batch_size)
    )

    chroma_std = (
        (lab_input[:, 1:, :, :] ** 2)
        .sum(dim=1)  # a**2 + b**2
        .sqrt()
        .std(dim=(-2, -1))
        .reshape_as(luminance_contrast)
    )

    saturation_mean = (
        (chroma_std / lab_input[:, :1, :, :])
        .squeeze(1)
        .mean(dim=(-2, -1))
        .reshape_as(luminance_contrast)
    )

    submetrics = torch.stack(
        (saturation_mean, luminance_contrast, chroma_std), dim=1
    )

    coefficients = torch.Tensor([0.468, 0.2745, 0.2576]).unsqueeze(0)

    return torch.sum(coefficients * submetrics, dim = 1).mean()
