import torch
from torch import Tensor

def total_variation(input: Tensor) -> Tensor:
    return (
        torch.abs(input[..., 1:] - input[..., :-1]).mean()
        + torch.abs(input[..., 1:, :] - input[..., :-1, :]).mean()
    )

def total_variation_calibrated(input: Tensor) -> Tensor:
    uieb_average = 0.05879312381148338

    return input / uieb_average
