import torch
from torch import Tensor


def psnr(input: Tensor, enhanced: Tensor) -> Tensor:
    batch_size, channels = input.shape[:2]
    eps = 1e-5
    max = torch.max(input.reshape(batch_size, channels, - 1), dim=-1).values

    mse = torch.mean((input - enhanced) ** 2 + eps, dim=(-1, -2))

    return 10 * torch.mean(2 * max.log10() - mse.log10())

def psnr_minimizable(input: Tensor, enhanced: Tensor) -> Tensor:
    return 1 / psnr(input, enhanced)

def psnr_minimazible_calibrated(input: Tensor, enhanced: Tensor) -> Tensor:
    uieb_average = 0.05976884812116623
    return psnr_minimizable(input, enhanced) / uieb_average
