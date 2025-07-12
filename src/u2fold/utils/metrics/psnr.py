import torch
from torch import Tensor


def psnr(input: Tensor, enhanced: Tensor) -> Tensor:
    batch_size, channels = input.shape[:2]
    max = torch.max(input.reshape(batch_size, channels - 1), dim=-1).values

    mse = torch.mean((input - enhanced) ** 2, dim=(-1, -2))

    return 10 * torch.mean(2 * max.log10() - mse.log10())
