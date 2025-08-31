import torch
from torch import Tensor

@torch.compile
def total_variation(input: Tensor) -> Tensor:
    return (
        torch.abs(input[..., 1:] - input[..., :-1]).mean()
        + torch.abs(input[..., 1:, :] - input[..., :-1, :]).mean()
    )
