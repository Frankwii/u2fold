import torch
from torch import Tensor
from .convolution import conv, double_flip

@torch.compile
def convolution(input: Tensor, kernel: Tensor) -> Tensor:
    return conv(input, kernel)

@torch.compile
def conjugate_convolution(input: Tensor, kernel: Tensor) -> Tensor:
    return conv(input, double_flip(kernel))
