import torch
from torch import Tensor
from .convolution import conv, double_flip

@torch.compile
def convolution(input: Tensor, kernel: Tensor) -> Tensor:
    return conv(kernel, input)

@torch.compile
def conjugate_convolution(input: Tensor, kernel: Tensor) -> Tensor:
    return conv(double_flip(kernel), input)
