import torch
from torch import Tensor
from .convolution import conv, double_flip

@torch.compile
def convolution(kernel: Tensor, input: Tensor,) -> Tensor:
    return conv(kernel, input)

@torch.compile
def conjugate_convolution(kernel: Tensor, input) -> Tensor:
    return conv(double_flip(kernel), input)
