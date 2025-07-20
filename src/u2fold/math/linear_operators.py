import torch
from torch import Tensor
from .convolution import convolve, double_flip

@torch.compile
def convolution(input: Tensor, kernel: Tensor) -> Tensor:
    return convolve(kernel, input)

@torch.compile
def conjugate_convolution(input: Tensor, kernel: Tensor) -> Tensor:
    return convolve(double_flip(kernel), input)
