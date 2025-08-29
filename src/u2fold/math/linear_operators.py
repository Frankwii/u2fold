from torch import Tensor
from .convolution import convolve, double_flip

def convolution(input: Tensor, kernel: Tensor) -> Tensor:
    return convolve(kernel, input)

def conjugate_convolution(input: Tensor, kernel: Tensor) -> Tensor:
    return convolve(double_flip(kernel), input)
