import torch
from torch import Tensor

def mse(input: Tensor, ground_truth: Tensor) -> Tensor:
    return torch.mean((input - ground_truth) ** 2)
