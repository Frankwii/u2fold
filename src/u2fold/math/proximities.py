from typing import Any, Callable
import torch
from torch import Tensor
from torch._dynamo.testing import CompileCounter

type ProximityOperator[A] = Callable[[Tensor, float, A], Tensor]
type LinearOperator[B] = Callable[[Tensor, B], Tensor]

@torch.compile
def conjugate_shifted_square_L2_norm(
    input: Tensor, step_size: float, shift: Tensor
) -> Tensor:
    """
    Computes the proximity operator of the square L2 norm evaluated in x

    More concretely, for

    \\[
        F(x) = \\|x - shift\\|_2^2
    ,\\]

    this function returns \\(prox_{\\tauF^*}(x)\\), where \\(F^*\\) is the
    Fenchel conjugate (a.k.a. convex conjugate) of F.

    Args:
        x: The input image.
            Shape: (batch_size, channels, height, width)
        shift: The "shifting" inside of the norm.
            Shape: (batch_size, channels, height, width)
        tau: The size step.

    Returns:
        \\(prox_{\\tauF^*}(x)\\)
            Shape: (batch_size, channels, height, width)
    """
    return (input - step_size * shift) / (step_size + 1)

@torch.compile
def shifted_square_L2_norm(
    input: Tensor, step_size: float, shift: Tensor
) -> Tensor:
    return (step_size / (1 + step_size)) * (input + shift) 


@torch.compile
def identity(
    input: Tensor, *_
) -> Tensor:
    return input
