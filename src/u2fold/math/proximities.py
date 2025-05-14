import torch


def shifted_square_L2_norm_conjugate(
    x: torch.Tensor, shift: torch.Tensor, tau: float
) -> torch.Tensor:
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
    return (x - tau * shift) / (tau + 1)
