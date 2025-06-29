import torch


def conv(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Batched 2D channel-wise convolution. Respects size of "input".

    Each channel of each image in the batch is convolved with the
    corresponding channel of the corresponding kernel. This preserves the
    number of channels in the output.

    Broadcasting is supported for the kernel's batch and channel dimensions.
    For example, a single-channel (grayscale) kernel can be broadcast to
    convolve every channel of a multi-channel (RGB) image. Similarly, and
    compatibly with the channel broadcasting, a (single or multi-channelled)
    kernel can be broadcast to convolve every image of a batched input.

    NOTE: Actually, pytorch implements cross correlation and not convolution, so
    symmetry should be assumed on \\(g\\) for this to be an actual convolution;
    or, by always using this method, variables will be flipped.

    Args:
        input: A tensor storing a batch of images.
            Shape: (B, C, H, W)
        kernel: A tensor storing either a single kernel or a batch of kernels
            Shape: (B_, C_, H', W'), where "B_" is either "B" or 1,
            and "C_" is either "C" or 1.

    Returns:
        \\(input\\ast kernel\\)
            Shape: (B, C, H, W)
    """

    batch_size, n_channels, height, width = input.shape
    kernel_height, kernel_width = kernel.shape[-2:]

    return torch.nn.functional.conv2d(
        input=input.reshape(1, batch_size * n_channels, height, width),
        weight=kernel.expand(
            batch_size, n_channels, kernel_height, kernel_width
        ).reshape(-1, 1, kernel_height, kernel_width),
        bias=None,
        stride=1,
        padding="same",
        dilation=1,
        groups=batch_size * n_channels,
    ).reshape(batch_size, n_channels, height, width)


def double_flip(x: torch.Tensor) -> torch.Tensor:
    return x.fliplr().flipud()


def convex_conjugate(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return conv(x, double_flip(y))
