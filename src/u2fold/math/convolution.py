import torch


def conv(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    "Regular" convolution between a batch of images and a fixed kernel (the same accross images), all represented as tensors.

    Actually, pytorch implements cross correlation and not convolution, so
    symmetry should be assumed on \\(g\\) for this to be an actual convolution.

    Args:
        f: A tensor storing a batch of images.
            Shape: (batch_size, channels, height, width)
        g: A tensor storing either a single kernel or a batch of kernels
            Shape: (channels, kernel_height, kernel_width)

    Returns:
        \\(f\\ast g\\)
            Shape: (batch_size, channels, height, width)
    """

    n_channels = f.shape[1]
    return torch.nn.functional.conv2d(
        input=f,
        weight=g.unsqueeze(1),
        bias=None,
        stride=1,
        padding="same",
        dilation=1,
        groups=n_channels,
    )
