import math
from enum import Enum, member
from functools import partial
from typing import NamedTuple, Optional

import torch


class PaddingStrategy(Enum):
    Mirror = member(partial(torch.nn.functional.pad, mode="reflect"))
    Zero = member(partial(torch.nn.functional.pad, mode="constant", value=0))


def convolve(
    kernel: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor:
    """Batched 2D channel-wise convolution. Respects size of "input".

    Each channel of each image in the batch is convolved with the
    corresponding channel of the corresponding kernel. This preserves the
    number of channels in the output.

    Broadcasting is supported for the kernel's batch and channel dimensions.
    For example, a single-channel (grayscale) kernel can be broadcast to
    convolve every channel of a multi-channel (RGB) image. Similarly, and
    compatibly with the channel broadcasting, a (single or multi-channelled)
    kernel can be broadcast to convolve every image of a batched input.

    The input is padded via mirroring in order to maintain shapes.

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

    return flexible_conv(
        kernel=kernel,
        input=input,
        output_shape=input.shape,
        padding_strategy=PaddingStrategy.Mirror,
    )


class CenteredDimensions(NamedTuple):
    down: int
    up: int
    left: int
    right: int

    def to_height_and_width(self) -> tuple[int, int]:
        return (self.up - self.down + 1, self.right - self.left + 1)

    @staticmethod
    def from_height_width(height: int, width: int) -> "CenteredDimensions":
        half_height = (height - 1) // 2
        half_width = (width - 1) // 2
        return CenteredDimensions(
            down=-(height - 1 - half_height),
            up=half_height,
            left=-(width - 1 - half_width),
            right=half_width,
        )

    def to_tensor_coordinates(
        self, height: int, width: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        half_height = math.ceil(height / 2)
        half_width = math.ceil(width / 2)

        return (
            (
                half_height + self.down,
                half_height + self.up + 1,
            ),
            (
                half_width + self.left,
                half_width + self.right + 1,
            ),
        )


def _crop_input(
    reshaped_input: torch.Tensor,
    kernel_dims: tuple[int, int],
    input_dims: tuple[int, int],
    output_dims: tuple[int, int],
) -> torch.Tensor:
    kernel_cdims = CenteredDimensions.from_height_width(*kernel_dims)
    output_cdims = CenteredDimensions.from_height_width(*output_dims)

    cropped_input_up = output_cdims.up - kernel_cdims.down
    cropped_input_down = output_cdims.down - kernel_cdims.up
    cropped_input_left = output_cdims.left - kernel_cdims.right
    cropped_input_right = output_cdims.right - kernel_cdims.left

    cropped_input_cdims = CenteredDimensions(
        down=cropped_input_down,
        up=cropped_input_up,
        left=cropped_input_left,
        right=cropped_input_right,
    )

    ((d, u), (l, r)) = cropped_input_cdims.to_tensor_coordinates(*input_dims)

    return reshaped_input[..., d:u, l:r]


def _pad_input(
    reshaped_input: torch.Tensor,
    kernel_dims: tuple[int, int],
    input_dims: tuple[int, int],
    output_dims: tuple[int, int],
    padding_strategy: PaddingStrategy,
) -> torch.Tensor:
    kernel_cdims = CenteredDimensions.from_height_width(*kernel_dims)
    input_cdims = CenteredDimensions.from_height_width(*input_dims)
    output_cdims = CenteredDimensions.from_height_width(*output_dims)

    down_pad = input_cdims.down - output_cdims.down + kernel_cdims.up
    up_pad = output_cdims.up - input_cdims.up - kernel_cdims.down
    left_pad = input_cdims.left - output_cdims.left + kernel_cdims.right
    right_pad = output_cdims.right - input_cdims.right - kernel_cdims.left

    padding = (left_pad, right_pad, up_pad, down_pad)

    return padding_strategy.value(reshaped_input, pad=(*padding,))


def flexible_conv(
    kernel: torch.Tensor,
    input: torch.Tensor,
    output_shape: tuple[int, ...] | torch.Size,
    padding_strategy: Optional[PaddingStrategy],
) -> torch.Tensor:
    """Batched 2D channel-wise convolution. Computes padding automatically.

    Each channel of each image in the batch is convolved with the
    corresponding channel of the corresponding kernel. This preserves the
    number of channels in the output.

    Broadcasting is supported for the kernel's batch and channel dimensions.
    For example, a single-channel (grayscale) kernel can be broadcast to
    convolve every channel of a multi-channel (RGB) image. Similarly, and
    compatibly with the channel broadcasting, a (single or multi-channelled)
    kernel can be broadcast to convolve every image of a batched input.

    Padding added to the image is computed, or the result is cropped in the
    center, so that the output has the specified spatial dimensions in
    "output_shape".

    Args:
        kernel: A tensor storing either a single kernel or a batch of kernels
            Shape: (B_, C_, H', W'), where "B_" is either "B" or 1,
        input: A tensor storing a batch of images.
            Shape: (B, C, H, W)
        output_shape: The desired spatial dimensions for the output.
            Shape: (H'', W'')
    Returns:
        \\(input\\ast kernel\\)
            Shape: (B, C, H'', W'')

    NOTE: Actually, pytorch implements cross correlation and not convolution, so
    symmetry should be assumed on either tensor for this to be an actual
    convolution; or, by always using this method, variables will be flipped.
    """
    kernel_height, kernel_width = kernel.shape[-2:]
    batch_size, n_channels, input_height, input_width = input.shape
    *_, output_height, output_width = output_shape

    reshaped_input = input.reshape(
        1, batch_size * n_channels, input_height, input_width
    )
    reshaped_kernel = kernel.expand(
        batch_size, n_channels, kernel_height, kernel_width
    ).reshape(-1, 1, kernel_height, kernel_width)

    if (
        output_height > input_height - kernel_height
        and output_width > output_height - kernel_height
    ):
        assert padding_strategy is not None
        reshaped_input = _pad_input(
            reshaped_input=reshaped_input,
            kernel_dims=(kernel_height, kernel_width),
            input_dims=(input_height, input_width),
            output_dims=(output_height, output_width),
            padding_strategy=padding_strategy,
        )
    elif (
        output_height <= input_height - kernel_height
        and output_width <= input_width - kernel_width
    ):
        reshaped_input = _crop_input(
            reshaped_input=reshaped_input,
            kernel_dims=(kernel_height, kernel_width),
            input_dims=(input_height, input_width),
            output_dims=(output_height, output_width),
        )
    else:
        raise ValueError("Invalid dimensions!")

    return torch.nn.functional.conv2d(
        input=reshaped_input,
        weight=reshaped_kernel,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=batch_size * n_channels,
    ).reshape(batch_size, n_channels, output_height, output_width)


def initialize_dirac_delta(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
):
    up = height // 2
    down = up - ((height + 1) % 2)
    up += 1

    right = width // 2
    left = right - ((width + 1) % 2)
    right += 1

    n_elements = (up - down) * (right - left)

    delta = torch.zeros(batch_size, channels, height, width)

    delta[:, :, down:up, left:right] = 1 / n_elements

    return delta


def double_flip(x: torch.Tensor) -> torch.Tensor:
    return x.fliplr().flipud()
