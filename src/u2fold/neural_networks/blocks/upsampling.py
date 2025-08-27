from typing import Literal, final, override
import torch


@final
class UpsamplingLayer(torch.nn.Module):
    """
    Deterministic upsampling, followed by a convolution with a
    3x3 kernel with a stride of 1.
    """

    def __init__(
        self,
        interpolation: Literal["bilinear", "nearest"],
        in_channels: int,
        out_channels: int,
        device: str | None
    ) -> None:
        torch.nn.Module.__init__(self)  # pyright: ignore[reportUnknownMemberType]
        self.__upsample = torch.nn.Upsample(scale_factor=2, mode=interpolation)
        self.__smooth = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            device=device
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__smooth(self.__upsample(x))  # pyright: ignore[reportAny]
