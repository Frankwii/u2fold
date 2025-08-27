from typing import final, override
import torch
from itertools import chain, pairwise

from u2fold.model.neural_network_spec.components.activation import Activation

@final
class UNetConvolutionalLayer(torch.nn.Module):
    """Layer containing several convolution sublayers"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sublayers: int,
        activation: Activation,
        device: str | None,
    ) -> None:
        torch.nn.Module.__init__(self)  # pyright: ignore[reportUnknownMemberType]

        channel_sizes = chain(
            (in_channels,), (out_channels for _ in range(sublayers - 1))
        )

        channel_size_pairs = pairwise(channel_sizes)

        self.__convlayers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                    device=device,
                )
                for in_channels, out_channels in channel_size_pairs
            ]
        )

        self.__activation = activation.instantiate()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for conv_layer in self.__convlayers:
            y: torch.Tensor = self.__activation(conv_layer(y))  # pyright: ignore[reportAny]

        return y
