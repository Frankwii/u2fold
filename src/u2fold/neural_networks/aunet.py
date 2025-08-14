from itertools import chain, pairwise
from typing import cast, final, override

import torch
from torch import nn
from torch.nn import Upsample

from u2fold.model.neural_network_spec.components.activation import Activation
from u2fold.model.neural_network_spec.aunet import AUNetSpec
from u2fold.utils.track import tag

from .generic import NeuralNetwork

@final
class UNetConvolutionalLayer(torch.nn.Module):
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

        self.__convlayers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
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


@final
@tag("model/aunet")
class AUNet(NeuralNetwork[AUNetSpec]):
    """Additive UNet.

    Similar to the classical UNet, but instead of concatenating the tensors on skip connections,
    it adds them. This draws it closer to a residual network.
    """

    def __init__(
        self, spec: AUNetSpec, device: str | None = None
    ) -> None:
        torch.nn.Module.__init__(self)  # pyright: ignore[reportUnknownMemberType]

        self.__depth = len(
            spec.channels_per_layer
        )

        size_sequence = (
            3,
            *spec.channels_per_layer,
            *reversed(spec.channels_per_layer),
            3,
        )
        channel_sizes = pairwise(size_sequence)

        all_layers = nn.ModuleList(
            [
                UNetConvolutionalLayer(
                    in_channels,
                    out_channels,
                    spec.sublayers_per_step,
                    spec.activation,
                    device,
                )
                for in_channels, out_channels in channel_sizes
            ]
        )

        all_normalization_layers = nn.ModuleList(
            torch.nn.InstanceNorm2d(in_channels, device=device)
            for in_channels in size_sequence[:-1]
        )

        self.__down_sublayers = all_layers[: self.__depth]
        self.__down_normalizations = all_normalization_layers[: self.__depth]
        self.__downsample = spec.pooling.instantiate()

        self.__bottleneck = all_layers[self.__depth]
        self.__bottleneck_normalization = all_normalization_layers[self.__depth]

        self.__up_sublayers = all_layers[self.__depth + 1 :]
        self.__up_normalizations = all_normalization_layers[self.__depth + 1 :]
        self.__upsample = Upsample(scale_factor=2)

    @override
    def forward(self, input: torch.Tensor, *_) -> torch.Tensor:
        down_layer_outputs = [torch.Tensor() for _ in range(self.__depth)]

        x = input
        for idx, (sublayer, sublayer_norm) in enumerate(
            zip(self.__down_sublayers, self.__down_normalizations)
        ):
            x = sublayer(sublayer_norm(x))  # pyright: ignore[reportAny]
            down_layer_outputs[idx] = x
            x = self.__downsample(x)  # pyright: ignore[reportAny]

        x = self.__bottleneck(self.__bottleneck_normalization(x))  # pyright: ignore[reportAny]

        for idx, (sublayer, sublayer_norm) in enumerate(
            zip(self.__up_sublayers, self.__up_normalizations)
        ):
            x = self.__upsample(x)  # pyright: ignore[reportAny]
            x = sublayer(sublayer_norm(x) + down_layer_outputs[~idx])  # pyright: ignore[reportAny]

        return cast(torch.Tensor, input + x)
