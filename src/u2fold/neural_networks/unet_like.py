from abc import ABC, abstractmethod
from itertools import pairwise
from typing import TypeVar, cast, final, override

import torch

from u2fold.model.neural_network_spec import NeuralNetworkSpec

from .blocks.convlayer import UNetConvolutionalLayer
from .blocks.upsampling import UpsamplingLayer
from .generic import NeuralNetwork

Spec = TypeVar("Spec", bound = NeuralNetworkSpec, covariant=True)
class ResidualUNetLikeNetwork(NeuralNetwork[Spec], ABC):
    @classmethod
    @abstractmethod
    def get_layer_channel_sizes(
        cls, channels_per_layer: list[int]
    ) -> list[tuple[int, int]]: ...

    @classmethod
    @abstractmethod
    def skip_connect(
        cls, encoder_output: torch.Tensor, upsampling_result: torch.Tensor
    ) -> torch.Tensor: ...

    @final
    def __init__(self, spec: Spec, device: str | None = None) -> None:
        torch.nn.Module.__init__(self)  # pyright: ignore[reportUnknownMemberType]
        self.__depth = len(spec.channels_per_layer)

        layer_channel_sizes = self.get_layer_channel_sizes(spec.channels_per_layer)

        all_block_layers = torch.nn.ModuleList(
            [
                *(
                    UNetConvolutionalLayer(
                        in_channels,
                        out_channels,
                        spec.sublayers_per_step,
                        spec.activation,
                        device,
                    )
                    for in_channels, out_channels in layer_channel_sizes
                ),
            ]
        )

        all_normalization_layers = torch.nn.ModuleList(
            torch.nn.InstanceNorm2d(in_channels, device=device)
            for (in_channels, _) in layer_channel_sizes
        )

        self.__encoder_layers = all_block_layers[: self.__depth - 1]
        self.__encoder_normalization_layers = all_normalization_layers[
            : self.__depth - 1
        ]
        self.__downsample = spec.pooling.instantiate()

        self.__bottleneck = all_block_layers[self.__depth - 1]
        self.__bottleneck_normalization_layer = all_normalization_layers[
            self.__depth - 1
        ]

        self.__decoder_layers = all_block_layers[self.__depth:]
        self.__decoder_normalization_layers = all_normalization_layers[self.__depth:]
        self.__upsampling_layers = torch.nn.ModuleList(
            [
                UpsamplingLayer("nearest", in_channels, out_channels, device)
                for in_channels, out_channels in pairwise(
                    reversed(spec.channels_per_layer)
                )
            ]
        )

        self.__final_convolution = torch.nn.Conv2d(
            spec.channels_per_layer[0],
            3,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            device=device,
        )
        self.__final_normalization = torch.nn.InstanceNorm2d(
            spec.channels_per_layer[0], device=device
        )

    @final
    @override
    def forward(self, input: torch.Tensor, *_) -> torch.Tensor:
        encoder_layer_outputs: list[torch.Tensor] = []

        x = input
        for layer, layer_norm in zip(
            self.__encoder_layers, self.__encoder_normalization_layers
        ):
            x = layer(layer_norm(x))  # pyright: ignore[reportAny]
            encoder_layer_outputs.append(x)  # pyright: ignore[reportAny]
            x = self.__downsample(x)  # pyright: ignore[reportAny]

        x = self.__bottleneck(self.__bottleneck_normalization_layer(x))  # pyright: ignore[reportAny]

        for idx, (layer, layer_norm, upsampling_layer) in enumerate(
            zip(
                self.__decoder_layers,
                self.__decoder_normalization_layers,
                self.__upsampling_layers,
            )
        ):
            x = self.skip_connect(
                cast(torch.Tensor, upsampling_layer(x)), encoder_layer_outputs[~idx]
            )
            x = layer(layer_norm(x))  # pyright: ignore[reportAny]

        return input + self.__final_convolution(self.__final_normalization(x))  # pyright: ignore[reportAny]
