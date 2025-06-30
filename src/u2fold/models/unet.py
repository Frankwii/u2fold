from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import torch
from torch import nn
from torch.nn import Upsample

from u2fold.models import supported_components as components
from u2fold.utils.sliding_window import sliding_window
from u2fold.utils.track import tag

from .generic import Model, ModelConfig


@tag("config/model/unet")
@dataclass
class ConfigUNet(ModelConfig):
    channels_per_layer: list[int] = field(
        metadata={"desc": "Number of channels for each layer of the UNet."}
    )

    sublayers_per_step: int = field(
        metadata={
            "desc": "Number of sublayers in each descending or ascending step.",
        }
    )

    pooling: str = field(
        metadata={
            "desc": "Pooling technique to use after convolutional layers.",
            "choices": components.get_supported_pooling_layers(),
        }
    )

    activation: str = field(
        metadata={
            "desc": "Activation function to use between layers.",
            "choices": components.get_supported_activaton_functions(),
        }
    )

    def __validate_channels_per_layer(self) -> None:
        if len(self.channels_per_layer) < 2:
            raise ValueError(
                "Insufficient number of layers for UNet. Must be at least 2."
            )

    def __validate_sublayers_per_step(self) -> None:
        if self.sublayers_per_step < 2:
            raise ValueError(
                "Insufficient sublayers per UNet step. Must be at least 2"
            )

    def validate(self) -> None:
        super().validate()

        self.__validate_channels_per_layer()
        self.__validate_sublayers_per_step()

    def __post_init__(self) -> None:
        super().__post_init__()


class UNetConvolutionalLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sublayers: int,
        activation: str,
        device: Optional[str],
    ) -> None:
        super().__init__()

        channel_sizes = chain(
            (in_channels,), (out_channels for _ in range(sublayers - 1))
        )

        channel_size_pairs = sliding_window(channel_sizes)

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

        self.__activation = components.get_activation_function(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_layer in self.__convlayers:
            x = self.__activation(conv_layer(x))

        return x


@tag("model/unet")
class UNet(Model[ConfigUNet]):
    """Mimick the UNet architecture."""

    def __init__(
        self, config: ConfigUNet, device: Optional[str] = None
    ) -> None:
        torch.nn.Module.__init__(self)

        self.__depth = len(config.channels_per_layer)

        size_sequence = (
            3,
            *config.channels_per_layer,
            *reversed(config.channels_per_layer),
            3
        )
        channel_sizes = sliding_window(size_sequence)

        all_layers = nn.ModuleList(
            [
                UNetConvolutionalLayer(
                    in_channels,
                    out_channels,
                    config.sublayers_per_step,
                    config.activation,
                    device,
                )
                for in_channels, out_channels in channel_sizes
            ]
        )

        self.__down_sublayers = all_layers[: self.__depth]
        pooling_class = components.get_pooling_layer(config.pooling)
        self.__downsample = pooling_class(2, 2)

        self.__bottleneck = all_layers[self.__depth]

        self.__up_sublayers = all_layers[self.__depth + 1 :]
        self.__upsample = Upsample(scale_factor=2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        down_layer_outputs = [torch.Tensor() for _ in range(self.__depth)]

        x = input
        counter = 0
        for idx, sublayer in enumerate(self.__down_sublayers):
            x = sublayer(x)
            down_layer_outputs[idx] = x
            x = self.__downsample(x)

        output_shapes = [y.shape for y in down_layer_outputs]
        x = self.__bottleneck(x)

        for idx, sublayer in enumerate(self.__up_sublayers):
            x = self.__upsample(x)
            x = sublayer(x + down_layer_outputs[~idx])

        return x
