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
            "choices": components.get_supported_pooling_layers()
        }
    )

    activation: str = field(
        metadata={
            "desc": "Activation function to use between layers.",
            "choices": components.get_supported_activaton_functions()
        }
    )

    def __validate_channels_per_layer(self) -> None:
        if (n := len(self.channels_per_layer)) < 2 or n % 2 == 0:
            raise ValueError(
                "Insufficient number of layers for UNet. Must be"
                " an odd number bigger than or equal to 3."
            )

        intermediate_layers = self.channels_per_layer[1:-1]
        if intermediate_layers[::-1] != intermediate_layers:
            raise ValueError(
                "Invalid number of channels per UNet layer. The"
                " array of intermediate channel sizes must be"
                " symmetrical."
            )

    def __validate_sublayers_per_step(self) -> None:
        if self.sublayers_per_step < 2:
            raise ValueError(
                "Insufficient sublayers per step. Must be at least 2"
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

        global activation_functions
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

        self.__depth = len(config.channels_per_layer) // 2
        channel_sizes = sliding_window(config.channels_per_layer)

        all_layers = nn.ModuleList(
            [
                UNetConvolutionalLayer(
                    in_channels,
                    out_channels,
                    config.sublayers,
                    config.activation,
                    device,
                )
                for in_channels, out_channels in channel_sizes
            ]
        )

        self.__down_sublayers = all_layers[: self.__depth]
        global pooling_classes
        pooling_class = components.get_pooling_layer(config.pooling)
        self.__downsampling_layer = pooling_class(2, 2)

        self.__bottom_sublayer = all_layers[self.__depth]

        self.__up_sublayers = all_layers[self.__depth + 1 :]
        self.__upsampling_layer = Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_layer_outputs = [torch.Tensor() for _ in range(self.__depth)]

        for idx, sublayer in enumerate(self.__down_sublayers):
            x = sublayer(x)
            down_layer_outputs[idx] = x
            x = self.__downsampling_layer(x)

        x = self.__bottom_sublayer(x)

        for idx, sublayer in enumerate(self.__up_sublayers):
            x = self.__upsampling_layer(x)
            x = sublayer(x + down_layer_outputs[-(idx + 1)])

        return x
