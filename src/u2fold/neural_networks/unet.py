from itertools import pairwise
from typing import final, override

import torch

from u2fold.model.neural_network_spec.unet import UNetSpec
from .unet_like import ResidualUNetLikeNetwork
from u2fold.utils.track import tag


@final
@tag("model/aunet")
class UNet(ResidualUNetLikeNetwork[UNetSpec]):
    """UNet.

    Similar to the classical UNet, but with instance normalization,
    reflection padding and upsampling is done with a traditional method
    followed by a single convolutional layer, as described in
    https://distill.pub/2016/deconv-checkerboard/.

    Moreover, the network itself only learns a residual.
    """
    @override
    @classmethod
    def get_layer_channel_sizes(
        cls, channels_per_layer: list[int]
    ) -> list[tuple[int, int]]:
        layer_size_sequence = (
            3,
            *channels_per_layer,
            *reversed(channels_per_layer[:-1]),
        )

        return list(pairwise(layer_size_sequence))

    @override
    @classmethod
    def skip_connect(cls, encoder_output: torch.Tensor, upsampling_result: torch.Tensor) -> torch.Tensor:
        return torch.cat((encoder_output, upsampling_result), dim=-3)
