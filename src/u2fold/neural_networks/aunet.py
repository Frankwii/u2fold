from itertools import chain, pairwise
from typing import cast, final, override

import torch
from torch import nn
from torch.nn import Upsample

from u2fold.model.neural_network_spec.aunet import AUNetSpec
from u2fold.neural_networks.unet_like import ResidualUNetLikeNetwork
from u2fold.utils.track import tag

@final
@tag("model/aunet")
class AUNet(ResidualUNetLikeNetwork[AUNetSpec]):
    """Additive UNet."""

    @classmethod
    @override
    def get_layer_channel_sizes(
        cls, channels_per_layer: list[int]
    ) -> list[tuple[int, int]]:
        return list(
            chain(
                pairwise((3, *channels_per_layer)),
                (
                    (num_channels, num_channels)
                    for num_channels in reversed(channels_per_layer[1:])
                ),
            )
        )

    @classmethod
    @override
    def skip_connect(
        cls, encoder_output: torch.Tensor, upsampling_result: torch.Tensor
    ) -> torch.Tensor:
        return encoder_output + upsampling_result
