from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .components.activation import Activation
from .components.pooling import PoolSpec


class UNetConfig(BaseModel):
    """Config for a UNet-like architecture"""

    name: Literal["unet"]
    activation: Activation = Field(title="Activation function", discriminator="name")
    pooling: PoolSpec = Field(
        title="Pooling function specification", discriminator="method"
    )
    channels_per_layer: list[int] = Field(
        title="Channels per layer",
        description=(
            "Number of channels in each decoder/encoder sublayer, and the "
            "bottleneck.\n"
            "The number of channels of a layer is defined as the number of "
            "channels of the output of the respective encoder and decoder step "
            "(which is the same for both).The number of channels of the "
            "bottleneck is defined as the number of channels of its "
            "intermediate blocks. The last element of this list is parsed as "
            "the number of channels of the bottleneck. The rest are parsed, "
            "from left to right, as the numbers of channels of the "
            "encoder/decoder layers, in execution order."
        ),
        examples=[4, 8, 16],
    )
    sublayers_per_step: int = Field(
        ge=3,
        title="Sublayers per step",
        description=(
            "Number of Conv2d sublayers in each encoder/decoder layer, and in "
            "the bottleneck."
        ),
    )

    @field_validator("channels_per_layer", mode="after")
    @classmethod
    def validate_channels_per_step(cls, l: list[int]) -> list[int]:
        if len(l) < 2:
            raise ValueError(
                "There must be at least two sizes: one for the bottleneck "
                "and one for each of the decoder/encoder layers."
            )

        return l
