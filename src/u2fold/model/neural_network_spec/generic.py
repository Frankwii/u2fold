from abc import ABC
from itertools import chain
from typing import Any, Iterable

from pydantic import BaseModel, Field, PositiveFloat, field_validator

from u2fold.model.neural_network_spec.components.activation.generic import (
    BaseActivationSpec,
)
from u2fold.model.neural_network_spec.components.pooling.generic import (
    BasePoolingMethod,
)

from .components.activation import Activation
from .components.pooling import PoolSpec

type Tree[A] = A | Iterable["Tree[A]"]


class BaseNeuralNetworkSpec(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    unfolded_step_size: PositiveFloat = Field(
        title="Unfolded step size",
        description="Step size for the unfolded proximity operator.",
    )

    def _format_attribute(self, attribute: Tree[Any]) -> str:
        if isinstance(attribute, BaseActivationSpec | BasePoolingMethod):
            return attribute.format_value()
        if isinstance(attribute, Iterable) and not isinstance(attribute, str):
            return "-".join(self._format_attribute(attr) for attr in attribute)
        else:
            substrings = str(attribute).split("_")
            [first, *others] = substrings

            camel_case_components = chain(
                (first,), (substr.capitalize() for substr in others)
            )

            return "".join(camel_case_components)

    def format_self(self) -> str:
        formattable_fields = (
            (field_name, getattr(self, field_name))
            for field_name in self.__class__.model_fields
            if field_name != "name"
        )

        formatted_name_value_pairs = (
            (self._format_attribute(name), self._format_attribute(value))
            for name, value in formattable_fields
        )

        return "__".join(
            f"{name}_{value}" for name, value in formatted_name_value_pairs
        )


class UNetLikeSpec(BaseNeuralNetworkSpec, ABC):
    """Config for a UNet-like architecture."""
    hidden_layers_activation: Activation = Field(title="Activation function", discriminator="name")
    final_residual_activation: Activation = Field(title="Activation function for the last residual connection", discriminator="name")
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
        ge=2,
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
