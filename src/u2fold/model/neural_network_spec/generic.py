from abc import ABC
from itertools import chain
from typing import Any, Iterable

from pydantic import BaseModel, Field, PositiveFloat

from u2fold.model.neural_network_spec.components.activation.generic import (
    BaseActivationSpec,
)
from u2fold.model.neural_network_spec.components.pooling.generic import (
    BasePoolingMethod,
)

type Tree[A] = A | Iterable["Tree[A]"]


class BaseNeuralNetworkSpec(BaseModel, ABC):
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
