from typing import Callable

import torch

from u2fold.exceptions.unsupported_parameter import UnsupportedParameter

SUPPORTED_ACTIVATION_FUNCTIONS: dict[
    str, Callable[[torch.Tensor], torch.Tensor]
] = {
    "relu": torch.nn.functional.relu,
    "gelu": torch.nn.functional.gelu,
}


def get_activation_function(
    name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if name not in SUPPORTED_ACTIVATION_FUNCTIONS.keys():
        raise UnsupportedParameter(
            name, list(SUPPORTED_ACTIVATION_FUNCTIONS.keys())
        )

    return SUPPORTED_ACTIVATION_FUNCTIONS[name]


def get_supported_activaton_functions() -> set[str]:
    return set(SUPPORTED_ACTIVATION_FUNCTIONS.keys())
