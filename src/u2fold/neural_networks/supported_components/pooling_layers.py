from typing import Callable

import torch
from torch.nn import AvgPool2d, LPPool2d, MaxPool2d

from u2fold.exceptions.unsupported_parameter import UnsupportedParameter

type TensorTransformation = Callable[[torch.Tensor], torch.Tensor]
type PoolingLayer = Callable[[int, int], TensorTransformation]

SUPPORTED_POOLING_LAYERS: dict[
    str, Callable[[int, int], TensorTransformation]
] = {
    "max": MaxPool2d,
    "avg": AvgPool2d,
    "l2": lambda kernel_size, stride: LPPool2d(
        norm_type=2, kernel_size=kernel_size, stride=stride
    ),
}

def get_pooling_layer(name: str) -> PoolingLayer:
    if name not in SUPPORTED_POOLING_LAYERS.keys():
        raise UnsupportedParameter(name, list(SUPPORTED_POOLING_LAYERS.keys()))

    return SUPPORTED_POOLING_LAYERS[name]

def get_supported_pooling_layers() -> set[str]:
    return set(SUPPORTED_POOLING_LAYERS.keys())
