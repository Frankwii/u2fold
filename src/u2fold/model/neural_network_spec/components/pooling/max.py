from typing import Literal

from torch import nn
from torch.nn import MaxPool2d
from u2fold.model.neural_network_spec.components.pooling.generic import BasePoolingMethod


class MaxPoolSpec(BasePoolingMethod):
    method: Literal["max"]

    def instantiate(self) -> nn.Module:
        return MaxPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride
        )
