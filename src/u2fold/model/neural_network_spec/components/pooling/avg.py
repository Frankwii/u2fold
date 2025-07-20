from typing import Literal

from torch import nn
from torch.nn import AvgPool2d
from u2fold.model.neural_network_spec.components.pooling.generic import BasePoolingMethod


class AvgPoolSpec(BasePoolingMethod):
    method: Literal["avg"]

    def instantiate(self) -> nn.Module:
        return AvgPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride
        )
