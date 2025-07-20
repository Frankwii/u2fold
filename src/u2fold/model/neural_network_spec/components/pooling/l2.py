from typing import Literal

from torch import nn
from torch.nn import LPPool2d
from u2fold.model.neural_network_spec.components.pooling.generic import BasePoolingMethod


class L2PoolSpec(BasePoolingMethod):
    method: Literal["l2"]

    def instantiate(self) -> nn.Module:
        return LPPool2d(
            norm_type=2,
            kernel_size=self.kernel_size,
            stride=self.stride
        )
