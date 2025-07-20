from typing import Literal

from torch import nn

from .generic import BaseActivationSpec


class GeLU(BaseActivationSpec):
    name: Literal["gelu"]

    def instantiate(self) -> nn.Module:
        return nn.GELU()
