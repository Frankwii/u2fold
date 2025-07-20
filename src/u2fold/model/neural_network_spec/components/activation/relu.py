from typing import Literal

from torch import nn

from .generic import BaseActivationSpec


class ReLU(BaseActivationSpec):
    name: Literal["relu"]

    def instantiate(self) -> nn.Module:
        return nn.ReLU()
