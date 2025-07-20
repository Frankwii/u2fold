from typing import Literal
from torch.optim import Adam
from torch.optim.optimizer import ParamsT

from .generic import BaseOptimizerSpec

class AdamSpec(BaseOptimizerSpec[Adam]):
    """Adam: Adaptive Moment Estimation"""
    optimizer: Literal["adam"]

    def instantiate(self, params: ParamsT) -> Adam:
        return Adam(
            params=params,
            lr = self.learning_rate
        )
