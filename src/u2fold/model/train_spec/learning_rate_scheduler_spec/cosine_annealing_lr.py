from typing import Literal

from pydantic import Field, NonNegativeFloat, PositiveInt
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer

from .generic import BaseLRScheduler, BaseLRSchedulerSpec


class CosineAnnealingLRScheduler(BaseLRScheduler[CosineAnnealingLR]):
    def step(self, loss: Tensor) -> None:
        self._scheduler.step()


class CosineAneallingLRSpec(BaseLRSchedulerSpec):
    """Varies the learning rate between its initial value and a lower bound
    in a sinusoidal fashion, starting with a peak (cosine).
    """

    scheduler: Literal["cosine_annealing_lr"]

    semiperiod: PositiveInt = Field(
        title="Semiperiod",
        description="Half of the period of the cosine. Corresponds to pytorch's T_max.",
    )

    minimum_learning_rate: NonNegativeFloat = Field(
        title="Minimum learning rate",
        description="Minimum value for the learning rate. This lower bound is "
        "achieved whenever the cosine reachs its minimum.",
        default=0,
    )

    def instantiate(self, optimizer: Optimizer) -> CosineAnnealingLRScheduler:
        return CosineAnnealingLRScheduler(
            CosineAnnealingLR(
                optimizer, T_max=self.semiperiod, eta_min=self.minimum_learning_rate
            )
        )
