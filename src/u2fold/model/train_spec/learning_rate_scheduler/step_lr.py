from typing import Literal

from pydantic import Field, PositiveFloat, PositiveInt
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer

from .generic import BaseLearningRateSchedulerModel


class StepLRModel(BaseLearningRateSchedulerModel[StepLR]):
    """Multiplies the learning rate by a fixed quantity each given number of
    steps."""

    scheduler: Literal["step_lr"]
    step_size: PositiveInt = Field(
        title="Step size",
        description="Number of steps elapsed between multiplications."
    )
    factor: PositiveFloat = Field(
        title="Factor",
        description="Multiplicative coefficient to apply."
    )

    def instantiate(self, optimizer: Optimizer) -> StepLR:
        return StepLR(optimizer, self.step_size, self.factor)

    @staticmethod
    def take_step(scheduler: StepLR, loss: Tensor) -> None:
        scheduler.step()
