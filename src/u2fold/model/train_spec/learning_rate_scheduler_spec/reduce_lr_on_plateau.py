from typing import Literal

from pydantic import Field, NonNegativeInt, PositiveFloat
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .generic import BaseLRScheduler, BaseLRSchedulerSpec


class ReduceLROnPlateauScheduler(BaseLRScheduler[ReduceLROnPlateau]):
    def step(self, loss: Tensor | float) -> None:
        self._scheduler.step(loss)


class ReduceLROnPlateauSpec(BaseLRSchedulerSpec):
    """Multiplies the learning rate by a small constant after a number of
    consecutive steps without a signficant loss decrease.
    """

    scheduler: Literal["reduce_lr_on_plateau"]
    factor: PositiveFloat = Field(
        title="Factor",
        description="Factor by which to multiply the learning rate.",
    )

    patience: NonNegativeInt = Field(
        title="Patience",
        description="Number of consecutive non-significant steps that are "
        "awaited before reducing the learning rate.",
    )

    threshold: float = Field(
        title="Threshold",
        description="Relative threshold by which to consider a step as not "
        "signficant. Concretely, a step is considered signficant if its "
        "current loss is smaller than (1-threshold) times the loss at the "
        "last signficant step. The only exception to this is first step, which "
        "is always signficant.",
    )

    def instantiate(self, optimizer: Optimizer) -> ReduceLROnPlateauScheduler:
        return ReduceLROnPlateauScheduler(
            ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
            )
        )
