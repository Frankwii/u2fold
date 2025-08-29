
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, PositiveFloat
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT


class BaseOptimizerSpec[Optim: Optimizer](BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    learning_rate: PositiveFloat = Field(
        title="Learning rate.",
        description="Step size for the optimizer (~gradient) updates. May be "
        "modified by a scheduler."
    )

    @abstractmethod
    def instantiate(self, params: ParamsT) -> Optim: ...
