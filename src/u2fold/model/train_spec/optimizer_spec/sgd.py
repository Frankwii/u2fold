from typing import Literal

from pydantic import Field, NonNegativeFloat
from torch.optim import SGD
from torch.optim.optimizer import ParamsT

from .generic import OptimizerModel


class SGDModel(OptimizerModel[SGD]):
    """Stochastic Gradient Descent"""

    optimizer: Literal["sgd"]

    momentum: NonNegativeFloat = Field(
        title="Momentum",
        description="Multiplicative coefficient assigned to the second-to-last"
        "computed gradient when updating weights.",
    )

    def instantiate(self, params: ParamsT) -> SGD:
        return SGD(params=params, lr=self.learning_rate, momentum=self.momentum)
