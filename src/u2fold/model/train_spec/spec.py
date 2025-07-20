from typing import Annotated, Literal

from pydantic import BaseModel, Field, PositiveInt

from .learning_rate_scheduler_spec import LRSchedulerSpec
from .loss_spec import LossSpec
from .optimizer_spec import OptimizerSpec


class TrainSpec(BaseModel):
    mode: Literal["train"]

    n_epochs: PositiveInt
    batch_size: PositiveInt

    optimizer_spec: OptimizerSpec = Field(
        description="Optimizer specification", discriminator="optimizer"
    )

    learning_rate_spec: LRSchedulerSpec = Field(
        description="Learning rate scheduler specification", discriminator="scheduler"
    )

    losses: frozenset[Annotated[LossSpec, Field(discriminator="loss")]]
