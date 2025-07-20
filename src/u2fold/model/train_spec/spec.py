from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .dataset_spec import DatasetSpec

from .learning_rate_scheduler_spec import LRSchedulerSpec
from .loss_spec import LossSpec
from .optimizer_spec import OptimizerSpec


class TrainSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    mode: Literal["train"]

    optimizer_spec: OptimizerSpec = Field(
        title="Optimizer",
        description="Optimizer specification.", discriminator="optimizer"
    )

    learning_rate_scheduler_spec: LRSchedulerSpec = Field(
        title="LR scheduler",
        description="Learning rate scheduler specification.", discriminator="scheduler"
    )

    dataset_spec: DatasetSpec = Field(
        title="Dataset specification",
        description="Specifications for the dataset used to train the model.\n"
        "Also includes information about the number of epochs and how to load the dataset."
    )

    losses: list[Annotated[LossSpec, Field(
        title = "Loss function specification",
        description="Specification for the loss functions to be used.\n"
        "The final loss function will be the sum of the losses specified in this attribute.",
        discriminator="loss"
    )]]
