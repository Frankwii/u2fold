from pydantic import BaseModel, Field, PositiveInt

from .learning_rate_scheduler import LRSchedulerModel
from .optimizer_spec import OptimizerModel


class TrainSpec(BaseModel):
    n_epochs: PositiveInt
    batch_size: PositiveInt

    optimizer_spec: OptimizerModel = Field(
        description="Optimizer specification",
        discriminator="optimizer"
    )

    learning_rate_spec: LRSchedulerModel = Field(
        description="Learning rate scheduler specification",
        discriminator="scheduler"
    )
