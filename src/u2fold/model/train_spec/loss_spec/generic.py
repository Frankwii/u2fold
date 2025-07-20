from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, NonNegativeFloat
from torch import Tensor

from u2fold.model.common_namespaces import ForwardPassResult


class BaseLoss(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)
    weight: NonNegativeFloat

    @abstractmethod
    def forward(self, result: ForwardPassResult, ground_truth: Tensor) -> Tensor: ...
