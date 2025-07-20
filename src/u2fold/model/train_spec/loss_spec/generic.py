from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, NonNegativeFloat
from torch import Tensor, nn

from u2fold.model.common_namespaces import ForwardPassResult


class BaseLossModule(nn.Module, ABC):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = weight

    @classmethod
    @abstractmethod
    def _forward(cls, result: ForwardPassResult, ground_truth: Tensor) -> Tensor: ...

    def forward(self, result: ForwardPassResult, ground_truth: Tensor) -> Tensor:
        return self.weight * self._forward(result, ground_truth)


class BaseLossSpec(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)
    weight: NonNegativeFloat

    @abstractmethod
    def instantiate(self) -> BaseLossModule: ...
