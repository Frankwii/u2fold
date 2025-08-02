from abc import ABC, abstractmethod
from typing import ClassVar, final, override
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, NonNegativeFloat
from torch import Tensor, nn
import torch

from u2fold.model.common_namespaces import ForwardPassResult


class BaseLossModule(nn.Module, ABC):
    calibration_average: ClassVar[float]
    def __init__(self, weight: float) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.weight: float = weight

    @classmethod
    @abstractmethod
    def _forward(cls, result: ForwardPassResult, ground_truth: Tensor) -> Tensor: ...

    @override
    def forward(self, result: ForwardPassResult, ground_truth: Tensor) -> Tensor:
        return self.weight * self._forward(result, ground_truth) / self.calibration_average


class BaseLossSpec(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    model_config = ConfigDict(frozen=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    weight: NonNegativeFloat

    @abstractmethod
    def instantiate(self) -> BaseLossModule: ...


@final
class Loss(nn.Module):
    def __init__(self, losses: Sequence[BaseLossModule]):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.losses = nn.ModuleList(list(losses))

    @override
    def forward(
        self,
        result: ForwardPassResult,
        ground_truth: Tensor,
    ) -> Tensor:
        return torch.stack(
            [l(result, ground_truth) for l in self.losses]
        ).sum(dim=0)
