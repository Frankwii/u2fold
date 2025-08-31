import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, final, override

import torch
from pydantic import BaseModel, ConfigDict, NonNegativeFloat
from torch import Tensor, nn

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
        return (self.weight / self.calibration_average) * self._forward(
            result, ground_truth
        )


class BaseLossSpec(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    model_config = ConfigDict(frozen=True)  # pyright: ignore[reportUnannotatedClassAttribute]
    weight: NonNegativeFloat

    @abstractmethod
    def instantiate(self) -> BaseLossModule: ...


@final
class Loss(nn.Module):
    def __init__(self, losses: Sequence[BaseLossModule]):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self._losses = nn.ModuleList(list(losses))
        self._last_losses: dict[str, float] = {}

    @classmethod
    def __get_module_name(cls, loss_module: BaseLossModule) -> str:
        name = loss_module.__class__.__name__

        return re.sub(r"(?<!^)([A-Z])", r"_\1", name.replace("Module", "")).lower()

    def __compute_and_store_loss(
        self,
        loss_module: BaseLossModule,
        result: ForwardPassResult,
        ground_truth: Tensor,
    ) -> Tensor:
        loss_tensor = loss_module(result, ground_truth)  # pyright: ignore[reportAny]
        self._last_losses[self.__get_module_name(loss_module)] = (
            loss_tensor.detach().mean().item()
        )  # pyright: ignore[reportAny]
        return loss_tensor

    @override
    def forward(
        self,
        result: ForwardPassResult,
        ground_truth: Tensor,
    ) -> Tensor:
        return torch.stack(
            tuple(
                self.__compute_and_store_loss(l, result, ground_truth)  # pyright: ignore[reportArgumentType]
                for l in self._losses
            )
        ).sum(dim=0)

    def get_last_losses(self) -> dict[str, float]:
        return self._last_losses
