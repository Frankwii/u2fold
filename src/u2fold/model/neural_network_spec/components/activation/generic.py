from abc import ABC, abstractmethod

from pydantic import BaseModel
from torch import nn


class BaseActivationSpec(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]

    @abstractmethod
    def instantiate(self) -> nn.Module: ...

    def format_value(self) -> str:
        return getattr(self, 'name')
