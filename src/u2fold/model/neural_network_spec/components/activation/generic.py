from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel
from torch import nn


class BaseActivationSpec(BaseModel, ABC):

    @abstractmethod
    def instantiate(self) -> nn.Module: ...

    def format_value(self) -> str:
        return getattr(self, 'name')
