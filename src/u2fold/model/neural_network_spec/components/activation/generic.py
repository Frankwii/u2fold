from abc import ABC, abstractmethod

from pydantic import BaseModel
from torch import nn


class BaseActivationSpec(BaseModel, ABC):
    @abstractmethod
    def instantiate(self) -> nn.Module: ...
