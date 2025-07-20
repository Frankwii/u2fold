from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class BaseLRScheduler[Sched: LRScheduler](ABC):
    def __init__(self, scheduler: Sched):
        self._scheduler = scheduler

    @abstractmethod
    def step(self, loss: Tensor) -> None: ...


class BaseLRSchedulerSpec(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def instantiate(self, optimizer: Optimizer) -> BaseLRScheduler[LRScheduler]: ...
