from abc import ABC, abstractmethod

from pydantic import BaseModel
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class BaseLearningRateSchedulerModel[Sched: LRScheduler](BaseModel, ABC):
    @abstractmethod
    def instantiate(self, optimizer: Optimizer) -> Sched: ...

    @abstractmethod
    @staticmethod
    def take_step(scheduler: Sched, loss: Tensor) -> None: ...
