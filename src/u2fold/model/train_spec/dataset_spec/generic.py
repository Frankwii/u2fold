from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel, PositiveInt


class BaseDatasetSpec(BaseModel, ABC):
    path: Path
    eager_load: bool
    n_epochs: PositiveInt
    batch_size: PositiveInt

    @abstractmethod
    def instantiate(self) -> Dataset: ...
