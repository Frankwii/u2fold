from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel, PositiveInt

from u2fold.data.dataloader_generics.base import U2FoldDataLoader
from u2fold.data.dataset_splits import SplitData


class BaseDatasetSpec[D: U2FoldDataLoader](BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance, reportMissingTypeArgument]
    path: Path
    n_epochs: PositiveInt
    batch_size: PositiveInt

    @abstractmethod
    def instantiate(self) -> SplitData[D]: ...
