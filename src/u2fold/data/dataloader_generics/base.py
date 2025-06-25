import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Iterator, Optional, cast, final

from torch.utils.data import DataLoader

from u2fold.data.dataset_splits import SplitData

from ..dataset_generics.base import U2FoldDataset

type CollationFunction[*Tensors] = Callable[
    [list[tuple[*Tensors]]],
    tuple[*Tensors],
]


@dataclass
class DataLoaderConfig[*Tensors]:
    dataset: U2FoldDataset
    batch_size: int
    shuffle: bool
    collate_fn: Optional[CollationFunction[*Tensors]]
    pin_memory: bool
    num_workers: int

    @final
    def instantiate_dataloader(self) -> DataLoader:
        field_names = (field.name for field in fields(self))
        args = {name: getattr(self, name) for name in field_names}
        return DataLoader(**args)


class U2FoldDataLoader[*Tensors](ABC):
    @final
    def __init__(self, device: str, config: DataLoaderConfig[*Tensors]) -> None:
        self._device = device
        self._logger = logging.getLogger(
            f"{__name__}|{self.__class__.__name__}"
        )

        self._dataloader = config.instantiate_dataloader()

    @classmethod
    def get_dataloaders[T](
        cls: type[T], dataset_path: Path, batch_size: int, device: str
    ) -> SplitData[T]: ...

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[*Tensors]]: ...

    @final
    def __len__(self) -> int:
        return len(self._dataloader)


class ToDeviceDataLoader[*Tensors](U2FoldDataLoader[*Tensors], ABC):
    @final
    def __iter__(self) -> Iterator[tuple[*Tensors]]:
        for paired_batch in self._dataloader:
            yield cast(
                tuple[*Tensors],
                tuple(
                    batch.to(self._device, non_blocking=True)
                    for batch in paired_batch
                ),
            )
