import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, cast, final
from collections.abc import Iterator

from torch.utils.data import DataLoader

from u2fold.data.dataset_splits import SplitData

from ..dataset_generics.base import U2FoldDataset

type CollationFunction[*Tensors] = Callable[
    [list[tuple[*Tensors]]],
    tuple[*Tensors],
]


@dataclass
class DataLoaderConfig[T, U, *Tensors]:
    dataset: U2FoldDataset[T, U]
    batch_size: int
    shuffle: bool
    collate_fn: CollationFunction[*Tensors] | None
    pin_memory: bool
    num_workers: int

    @final
    def instantiate_dataloader(self) -> DataLoader[U]:
        field_names = (field.name for field in fields(self))
        args = {name: getattr(self, name) for name in field_names}
        return DataLoader(**args)


class U2FoldDataLoader[T, U, *Tensors](ABC):
    @final
    def __init__(self, device: str, config: DataLoaderConfig[T, U, *Tensors]) -> None:
        self._device = device
        self._logger = logging.getLogger(
            f"{__name__}|{self.__class__.__name__}"
        )

        self._dataloader = config.instantiate_dataloader()

    @classmethod
    @abstractmethod
    def get_dataloaders[X](
        cls: type[X], dataset_path: Path, batch_size: int, device: str
    ) -> SplitData[X]: ...

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[*Tensors]]: ...

    @final
    def __len__(self) -> int:
        return len(self._dataloader)


class ToDeviceDataLoader[T, U, *Tensors](U2FoldDataLoader[T, U, *Tensors], ABC):
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
