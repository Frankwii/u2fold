from pathlib import Path
from typing import Iterator, cast

from torch import Tensor
from torch.utils.data import DataLoader

from ..dataset_splits import DatasetSplits, SplitData, split_dataset
from .collation import UIEBCollateAndTransform
from .dataset import UIEBDataset


class UIEBDataLoader:
    """Dataloader for the UIEB dataset.

    There are three important implementation details for this class:
    + The underlying dataset must be fully stored in memory (ideally, CPU).
    + The returned tensors are already transformed (flipped, rotated...)
      and loaded into "device".
    + The last "moving to device" operation is asynchronous. Therefore, there
      is room for optimization by taking this into account when using the
      dataloader.
    """

    def __init__(
        self,
        dataset: UIEBDataset,
        batch_size: int,
        shuffle: bool,
        device: str,
    ) -> None:
        self.__dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=0,
            collate_fn=UIEBCollateAndTransform(),
        )

        self.__device = device

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for input_batch, ground_truth_batch in self.__dataloader:
            input_batch = cast(Tensor, input_batch)
            ground_truth_batch = cast(Tensor, ground_truth_batch)

            yield (
                input_batch.to(self.__device, non_blocking=True),
                ground_truth_batch.to(self.__device, non_blocking=True),
            )

    def __len__(self) -> int:
        return len(self.__dataloader)


def get_dataloaders(
    uieb_path: Path, batch_size: int, device: str
) -> SplitData[UIEBDataLoader]:
    dataset = UIEBDataset(uieb_path)
    splits = DatasetSplits(0.8, 0.1, 0.1)

    dataset_splits = split_dataset(dataset, splits)

    def __instantiate_uieb_dataloader(
        dataset: UIEBDataset, shuffle: bool
    ) -> UIEBDataLoader:
        return UIEBDataLoader(dataset, batch_size, shuffle, device)

    shuffling_config = SplitData(training=True, validation=False, test=False)

    dataloaders = dataset_splits.map(
        __instantiate_uieb_dataloader, shuffling_config
    )

    return dataloaders
