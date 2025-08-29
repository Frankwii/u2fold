import logging
from pathlib import Path

from torch import Tensor

from u2fold.utils import tag

from ..dataloader_generics.base import (
    CollationFunction,
    DataLoaderConfig,
    ToDeviceDataLoader,
)
from ..dataset_splits import DatasetSplits, SplitData, split_dataset
from .collation import UIEBRandomCollateAndTransform, UIEBTopLeftCropCollate
from .dataset import UIEBDataset

_logger = logging.getLogger(__name__)


class UIEBDataLoaderConfig[T, U, *Tensors](DataLoaderConfig[T, U, *Tensors]): ...


@tag("data/dataloader/uieb")
class UIEBDataLoader(ToDeviceDataLoader[Tensor, Tensor]):
    @classmethod
    def get_dataloaders(
        cls, dataset_path: Path, batch_size: int, device: str
    ) -> SplitData["UIEBDataLoader"]:
        dataset = UIEBDataset(dataset_path)
        dataset_class = dataset.__class__.__name__
        splits = DatasetSplits(0.8, 0.1, 0.1)

        dataset_splits = split_dataset(dataset, splits)  # pyright: ignore[reportArgumentType]

        def __instantiate_uieb_dataloader[*Tensors](
            dataset: UIEBDataset, config: tuple[bool, CollationFunction[*Tensors]]
        ) -> UIEBDataLoader:
            _logger.info(f"Instantiating dataloader for {dataset_class}...")
            dataloader_config = UIEBDataLoaderConfig(
                dataset,
                batch_size=batch_size,
                shuffle=config[0],
                collate_fn=config[1],
                pin_memory=True,
                num_workers=0,
            )

            dataloader = UIEBDataLoader(device, dataloader_config)  # pyright: ignore[reportArgumentType]
            _logger.debug(
                f"Successfully instantiated dataloader for {dataset_class}."
            )

            return dataloader

        instantiation_config = SplitData(
            training=(True, UIEBRandomCollateAndTransform()),
            validation=(False, UIEBTopLeftCropCollate()),
            test=(False, UIEBTopLeftCropCollate()),
        )

        dataloaders = dataset_splits.map(
            __instantiate_uieb_dataloader, instantiation_config
        )

        return dataloaders
