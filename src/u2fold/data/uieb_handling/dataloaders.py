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


class UIEBDataLoaderConfig(DataLoaderConfig):
    ...

@tag("data/dataloader/uieb")
class UIEBDataLoader(ToDeviceDataLoader[Tensor, Tensor]):
    ...

def get_dataloaders(
    uieb_path: Path, batch_size: int, device: str
) -> SplitData[UIEBDataLoader]:
    dataset = UIEBDataset(uieb_path)
    splits = DatasetSplits(0.8, 0.1, 0.1)

    dataset_splits = split_dataset(dataset, splits)

    def __instantiate_uieb_dataloader(
        dataset: UIEBDataset, config: tuple[bool, CollationFunction]
    ) -> UIEBDataLoader:
        dataloader_config = UIEBDataLoaderConfig(
            dataset,
            batch_size=batch_size,
            shuffle=config[0],
            collate_fn=config[1],
            pin_memory=True,
            num_workers=0
        )

        return UIEBDataLoader(device, dataloader_config)

    instantiation_config = SplitData(
        training=(True, UIEBRandomCollateAndTransform()),
        validation=(False, UIEBTopLeftCropCollate()),
        test=(False, UIEBTopLeftCropCollate())
    )

    dataloaders = dataset_splits.map(
        __instantiate_uieb_dataloader, instantiation_config
    )

    return dataloaders
