from typing import Literal

from u2fold.data import SplitData
from u2fold.data.uieb_handling.dataloaders import UIEBDataLoader

from .generic import BaseDatasetSpec
from u2fold.data import get_dataloaders
from u2fold.utils import get_device


class UiebSpec(BaseDatasetSpec[UIEBDataLoader]):
    name: Literal["uieb"]

    def instantiate(self) -> SplitData[UIEBDataLoader]:
        return get_dataloaders(
            dataset=self.name,
            dataset_path=self.path,
            batch_size=self.batch_size,
            device=get_device()
        )
