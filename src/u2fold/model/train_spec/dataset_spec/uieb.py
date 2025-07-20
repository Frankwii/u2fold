from typing import Literal

from torch.utils.data import Dataset

from .generic import BaseDatasetSpec


class UiebSpec(BaseDatasetSpec):
    name: Literal["uieb"]

    def instantiate(self) -> Dataset:
        raise NotImplementedError()
