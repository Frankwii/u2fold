from abc import ABC
from typing import final

from torch import Tensor

from .base import _GenericDataset


class GroundTruthDataset(_GenericDataset, ABC):
    _dataset_parts = ("input", "ground_truth")

    @final
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return (
            self.get_part_element("input", index),
            self.get_part_element("ground_truth", index),
        )


class UnsupervisedDataset(_GenericDataset, ABC):
    _dataset_parts = ("input",)

    @final
    def __getitem__(self, index: int) -> Tensor:
        return self.get_part_element("input", index)
