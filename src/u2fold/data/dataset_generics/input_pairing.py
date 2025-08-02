from abc import ABC
from typing import final

from torch import Tensor

from .base import U2FoldDataset


class GroundTruthDataset[T](U2FoldDataset[T, tuple[Tensor, Tensor]], ABC):
    _dataset_parts: tuple[str, str] = ("input", "ground_truth")  # pyright: ignore[reportIncompatibleVariableOverride]

    @final
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return (
            self.get_part_element("input", index),
            self.get_part_element("ground_truth", index),
        )


class UnsupervisedDataset[T](U2FoldDataset[T, Tensor], ABC):
    _dataset_parts: tuple[str] = ("input",)  # pyright: ignore[reportIncompatibleVariableOverride]

    @final
    def __getitem__(self, index: int) -> Tensor:
        return self.get_part_element("input", index)
