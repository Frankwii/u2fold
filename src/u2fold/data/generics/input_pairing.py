from abc import ABC, abstractmethod

from torch import Tensor
from .base import _GenericDataset

class GroundTruthDataset(_GenericDataset, ABC):
    _dataset_parts = ("input", "ground_truth")

    @abstractmethod
    def _get_ground_truth_element(self, index: int) -> Tensor: ...

    @abstractmethod
    def _get_input_element(self, index: int) -> Tensor: ...

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return (
            self._get_input_element(index),
            self._get_ground_truth_element(index),
        )

class UnsupervisedDataset(_GenericDataset, ABC):
    @abstractmethod
    def __getitem__(self, index: int) -> Tensor: ...

