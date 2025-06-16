from abc import ABC
from typing import final

from torch import Tensor
from u2fold.data.generics.input_pairing import GroundTruthDataset
from u2fold.data.generics.memory_loading import RAMLoadedDataset


class RAMLoadedGroundTruthDataset(RAMLoadedDataset, GroundTruthDataset, ABC):
    @final
    def _get_ground_truth_element(self, index: int) -> Tensor:
        return self.get_part_element("ground_truth", index)

    @final
    def _get_input_element(self, index: int) -> Tensor:
        return self.get_part_element("input", index)
