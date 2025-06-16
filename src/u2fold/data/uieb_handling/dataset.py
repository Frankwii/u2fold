from pathlib import Path

import PIL.Image
from torch import Tensor
from torchvision.transforms.functional import to_tensor

from u2fold.data.generics import GroundTruthDataset, RAMLoadedDataset
from u2fold.exceptions.dataset_pairing import DatasetPairingError


class UIEBDataset(RAMLoadedDataset, GroundTruthDataset):
    """Subclass of PyTorch's Dataset for the UIEB dataset.

    Importantly, this loads the full (preprocessed) dataset into memory as
    it should occupy less than half a gigabyte. This is implemented via
    inheritance.
    """
    def _get_ground_truth_element(self, index: int) -> Tensor:
        return self.get_part_element("ground_truth", index)

    def _get_input_element(self, index: int) -> Tensor:
        return self.get_part_element("input", index)

    def _pad_tensors_to_common_shape(
        self, tensors: list[Tensor]
    ) -> list[Tensor]:
        return tensors

    def _postvalidate_part_pairing(self) -> None:
        """Check whether corresponding input and ground truth Tensors have
        the same shape.
        """
        self._check_size_homogeneity_among_parts()

    def _prevalidate_part_pairing(self) -> None:
        input_names = self.__get_names_of_directory(
            self._get_part_path("input")
        )

        ground_truth_names = self.__get_names_of_directory(
            self._get_part_path("ground_truth")
        )

        differences = input_names ^ ground_truth_names

        if len(differences) > 0:
            errmsg = (
                f"Input and ground truth images do not match: Differences:"
                f"\n {differences}."
            )

            raise DatasetPairingError(errmsg)

    def __get_names_of_directory(self, dir: Path) -> set[str]:
        return set(file.name for file in dir.iterdir())

    @staticmethod
    def _load_element(path: Path) -> Tensor:
        return to_tensor(PIL.Image.open(path).convert("RGB"))
