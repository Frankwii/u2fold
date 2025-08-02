from pathlib import Path
from typing import override

import PIL.Image
from PIL.Image import Image

from u2fold.data.dataset_generics import GroundTruthDataset, RAMLoadedDataset
from u2fold.exceptions.dataset_pairing import DatasetPairingError
from u2fold.utils.track import tag


@tag("data/dataset/uieb")
class UIEBDataset(RAMLoadedDataset[Image], GroundTruthDataset[Image]):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """Subclass of PyTorch's Dataset for the UIEB dataset.

    Importantly, this loads the full (preprocessed) dataset into memory as
    it should occupy less than half a gigabyte. This is implemented via
    inheritance.
    """

    @override
    def _postvalidate_part_pairing(self) -> None:
        """Check whether corresponding input and ground truth Tensors have
        the same shape.
        """
        self.assert_pairing_homogeneity_of_mapping(lambda tensor: tensor.shape)

    @override
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

            self._logger.error(errmsg)

            raise DatasetPairingError(errmsg)

    def __get_names_of_directory(self, dir: Path) -> set[str]:
        return set(file.name for file in dir.iterdir())

    @override
    @staticmethod
    def _load_element(path: Path) -> Image:
        return PIL.Image.open(path).convert("RGB")
