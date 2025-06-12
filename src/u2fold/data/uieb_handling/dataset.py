from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import PIL.Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from u2fold.utils.singleton_metaclasses import Singleton


def _load_image(image_path: Path) -> Tensor:
    return to_tensor(PIL.Image.open(image_path).convert("RGB"))

class UIEBDataset(Dataset, metaclass=Singleton):
    """Subclass of PyTorch's Dataset for the UIEB dataset.

    Importantly, this loads the full (preprocessed) dataset into memory as
    it should occupy less than half a gigabyte.
    """

    def __init__(self, uieb_path: Path) -> None:
        input_images_path = uieb_path / "processed" / "input"
        ground_truth_images_path = uieb_path / "processed" / "ground_truth"

        self.__validate_image_pairing(
            input_images_path, ground_truth_images_path
        )
        image_names = self.__get_names_of_directory(input_images_path)

        self.__input_images, self.__ground_truth_images = self.__load_uieb(
            image_names, input_images_path, ground_truth_images_path
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        input = self.__input_images[index]
        ground_truth = self.__ground_truth_images[index]

        return input, ground_truth

    def __len__(self) -> int:
        return len(self.__input_images)

    def __load_uieb(
        self,
        image_names: set[str],
        input_images_path: Path,
        ground_truth_images_path: Path,
    ) -> tuple[Tensor, Tensor]:
        input_images = self.__load_images(image_names, input_images_path)
        ground_truth_images = self.__load_images(
            image_names, ground_truth_images_path
        )

        return input_images, ground_truth_images

    def __load_images(self, names: Iterable[str], images_path: Path) -> Tensor:

        image_paths = [images_path / name for name in names]

        # Parallelize a few of the loading tasks. It is better to use processes
        # than threads since there is some number-crunching involved in the RGB
        # conversion (CPU bottleneck there) and PIL is blocking anyway.
        # More or less, spawn a process for each 50 images.
        with ProcessPoolExecutor(max_workers=None) as executor:
            images = list(
                executor.map(_load_image, image_paths, chunksize=50)
            )

        return torch.stack(images)

    def __get_names_of_directory(self, dir: Path) -> set[str]:
        return set(file.name for file in dir.iterdir())

    def __validate_image_pairing(
        self,
        input_images_path: Path,
        ground_truth_images_path: Path,
    ) -> None:
        """
        Validates that each input image has an associated ground truth
        and vice versa.
        """

        input_names = self.__get_names_of_directory(input_images_path)
        ground_truth_names = self.__get_names_of_directory(
            ground_truth_images_path
        )

        differences = input_names.symmetric_difference(ground_truth_names)

        if len(differences) > 0:
            errmsg = (
                f"Input and ground truth images do not match: Differences:"
                f"\n {differences}."
            )

            raise ValueError(errmsg)
