import logging
from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
from typing import ClassVar

import numpy
import PIL.Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from u2fold.exceptions.empty_directory import EmptyDirectoryError
from u2fold.utils.singleton_metaclasses import AbstractSingleton


class U2FoldDataset[T](Dataset, ABC, metaclass=AbstractSingleton):
    """Common interface for Dataset classes in this program.

    Subclasses should provide, besides the usual pytorch Dataset
    `__getitem__` and `__len__` dunders, a class attribute `_dataset_parts`
    which specifies how many parts the dataset should have and their names.
    For instance, a labelled dataset could have "input" and "label" parts.

    Each subclass must provide its own validation for the pairing of such
    parts, e.g. checking whether every input has a label or a ground truth.
    """
    @property
    def _logger(self) -> Logger:
        return logging.getLogger(f"{__name__}|{self.__class__.__name__}")

    @abstractmethod
    def __init__(self, dataset_path: Path) -> None: ...

    _dataset_parts: ClassVar[tuple[str, ...]] = ("input",)

    # Important to have this be a static method for parallelization.
    @staticmethod
    @abstractmethod
    def _load_element(path: Path) -> T:
        """Load a single element into something castable to a Tensor given its
        path.

        This method will be called in multiple processes inside a
        RAMLoadedDataset.
        """
        ...

    def _cast_to_tensor(self, potential_tensor: T) -> Tensor:
        if isinstance(potential_tensor, Tensor):
            return potential_tensor
        if isinstance(potential_tensor, PIL.Image.Image | numpy.ndarray):
            return to_tensor(potential_tensor)
        return Tensor(potential_tensor)

    @abstractmethod
    def get_part_element(self, part: str, index: int) -> Tensor: ...

    @abstractmethod
    def _prevalidate_part_pairing(self) -> None:
        """Validates that the different parts of the dataset are paired.

        This is executed *before* loading any of the elements.

        For instance, for a dataset with input and ground truth images,
        this method should validate whether every input image has a ground
        truth (and vice versa).
        """

    def _validate_directory_path(self, path: Path) -> None:
        if not (path.exists() and path.is_dir()):
            errmsg = (
                f"Path {path} is not a valid directory under dataset"
                f" {self.__class__.__name__}."
            )

            self._logger.error(errmsg)

            raise NotADirectoryError(errmsg)

        if next(path.iterdir(), None) is None:

            self._logger.error(f"Encountered empty directory: {path}.")
            raise EmptyDirectoryError(path)
