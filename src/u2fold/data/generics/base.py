from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import ClassVar, Iterable, final
from torch import Size, Tensor
from torch.utils.data import Dataset

from u2fold.exceptions.dataset_pairing import DatasetPairingError
from u2fold.exceptions.empty_directory import EmptyDirectoryError
from u2fold.utils.singleton_metaclasses import AbstractSingleton


class _GenericDataset(Dataset, ABC, metaclass=AbstractSingleton):
    """Common interface for Dataset classes in this program.

    Subclasses should provide, besides the usual pytorch Dataset
    `__getitem__` and `__len__` dunders, a class attribute `_dataset_parts`
    which specifies how many parts the dataset should have and their names.
    For instance, a labelled dataset could have "input" and "label" parts.

    Each subclass must provide its own validation for the pairing of such
    parts, e.g. checking whether every input has a label or a ground truth.
    """

    _dataset_parts: ClassVar[tuple[str, ...]] = ("input",)

    @abstractmethod
    def _prevalidate_part_pairing(self) -> None:
        """Validates that the different parts of the dataset are paired.

        This is executed *before* loading any of the elements.

        For instance, for a dataset with input and ground truth images,
        this method should validate whether every input image has a ground
        truth (and vice versa).
        """

    @classmethod
    def _validate_directory_path(cls, path: Path) -> None:
        if not (path.exists() and path.is_dir()):
            errmsg = (
                f"Path {path} is not a valid directory under dataset"
                f" {cls.__name__}."
            )

            raise NotADirectoryError(errmsg)

        if next(path.iterdir(), None) is None:
            raise EmptyDirectoryError(path)
