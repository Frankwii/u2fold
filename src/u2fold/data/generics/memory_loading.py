import multiprocessing
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Hashable, Iterable, final

from torch import Tensor

from u2fold.exceptions.dataset_pairing import DatasetPairingError

from .base import _GenericDataset


class RAMLoadedDataset(_GenericDataset, ABC):
    """A Dataset with information that is fully loaded in CPU memory.

    Each part (as defined in GenericDataset) is loaded into a list.
    """

    # Important to have this be a static method for parallelization.
    @staticmethod
    @abstractmethod
    def _load_element(path: Path) -> Tensor:
        """Load a single element into a Tensor given its path."""

    def _postvalidate_part_pairing(self) -> None:
        """Validates that the different parts of the dataset are paired.

        This is executed *after* loading the elements, so it is possible to
        use their values.

        For instance, for a dataset with input and ground truth images,
        this method could validate whether every input image has the same
        shape as its ground truth.
        """
        pass

    @final
    def _validate_part_pairing(self) -> None:
        """Generic validation of the part pairing.

        Validates whether each part Tensor has the same length (axis 0),
        and calls `_postvalidate_part_pairing`.
        """

        lengths = (
            len(tensor) for tensor in self._indexed_dataset_parts.values()
        )

        if not len(set(lengths)) == 1:
            class_name = type(self).__name__
            errmsg = f"Parts in {class_name} do not have identical length."

            raise DatasetPairingError(errmsg)

    @final
    def __init__(self, dataset_path: Path) -> None:
        self._dataset_path = dataset_path
        self._indexed_dataset_parts: dict[str, list[Tensor]] = {}

        dataset_parts = self._dataset_parts

        for part in dataset_parts:
            part_path = self._dataset_path / part
            self._validate_directory_path(part_path)

        self._prevalidate_part_pairing()

        for part in dataset_parts:
            element_paths = sorted(self._get_part_path(part).iterdir())
            self._indexed_dataset_parts[part] = self._load_elements(
                *element_paths
            )

        self._validate_part_pairing()
        self._postvalidate_part_pairing()

    def _get_part_path(self, part: str) -> Path:
        return self._dataset_path / part

    def __get_part_tensor_slices(self) -> Iterable[tuple[Tensor]]:
        """Returns an iterable with "slices" of paired tensors for different
        parts of the dataset.

        Elements of this iterable are tuples of paired Tensors, that is, they
        containing one Tensor per dataset part. They are yielded in the order
        they are indexed.

        Useful for validation tasks such as checking that paired input and
        ground truth tensors have the same shape.
        """
        return zip(
            *(
                (tensor for tensor in self._indexed_dataset_parts[part])
                for part in self._dataset_parts
            )
        )

    def _map_to_tensor_slices[A](
        self, f: Callable[[Tensor], A]
    ) -> Iterable[Iterable[A]]:
        """Maps the given callable to each tensor slice given by
        "__get_part_tensor_slices."""
        return (
            (f(tensor) for tensor in slice)
            for slice in self.__get_part_tensor_slices()
        )

    def _assert_pairing_homogeneity_of_mapping[A: Hashable](
        self, f: Callable[[Tensor], A]
    ) -> None:
        """Checks that the function "f" is constant across slices as given by
        "__get_part_tensor_slices".
        """
        mapped_slices = self._map_to_tensor_slices(f)

        slice_sets = (set(slice) for slice in mapped_slices)

        for idx, slice_set in enumerate(slice_sets):
            if len(slice_set) != 1:
                errmsg = (
                    f"Paired tensors on index {idx} map to different values"
                    f" of {f}!"
                )

                raise DatasetPairingError(errmsg)

    @final
    def _load_elements(self, *paths: Path) -> list[Tensor]:
        # Loading tasks are parallelized via processes instead of threads
        # since there is often some number-crunching involved in the loading
        # (for instance, RGB conversion for images; so CPU bottleneck there)
        # and PIL is blocking anyway (also in the image case).
        with ProcessPoolExecutor(max_workers=None) as executor:
            dataset_elements = list(executor.map(self._load_element, paths))

        return dataset_elements

    def get_part_element(self, part: str, index: int) -> Tensor:
        return self._indexed_dataset_parts[part][index]

    def __len__(self) -> int:
        return len(next(iter(self._indexed_dataset_parts.values())))


class LazilyLoadedDataset(_GenericDataset, ABC): ...
