from abc import ABC
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Hashable, Iterable, final

from torch import Tensor

from u2fold.exceptions.dataset_pairing import DatasetPairingError

from .base import U2FoldDataset


class RAMLoadedDataset[T](U2FoldDataset[T], ABC):
    """A Dataset with information that is fully loaded in CPU memory.

    Each part (as defined in GenericDataset) is loaded into a list.
    """

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
    def __validate_part_pairing(self) -> None:
        """Generic validation of the part pairing.

        Validates whether each part Tensor has the same length (axis 0),
        and calls `_postvalidate_part_pairing`.
        """

        lengths = (
            len(tensor) for tensor in self.__indexed_dataset_parts.values()
        )

        if not len(set(lengths)) == 1:
            class_name = type(self).__name__
            errmsg = f"Parts in {class_name} do not have identical length."

            self._logger.error(
                f"Encountered error while validating dataset contents: {errmsg}"
            )

            raise DatasetPairingError(errmsg)

    @final
    def __init__(self, dataset_path: Path) -> None:
        self._dataset_path = dataset_path
        self.__indexed_dataset_parts: dict[str, list[Tensor]] = {}

        dataset_parts = self._dataset_parts

        part_paths = [self._get_part_path(part) for part in dataset_parts]

        self._logger.info(
            f"Checking paths for corresponding dataset parts: {part_paths}..."
        )
        for part in dataset_parts:
            part_path = self._get_part_path(part)
            self._validate_directory_path(part_path)

        self._prevalidate_part_pairing()

        self._logger.info("Loading dataset parts...")
        for part in dataset_parts:
            self._logger.info(f"Loading part `{part}`.")
            element_paths = sorted(self._get_part_path(part).iterdir())
            self.__indexed_dataset_parts[part] = self.__load_elements(
                *element_paths
            )
            self._logger.debug(f"Successfully loaded part `{part}`.")

        self._logger.info("Dataset loaded. Validating contents...")
        self.__validate_part_pairing()
        self._postvalidate_part_pairing()

        self._logger.debug("Successfully loaded dataset!")

    @final
    def _get_part_path(self, part: str) -> Path:
        return self._dataset_path / part

    @final
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
                (tensor for tensor in self.__indexed_dataset_parts[part])
                for part in self._dataset_parts
            )
        )

    @final
    def map_to_tensor_slices[A](
        self, f: Callable[[Tensor], A]
    ) -> Iterable[Iterable[A]]:
        """Maps the given callable to each tensor slice given by
        "__get_part_tensor_slices."""
        return (
            (f(tensor) for tensor in slice)
            for slice in self.__get_part_tensor_slices()
        )

    @final
    def assert_pairing_homogeneity_of_mapping[A: Hashable](
        self, f: Callable[[Tensor], A]
    ) -> None:
        """Checks that the function "f" is constant across slices as given by
        "__get_part_tensor_slices".
        """
        mapped_slices = self.map_to_tensor_slices(f)

        slice_sets = (set(slice) for slice in mapped_slices)

        for idx, slice_set in enumerate(slice_sets):
            if len(slice_set) != 1:
                errmsg = (
                    f"Paired tensors on index {idx} map to different values"
                    f" of {f}!"
                )

                raise DatasetPairingError(errmsg)

    @final
    def __load_elements(self, *paths: Path) -> list[Tensor]:
        # Loading tasks are parallelized via processes instead of threads
        # since there is often some number-crunching involved in the loading
        # (for instance, RGB conversion for images; so CPU bottleneck there)
        # and PIL is blocking anyway (also in the image case).
        with ProcessPoolExecutor() as executor:
            images = list(executor.map(self._load_element, paths))

        return [self._cast_to_tensor(image) for image in images]

    @final
    def get_part_element(self, part: str, index: int) -> Tensor:
        return self.__indexed_dataset_parts[part][index]

    @final
    def __len__(self) -> int:
        return len(next(iter(self.__indexed_dataset_parts.values())))


class LazilyLoadedDataset[T](U2FoldDataset[T], ABC):
    """A Dataset with elements that are loaded each time they are requested."""

    @final
    def __init__(self, dataset_path: Path) -> None:
        self.__dataset_path = dataset_path

        self.__indexed_paths: dict[str, list[Path]] = {}

        for part in self._dataset_parts:
            part_path = self.__dataset_path / part
            self._validate_directory_path(part_path)
            self.__indexed_paths[part] = list(part_path.iterdir())

        self._prevalidate_part_pairing()

    @final
    def _prevalidate_part_pairing(self) -> None:
        part_lengths = set(
            len(part_paths) for part_paths in self.__indexed_paths.values()
        )

        if len(part_lengths) != 1:
            paths = [
                self.__dataset_path / path
                for path in self.__indexed_paths.keys()
            ]
            errmsg = f"Given paths have varying amounts of elements! {paths}."
            raise DatasetPairingError(errmsg)

    @final
    def __len__(self) -> int:
        return len(next(iter(self.__indexed_paths.values())))

    @final
    def get_part_element(self, part: str, index: int) -> Tensor:
        return self._cast_to_tensor(
            self._load_element(self.__indexed_paths[part][index])
        )
