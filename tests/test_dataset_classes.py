from pathlib import Path

import pytest
import torch
import gc
from torch import Tensor

from u2fold.data.generics.combinations import RAMLoadedGroundTruthDataset
from u2fold.exceptions.dataset_pairing import DatasetPairingError
from u2fold.utils.singleton_metaclasses import AbstractSingleton

from .utils_testing import (
    TemporaryDir,
    check_numeric_tensor_collection_equality,
)


def write_tensors(dir: Path, tensors: list[Tensor]) -> None:
    dir.mkdir(exist_ok=True)
    for idx, tensor in enumerate(tensors):
        path = dir / f"{idx}.pt"
        torch.save(tensor, path)


def load_tensors(dir: Path) -> list[Tensor]:
    return [torch.load(file) for file in dir.iterdir()]


def write_mock_ground_truth_dataset(
    tmp_dir: Path,
    mock_input_tensors: list[Tensor],
    mock_ground_truth_tensors: list[Tensor],
) -> tuple[Path, Path]:
    input_path = tmp_dir / "input"
    ground_truth_path = tmp_dir / "ground_truth"
    write_tensors(input_path, mock_input_tensors)
    write_tensors(ground_truth_path, mock_ground_truth_tensors)

    return (input_path, ground_truth_path)


def get_mock_tensors(shapes: list[tuple[int, ...]]) -> list[Tensor]:
    return [torch.randn(shape) for shape in shapes]


def get_mock_homogenous_data() -> tuple[list[Tensor], list[Tensor]]:
    input_shapes = ground_truth_shapes = [
        (3, 100, 100),
        (3, 120, 80),
        (3, 300, 50),
    ]

    mock_inputs = get_mock_tensors(input_shapes)
    mock_gts = get_mock_tensors(ground_truth_shapes)

    return mock_inputs, mock_gts


def get_mock_inhomogenous_channel_data() -> tuple[list[Tensor], list[Tensor]]:
    input_shapes = [
        (3, 100, 100),
        (3, 120, 80),
        (3, 300, 50)
    ]

    ground_truth_shapes = [
        (3, 100, 100),
        (1, 120, 80),
        (3, 300, 50)
    ]

    mock_inputs = get_mock_tensors(input_shapes)
    mock_gts = get_mock_tensors(ground_truth_shapes)

    return mock_inputs, mock_gts


def get_mock_inhomogenous_shape_data() -> tuple[list[Tensor], list[Tensor]]:
    input_shapes = [
        (3, 100, 100),
        (3, 120, 80),
        (3, 300, 50),
    ]

    ground_truth_shapes = [
        (3, 99, 100),
        (3, 120, 80),
        (3, 300, 50),
    ]

    mock_inputs = get_mock_tensors(input_shapes)
    mock_gts = get_mock_tensors(ground_truth_shapes)

    return mock_inputs, mock_gts


class MockGroundTruthDataset(RAMLoadedGroundTruthDataset):
    @staticmethod
    def _load_element(path: Path) -> Tensor:
        return torch.load(path)

    def _prevalidate_part_pairing(self) -> None:
        input_dir = self._dataset_path / "input"
        ground_truth_dir = self._dataset_path / "ground_truth"

        input_names = [file.name for file in input_dir.iterdir()]
        ground_truth_names = [file.name for file in ground_truth_dir.iterdir()]

        assert list(input_names) == list(ground_truth_names)

    def _postvalidate_part_pairing(self) -> None:
        self._assert_pairing_homogeneity_of_mapping(lambda tensor: tensor.shape)


def test_ground_truth_dataset1():
    mock_inputs, mock_gts = get_mock_homogenous_data()
    with TemporaryDir("mock_dataset") as dataset_dir:
        write_mock_ground_truth_dataset(dataset_dir, mock_inputs, mock_gts)

        mock_dataset = MockGroundTruthDataset(dataset_dir)

        assert len(mock_dataset) == len(mock_inputs)
        dataset_parts = getattr(mock_dataset, "_indexed_dataset_parts")
        assert dataset_parts is not None
        input_tensors = dataset_parts["input"]
        ground_truth_tensors = dataset_parts["ground_truth"]

        check_numeric_tensor_collection_equality(input_tensors, mock_inputs)
        check_numeric_tensor_collection_equality(ground_truth_tensors, mock_gts)

        del mock_dataset
        del AbstractSingleton._instances[MockGroundTruthDataset]
        gc.collect()


def test_should_raise_if_impossible_channel_pairing():
    mock_inputs1, mock_gts1 = get_mock_inhomogenous_channel_data()

    with pytest.raises(DatasetPairingError, match=" 1 "):
        with TemporaryDir("mock_dataset2") as dataset_dir:
            write_mock_ground_truth_dataset(
                dataset_dir, mock_inputs1, mock_gts1
            )
            mock_dataset = MockGroundTruthDataset(dataset_dir)

            del mock_dataset
            del AbstractSingleton._instances[MockGroundTruthDataset]
            gc.collect()


def test_should_raise_if_impossible_shape_pairing():
    mock_inputs2, mock_gts2 = get_mock_inhomogenous_shape_data()

    with pytest.raises(DatasetPairingError, match=" 0 "):
        with TemporaryDir("mock_dataset3") as dataset_dir:
            write_mock_ground_truth_dataset(
                dataset_dir, mock_inputs2, mock_gts2
            )
            mock_dataset = MockGroundTruthDataset(dataset_dir)

            del mock_dataset
            del AbstractSingleton._instances[MockGroundTruthDataset]
            gc.collect()

