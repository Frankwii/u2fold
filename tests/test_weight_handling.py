import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import torch
from torch import nn

from u2fold.models.generic import Model, ModelConfig
from u2fold.models.weight_handling import (
    ModelInitBundle,
    TrainWeightHandler,
    U2FoldWeightTuple,
)
from u2fold.utils.nested_map import nested_map

mock_path = Path("/tmp/u2fold/mock_model_weights")
if mock_path.exists() and mock_path.is_dir():
    shutil.rmtree(mock_path)

mock_path.mkdir(parents=True, exist_ok=True)


def generate_sorted_file_names(min: int, max: int) -> list[str]:
    max_digits = int(math.log10(max)) + 1
    return [str(i).rjust(max_digits, "0") for i in range(min, max + 1)]


def build_mock_filetree(greedy_iters: int, stages: int, mock_path=mock_path):
    greedy_iter_names = generate_sorted_file_names(1, greedy_iters)
    stage_names = generate_sorted_file_names(1, stages)

    return [
        U2FoldWeightTuple(
            image=[
                mock_path.joinpath(f"{greedy_iter}/image/{weigth_file}.pt")
                for weigth_file in stage_names
            ],
            kernel=[
                mock_path.joinpath(f"{greedy_iter}/kernel/{weight_file}.pt")
                for weight_file in stage_names
            ],
        )
        for greedy_iter in greedy_iter_names
    ]


def test_mocking():
    greedy_iters = [1, 2, 5, 10, 20, 101]
    stages = [2, 5, 10, 100]

    for greedy_iter_n in greedy_iters:
        for stage_n in stages:
            mock_filetree = build_mock_filetree(greedy_iter_n, stage_n)

            assert len(mock_filetree) == greedy_iter_n
            for greedy_iter in range(greedy_iter_n):
                assert len(mock_filetree[greedy_iter].kernel) == stage_n
                assert len(mock_filetree[greedy_iter].image) == stage_n

    greedy_iters = 1
    stages = 10

    mock_filetree = build_mock_filetree(greedy_iters, stages)

    assert mock_filetree[0].image[0] == mock_path.joinpath("1/image/01.pt")
    assert mock_filetree[0].kernel[0] == mock_path.joinpath("1/kernel/01.pt")

    assert mock_filetree[0].image[1] == mock_path.joinpath("1/image/02.pt")
    assert mock_filetree[0].kernel[1] == mock_path.joinpath("1/kernel/02.pt")

    assert mock_filetree[0].image[9] == mock_path.joinpath("1/image/10.pt")
    assert mock_filetree[0].kernel[9] == mock_path.joinpath("1/kernel/10.pt")


def test_filetree_building(mock_path=mock_path):
    greedy_iters = 10
    stages = 2

    mock_filetree = build_mock_filetree(greedy_iters, stages)

    print(mock_filetree)

    handler = TrainWeightHandler(mock_path, greedy_iters, stages)

    filetree = handler._filetree

    assert filetree == mock_filetree


def save_mock_weights(greedy_iters: int, stages: int):
    handler = TrainWeightHandler(mock_path, greedy_iters, stages)

    @dataclass
    class MockConfig(ModelConfig):
        ...

    class MockModel(Model[MockConfig]):
        def __init__(
            self, config: MockConfig, device: Optional[str] = None
        ) -> None:
            nn.Module.__init__(self)
            self.fc1 = nn.Linear(10, 10, device=device)

    model = MockModel(MockConfig(0.5), None)

    filetree = handler._filetree

    nested_map(
        lambda file: handler.save_weights(cast(Path, file), model), filetree
    )

    return model.state_dict()


def test_weight_saving(mock_path=mock_path):
    save_mock_weights(10, 10)

    assert mock_path.joinpath("01/kernel/01.pt").exists
    assert mock_path.joinpath("10/kernel/10.pt").exists


def sqnorm(t: torch.Tensor) -> float:
    return float(torch.sum(t**2))


def test_weight_loading(mock_path=mock_path):
    state_dict = save_mock_weights(10, 10)

    handler = TrainWeightHandler(mock_path, 10, 10)

    @dataclass
    class MockConfig2(ModelConfig):
        ...

    class MockModel2(Model[MockConfig2]):
        def __init__(self, config: MockConfig2, device: Optional[str]) -> None:
            nn.Module.__init__(self)
            self.fc1 = nn.Linear(10, 10)

    config = MockConfig2(0.2)

    model_bundle = ModelInitBundle(config, MockModel2, None)

    models = handler.load_models(model_bundle, model_bundle)

    assert set(models[0].image[0].state_dict().keys()) == set(state_dict.keys())
    assert set(models[9].image[9].state_dict().keys()) == set(state_dict.keys())

    for key in state_dict.keys():
        assert (
            sqnorm(models[9].image[9].state_dict()[key] - state_dict[key])
            < 1e-9
        )
        assert (
            sqnorm(models[9].image[9].state_dict()[key] - state_dict[key])
            < 1e-9
        )
