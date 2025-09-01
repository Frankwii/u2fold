from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import cast

import PIL.Image as I
import pytest
import torch

from tests.utils.file import TmpFiles
from u2fold.model.common_namespaces import DeterministicComponents
from u2fold.model.neural_network_spec.aunet import AUNetSpec
from u2fold.model.spec import U2FoldSpec
from u2fold.model.train_spec.spec import TrainSpec

type JSON = (
    Mapping[str, "JSON"] | Sequence["JSON"] | Path | str | int | float | bool | None
)


@pytest.fixture
def tmp_img_files() -> Iterator[list[Path]]:
    img = I.new(mode="RGB", size=(100, 100))
    with TmpFiles(Path("img1.png"), Path("img2.png")) as tmp_files:
        for path in tmp_files:
            img.save(path)
        yield tmp_files


@pytest.fixture
def valid_exec_spec(tmp_img_files: list[Path]) -> JSON:
    return {
        "log_level": "info",
        "mode_spec": {
            "mode": "exec",
            "input": tmp_img_files,
            "output_dir": "/home/frank/builds/u2fold/output",
            "override_dir_contents": True,
        },
        "algorithmic_spec": {
            "transmission_map_patch_radius": 8,
            "guided_filter_patch_radius": 15,
            "transmission_map_saturation_coefficient": 0.75,
            "guided_filter_regularization_coefficient": 0.01,
            "step_size": 0.01,
            "greedy_iterations": 3,
            "stages": 3,
            "share_network_weights": True,
        },
        "neural_network_spec": {
            "name": "unet",
            "hidden_layers_activation": {"name": "gelu"},
            "final_residual_activation": {"name": "relu"},
            "pooling": {"method": "max", "kernel_size": 2, "stride": 2},
            "channels_per_layer": [4, 8, 16],
            "sublayers_per_step": 3,
            "unfolded_step_size": 0.01,
        },
    }


@pytest.fixture
def valid_train_spec() -> JSON:
    return {
        "log_level": "info",
        "mode_spec": {
            "mode": "train",
            "losses": [
                {"loss": "ground_truth", "weight": 1.0},
                {"loss": "fidelity", "weight": 0.01},
                {"loss": "color_cosine_similarity", "weight": 1.0},
            ],
            "dataset_spec": {
                "name": "uieb",
                "path": "uieb/processed",
                "n_epochs": 500,
                "batch_size": 4,
            },
            "optimizer_spec": {"optimizer": "adam", "learning_rate": 0.002},
            "learning_rate_scheduler_spec": {
                "scheduler": "step_lr",
                "step_size": 50,
                "factor": 0.5,
            },
        },
        "algorithmic_spec": {
            "transmission_map_patch_radius": 8,
            "guided_filter_patch_radius": 15,
            "transmission_map_saturation_coefficient": 0.75,
            "guided_filter_regularization_coefficient": 0.01,
            "step_size": 0.01,
            "greedy_iterations": 3,
            "stages": 3,
            "share_network_weights": False,
        },
        "neural_network_spec": {
            "name": "unet",
            "hidden_layers_activation": {"name": "gelu"},
            "final_residual_activation": {"name": "relu"},
            "pooling": {"method": "max", "kernel_size": 2, "stride": 2},
            "channels_per_layer": [8, 16, 32],
            "sublayers_per_step": 3,
            "unfolded_step_size": 0.01,
        },
    }


def test_train_spec_initialization_from_python(valid_train_spec):
    U2FoldSpec.model_validate(valid_train_spec)


def test_exec_spec_initialization_from_python(valid_exec_spec):
    U2FoldSpec.model_validate(valid_exec_spec)


import torch.nn as nn


class MockUnet(nn.Module):
    def __init__(self, conf: AUNetSpec) -> None:
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


def test_train_spec_instantiation(valid_train_spec):
    spec = U2FoldSpec.model_validate(valid_train_spec)
    train_spec = cast(TrainSpec, spec.mode_spec)

    model = MockUnet(spec.neural_network_spec)
    optimizer = train_spec.optimizer_spec.instantiate(model.parameters())
    scheduler = train_spec.learning_rate_scheduler_spec.instantiate(optimizer)
    losses = [loss.instantiate() for loss in train_spec.losses]


def test_train_forward_pass(valid_train_spec):
    spec = U2FoldSpec.model_validate(valid_train_spec)
    train_spec = cast(TrainSpec, spec.mode_spec)

    model = MockUnet(spec.neural_network_spec)
    optimizer = train_spec.optimizer_spec.instantiate(model.parameters())
    scheduler = train_spec.learning_rate_scheduler_spec.instantiate(optimizer)
    losses = [loss.instantiate() for loss in train_spec.losses]

    mock_input = torch.Tensor([1.0]).reshape(1, 1, 1, 1)
    model_output = model(mock_input)

    from u2fold.model.common_namespaces import ForwardPassResult

    result = ForwardPassResult(
        primal_variable_history=[model_output] * 2,
        kernel_history=[model_output] * 2,
        deterministic_components=DeterministicComponents(
            fidelity=torch.Tensor([1.0]).reshape_as(mock_input),
            transmission_map=torch.Tensor([1.0]).reshape_as(mock_input),
            background_light=torch.Tensor([1.0]).reshape_as(mock_input),
        ),
    )

    loss = sum(
        loss(result, torch.Tensor([1.0]).reshape_as(mock_input)) for loss in losses
    )
