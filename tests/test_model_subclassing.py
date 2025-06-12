from dataclasses import dataclass
from typing import Optional

import torch

from u2fold.models.generic import Model, ModelConfig


@dataclass
class FeedForwardConfig(ModelConfig):
    layer_dimensions: list[int]

    def validate(self) -> None:
        super().validate()

        assert len(self.layer_dimensions) >= 2


class FeedForwardBlock(Model[FeedForwardConfig]):
    def __init__(
        self, config: FeedForwardConfig, device: Optional[str] = None
    ) -> None:
        torch.nn.Module.__init__(self)
        layer_dimensions = config.layer_dimensions

        dimension_pairs = (
            (layer_dimensions[i], layer_dimensions[i + 1])
            for i in range(len(layer_dimensions) - 1)
        )

        self.__layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(input_dim, output_dim, device=device)
                for input_dim, output_dim in dimension_pairs
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.__layers:
            x = layer(x)
        return x


def test_subclassing():
    config = FeedForwardConfig(0.2, 0.01, [10, 100, 5])

    model = FeedForwardBlock(config)

    mock_inputs = [torch.rand((10,)), torch.rand((64, 100, 10))]
    expected_shapes = [torch.Size((5,)), torch.Size((64, 100, 5))]

    for idx, mock_input in enumerate(mock_inputs):
        assert model(mock_input).shape == expected_shapes[idx]


def test_skip_init():
    config = FeedForwardConfig(0.5, 0.1, [10, 100, 5])

    # This should not raise an exception
    model = torch.nn.utils.skip_init(FeedForwardBlock, config)

    mock_inputs = [torch.rand((10,)), torch.rand((64, 100, 10))]
    expected_shapes = [torch.Size((5,)), torch.Size((64, 100, 5))]

    for idx, mock_input in enumerate(mock_inputs):
        assert model(mock_input).shape == expected_shapes[idx]


def test_fails_with_wrong_dropout():
    try:
        FeedForwardConfig(2, 0.1, [10, 10])
        errmsg = (
            "Initializing config should fail before this line due to"
            "wrong dropout value (should be between 0 and 1)"
        )
        raise AssertionError(errmsg)
    except ValueError:
        # Ok; should raise this exception
        pass
