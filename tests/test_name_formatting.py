from dataclasses import dataclass

from u2fold.models.generic import ModelConfig
from u2fold.utils.name_conversions import cli_to_snake, snake_to_cli


@dataclass
class FeedForwardConfig(ModelConfig):
    layer_dimensions: list[int]

    def validate(self) -> None:
        super().validate()

        assert len(self.layer_dimensions) >= 2


def test_feedforward_name():
    conf = FeedForwardConfig(0.5, 0.05, [10, 2, 5])

    assert conf.format_self() == "unfoldedStepSize_0.05__layerDimensions_10-2-5"


@dataclass
class MockConfig(ModelConfig):
    foo_bar: int
    bar: list[str]


def test_mock_name():
    conf = MockConfig(0, 0.1, foo_bar=1_000, bar=["baz_bar", "b"])

    assert (
        conf.format_self() == "unfoldedStepSize_0.1__fooBar_1000__bar_bazBar-b"
    )


def get_conversion_examples():
    snake_case = ["snake_case", "layer_channels", "dropout"]

    cli = ["--snake-case", "--layer-channels", "--dropout"]

    return snake_case, cli


def test_cli_to_snake():
    inputs, expected = get_conversion_examples()

    outputs = [snake_to_cli(i) for i in inputs]

    assert outputs == expected


def test_snake_to_cli():
    expected, inputs = get_conversion_examples()

    outputs = [cli_to_snake(i) for i in inputs]

    assert outputs == expected
