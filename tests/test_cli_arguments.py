from typing import NamedTuple

from u2fold import build_parser
from u2fold.cli_parsing import CLIArgument


def test_type_inheritance_type_preservation():
    class MyCLIArgument(CLIArgument[int]):
        def long_name(self) -> str: ...
        def _validate_value(self, value: int) -> None: ...
        def help(self) -> str: ...

    arg = MyCLIArgument()
    assert arg.value_type() is int

    class MyDerivedCLIArgument(MyCLIArgument): ...

    arg = MyDerivedCLIArgument()
    assert arg.value_type() is int


def test_indirect_type_tracking():
    class MyParametrizedCLIArgument[U: object](CLIArgument[U]):
        def long_name(self) -> str: ...
        def _validate_value(self, value: U) -> None: ...
        def help(self) -> str: ...

    class MyConcreteCLIArgument(MyParametrizedCLIArgument[int]): ...

    arg = MyConcreteCLIArgument()
    try:
        arg.value_type()
        raise AssertionError("Should not be possible to infer type indirectly.")
    except TypeError as _:
        pass  # ok


def test_unet_parsing():
    parser = build_parser()

    cli_args = [
        "-p",
        "4",
        "--regularization-coefficient",
        "0.5",
        "--saturation-coefficient",
        "1.2",
        "train",
        "-b",
        "16",
        "--dropout",
        "0.1",
        "-n",
        "10",
        "--loss-strategy",
        "intermediate",
        "unet",
        "--unfolded-step-size",
        "0.02",
        "--pooling",
        "max",
        "--activation",
        "gelu",
        "--channels-per-layer",
        "256",
        "512",
        "256",
        "--sublayers-per-step",
        "3",
    ]

    args = parser.parse_args(cli_args)

    class ExpectedNamespace(NamedTuple):
        patch_radius = 4
        regularization_coefficient = 0.5
        saturation_coefficient = 1.2
        unfolded_step_size = 0.02
        batch_size = 16
        step_size = 0.01
        mode = "train"
        dropout = 0.1
        n_epochs = 10
        loss_strategy = "intermediate"
        model = "unet"
        pooling = "max"
        activation = "gelu"
        channels_per_layer = [256, 512, 256]
        sublayers_per_step = 3

    expected_namespace = ExpectedNamespace()
    attrs = [
        "patch_radius",
        "saturation_coefficient",
        "regularization_coefficient",
        "mode",
        "batch_size",
        "dropout",
        "n_epochs",
        "loss_strategy",
        "model",
        "pooling",
        "activation",
        "channels_per_layer",
        "sublayers_per_step",
        "step_size",
        "unfolded_step_size",
    ]

    for attr in attrs:
        assert getattr(expected_namespace, attr) == getattr(args, attr)
