import pytest

from u2fold.cli_parsing.argparse import build_parser
from u2fold.config_parsing.generic import parse_model_arguments
from u2fold.models.unet import ConfigUNet


def test_model_config_parsing():
    parser = build_parser()

    cli_args = [
        "train",
        "--dropout",
        "0.1",
        "-n",
        "10",
        "--loss-strategy",
        "intermediate",
        "unet",
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

    config = parse_model_arguments(args)

    expected_config = ConfigUNet(
        channels_per_layer = [256, 512, 256],
        sublayers_per_step = 3,
        pooling = "max",
        activation = "gelu",
        dropout = 0.1
    )

    assert config == expected_config

def test_model_config_validation():
    parser = build_parser()

    cli_args = [
        "train",
        "--dropout",
        "2",
        "-n",
        "10",
        "--loss-strategy",
        "intermediate",
        "unet",
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

    with pytest.raises(ValueError, match="Dropout must be between 0 and 1."):
        parse_model_arguments(args)

