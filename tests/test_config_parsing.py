from dataclasses import fields
from pathlib import Path

import pytest
import torch

from u2fold.cli_parsing.argparse import build_parser
from u2fold.config_parsing.config_dataclasses import TrainConfig, TransmissionMapEstimationConfig
from u2fold.config_parsing.validation_and_parsing import (
    __parse_model_arguments,
    parse_and_validate_config,
)
from u2fold.models.unet import ConfigUNet


def test_model_config_parsing():
    parser = build_parser()

    cli_args = [
        "--patch-radius",
        "6",
        "--saturation-coefficient",
        "0.9",
        "--regularization-coefficient",
        "0.1",
        "train",
        "-b",
        "1",
        "--dropout",
        "0.1",
        "-n",
        "10",
        "--loss-strategy",
        "intermediate",
        "unet",
        "--unfolded-step-size",
        "0.01",
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

    config = __parse_model_arguments(args)

    expected_config = ConfigUNet(
        channels_per_layer=[256, 512, 256],
        sublayers_per_step=3,
        pooling="max",
        activation="gelu",
        dropout=0.1,
        unfolded_step_size=0.01,
    )

    assert config == expected_config


def test_model_config_validation():
    parser = build_parser()

    cli_args = [
        "--patch-radius",
        "6",
        "--saturation-coefficient",
        "0.9",
        "--regularization-coefficient",
        "0.1",
        "train",
        "-b",
        "1",
        "--dropout",
        "2",
        "-n",
        "10",
        "--loss-strategy",
        "intermediate",
        "unet",
        "--unfolded-step-size",
        "0.01",
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
        __parse_model_arguments(args)


def test_full_config_parsing():
    parser = build_parser()

    cli_args = [
        "--patch-radius",
        "6",
        "--saturation-coefficient",
        "0.9",
        "--regularization-coefficient",
        "0.1",
        "--step-size",
        "0.1",
        "--log-level",
        "info",
        "--weight-dir",
        "/tmp/path/to/weights",
        "--log-dir",
        "/tmp/path/to/logs",
        "train",
        "-b",
        "10",
        "--dataset-dir",
        "/tmp/path/to/dataset",
        "--dropout",
        "0.1",
        "-n",
        "10",
        "--loss-strategy",
        "intermediate",
        "unet",
        "--unfolded-step-size",
        "0.01",
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

    weight_mock_path = Path("/tmp/path/to/weights")
    log_mock_path = Path("/tmp/path/to/logs")
    dataset_mock_path = Path("/tmp/path/to/dataset")

    weight_mock_path.mkdir(parents=True, exist_ok=True)
    log_mock_path.mkdir(parents=True, exist_ok=True)
    dataset_mock_path.mkdir(parents=True, exist_ok=True)

    args = parser.parse_args(cli_args)

    config = parse_and_validate_config(args)

    assert isinstance(config, TrainConfig)

    expected_model_config = ConfigUNet(
        channels_per_layer=[256, 512, 256],
        sublayers_per_step=3,
        pooling="max",
        activation="gelu",
        dropout=0.1,
        unfolded_step_size=0.01,
    )

    tmc = TransmissionMapEstimationConfig(
        patch_radius=6,
        regularization_coefficient=0.1,
        saturation_coefficient=0.9
    )

    model_config_path = expected_model_config.format_self(tmc)

    expected_config = TrainConfig(
        transmission_map_estimation_config=tmc,
        model_name="unet",
        log_level="info",
        batch_size=10,
        weight_dir=Path("/tmp/path/to/weights") / "unet" / model_config_path,
        execution_log_dir=Path("/tmp/path/to/logs")
        / "execution"
        / "unet"
        / model_config_path,
        tensorboard_log_dir=Path("/tmp/path/to/logs")
        / "tensorboard"
        / "unet"
        / model_config_path,
        dataset_dir=Path("/tmp/path/to/dataset"),
        model_config=expected_model_config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        loss_strategy="intermediate",
        n_epochs=10,
        step_size=0.1,
        unfolded_step_size=0.01,
    )

    assert config == expected_config


def test_should_raise_if_incorrect_layer_sizes():
    parser = build_parser()

    cli_args = [
        "--patch-radius",
        "6",
        "--saturation-coefficient",
        "0.9",
        "--regularization-coefficient",
        "0.1",
        "--log-level",
        "info",
        "--weight-dir",
        "/tmp/path/to/weights",
        "--log-dir",
        "/tmp/path/to/logs",
        "train",
        "-b",
        "64",
        "--dataset-dir",
        "/tmp/path/to/dataset",
        "--dropout",
        "0.1",
        "-n",
        "10",
        "--loss-strategy",
        "intermediate",
        "unet",
        "--unfolded-step-size",
        "0.01",
        "--pooling",
        "max",
        "--activation",
        "gelu",
        "--channels-per-layer",
        "3",
        "256",
        "512",
        "1",
        "3",
        "--sublayers-per-step",
        "3",
    ]

    weight_mock_path = Path("/tmp/path/to/weights")
    log_mock_path = Path("/tmp/path/to/logs")
    dataset_mock_path = Path("/tmp/path/to/dataset")

    weight_mock_path.mkdir(parents=True, exist_ok=True)
    log_mock_path.mkdir(parents=True, exist_ok=True)
    dataset_mock_path.mkdir(parents=True, exist_ok=True)

    args = parser.parse_args(cli_args)

    with pytest.raises(
        ValueError, match="Invalid number of channels per UNet layer."
    ):
        config = parse_and_validate_config(args)

        assert isinstance(config, TrainConfig)


def test_should_raise_if_invalid_sublayers_per_step():
    parser = build_parser()

    cli_args = [
        "--patch-radius",
        "6",
        "--saturation-coefficient",
        "0.9",
        "--regularization-coefficient",
        "0.1",
        "--log-level",
        "info",
        "--weight-dir",
        "/tmp/path/to/weights",
        "--log-dir",
        "/tmp/path/to/logs",
        "train",
        "--dataset-dir",
        "/tmp/path/to/dataset",
        "--dropout",
        "0.1",
        "-b",
        "4",
        "-n",
        "10",
        "--loss-strategy",
        "intermediate",
        "unet",
        "--unfolded-step-size",
        "0.01",
        "--pooling",
        "max",
        "--activation",
        "gelu",
        "--channels-per-layer",
        "3",
        "256",
        "512",
        "256",
        "3",
        "--sublayers-per-step",
        "-1",
    ]

    weight_mock_path = Path("/tmp/path/to/weights")
    log_mock_path = Path("/tmp/path/to/logs")
    dataset_mock_path = Path("/tmp/path/to/dataset")

    weight_mock_path.mkdir(parents=True, exist_ok=True)
    log_mock_path.mkdir(parents=True, exist_ok=True)
    dataset_mock_path.mkdir(parents=True, exist_ok=True)

    args = parser.parse_args(cli_args)

    with pytest.raises(
        ValueError, match="Insufficient sublayers per UNet step."
    ):
        config = parse_and_validate_config(args)

        assert isinstance(config, TrainConfig)


def test_should_raise_if_incorrect_batch_sizes():
    parser = build_parser()

    negative_test_cases = [0, -1, -10]

    for case in negative_test_cases:
        cli_args = [
            "--patch-radius",
            "6",
            "--saturation-coefficient",
            "0.9",
            "--regularization-coefficient",
            "0.1",
            "--log-level",
            "info",
            "--weight-dir",
            "/tmp/path/to/weights",
            "--log-dir",
            "/tmp/path/to/logs",
            "train",
            "--dataset-dir",
            "/tmp/path/to/dataset",
            "--dropout",
            "0.1",
            "-b",
            f"{case}",
            "-n",
            "10",
            "--loss-strategy",
            "intermediate",
            "unet",
            "--unfolded-step-size",
            "0.01",
            "--pooling",
            "max",
            "--activation",
            "gelu",
            "--channels-per-layer",
            "3",
            "256",
            "512",
            "256",
            "3",
            "--sublayers-per-step",
            "-1",
        ]

        with pytest.raises(
            ValueError, match="Batch size must be a positive integer"
        ):
            args = parser.parse_args(cli_args)

            parse_and_validate_config(args)

    too_large_test_cases = [20000, 1025, 2048]
    for case in too_large_test_cases:
        cli_args = [
            "--patch-radius",
            "6",
            "--saturation-coefficient",
            "0.9",
            "--regularization-coefficient",
            "0.1",
            "--log-level",
            "info",
            "--weight-dir",
            "/tmp/path/to/weights",
            "--log-dir",
            "/tmp/path/to/logs",
            "train",
            "--dataset-dir",
            "/tmp/path/to/dataset",
            "--dropout",
            "0.1",
            "-b",
            f"{case}",
            "-n",
            "10",
            "--loss-strategy",
            "intermediate",
            "unet",
            "--unfolded-step-size",
            "0.01",
            "--pooling",
            "max",
            "--activation",
            "gelu",
            "--channels-per-layer",
            "3",
            "256",
            "512",
            "256",
            "3",
            "--sublayers-per-step",
            "-1",
        ]

        with pytest.raises(ValueError, match="at most 1024"):
            args = parser.parse_args(cli_args)

            parse_and_validate_config(args)
