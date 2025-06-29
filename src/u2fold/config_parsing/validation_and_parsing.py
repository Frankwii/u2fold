from argparse import Namespace
from dataclasses import _MISSING_TYPE, fields
from pathlib import Path
from typing import Any, Literal, cast

import torch

from u2fold.models.generic import ModelConfig
from u2fold.utils.track import get_from_tag, get_tag_group

from .config_dataclasses import (
    ExecConfig,
    TrainConfig,
    TransmissionMapEstimationConfig,
    U2FoldConfig,
)


def __validate_cliarguments(args: Namespace) -> dict[str, Any]:
    mode = args.mode

    merged = __validate_group_cliarguments(
        args, "common"
    ) | __validate_group_cliarguments(args, mode)

    return merged


def __validate_group_cliarguments(
    args: Namespace, group: Literal["train", "exec", "common"]
) -> dict[str, Any]:
    argument_values = {}
    for name, cliarg_class in get_tag_group(f"cli_argument/{group}").items():
        if (value := getattr(args, name)) is not None:
            cliarg_class().validate_value(value)
            argument_values[name] = value

    return argument_values


def __parse_model_arguments(args: Namespace) -> ModelConfig:
    model = args.model

    if model not in get_tag_group("model"):
        raise ValueError(f"Model {model} is unsupported.")

    model_config_class = get_from_tag(f"config/model/{model}")

    model_param_values = {}
    for field in fields(model_config_class):
        name = field.name
        value = getattr(args, name) or field.default

        if isinstance(value, _MISSING_TYPE):
            raise ValueError("Missing a required model parameter!")

        model_param_values[name] = value

    return model_config_class(**model_param_values)


def __instantiate_transmission_map_config(
    config_args: dict[str, Any],
) -> None:
    patch_radius = config_args.pop("patch_radius")
    saturation_coefficient = config_args.pop("saturation_coefficient")
    regularization_coefficient = config_args.pop("regularization_coefficient")

    config_args["transmission_map_estimation_config"] = (
        TransmissionMapEstimationConfig(
            patch_radius, saturation_coefficient, regularization_coefficient
        )
    )


def __handle_exceptions(
    config_args: dict[str, Any],
    model_name: str,
    mode: str,
    model_config: ModelConfig,
) -> None:
    """Manually handle arguments with validation that cannot be automated.

    Mutates "config_args" in place.
    """
    __instantiate_transmission_map_config(config_args)

    model_path = Path(model_name) / Path(
        model_config.format_self(
            config_args["transmission_map_estimation_config"]
        )
    )

    config_args["weight_dir"] = config_args.pop("weight_dir") / model_path

    log_dir = cast(Path, config_args.pop("log_dir"))
    path = log_dir / "execution" / model_path
    path.mkdir(parents=True, exist_ok=True)
    config_args["execution_log_dir"] = path

    if mode == "train":
        path = log_dir / "tensorboard" / model_path
        path.mkdir(parents=True, exist_ok=True)
        config_args["tensorboard_log_dir"] = path

    config_args["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    config_args["unfolded_step_size"] = model_config.unfolded_step_size


def parse_and_validate_config(args: Namespace) -> U2FoldConfig:
    mode = args.mode

    if mode not in {"train", "exec"}:
        raise ValueError(f"Unsupported mode: {mode}")

    config_args = __validate_cliarguments(args)

    model_config = __parse_model_arguments(args)

    __handle_exceptions(config_args, args.model, mode, model_config)

    config_args["model_config"] = model_config
    config_args["model_name"] = args.model

    match mode:
        case "train":
            return TrainConfig(**config_args)
        case "exec":
            return ExecConfig(**config_args)
        case _:
            raise ValueError(f"Unsupported mode: {mode}")
