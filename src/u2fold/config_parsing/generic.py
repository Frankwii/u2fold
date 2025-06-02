from argparse import Namespace
from dataclasses import _MISSING_TYPE, fields
from typing import Any, Literal

from u2fold.models.generic import ModelConfig
from u2fold.utils.track import get_from_tag, get_tag_group


def parse_mode_cliarguments(args: Namespace) -> dict[str, Any]:
    mode = args.mode

    merged = __parse_group_cliarguments(
        args, "common"
    ) | __parse_group_cliarguments(args, mode)

    return merged


def __parse_group_cliarguments(
    args: Namespace, group: Literal["train", "exec", "common"]
) -> dict[str, Any]:
    argument_values = {}
    for name, cliarg_class in get_tag_group(f"cli_argument/{group}").items():
        if (value := getattr(args, name)) is not None:
            cliarg_class().validate_value(value)
            argument_values[name] = value

    return argument_values


def parse_model_arguments(args: Namespace) -> ModelConfig:
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
