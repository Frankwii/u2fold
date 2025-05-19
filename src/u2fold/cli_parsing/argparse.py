from argparse import ArgumentParser
from dataclasses import Field, fields
from typing import cast, get_args, get_origin

from u2fold.models.generic import ModelConfig
from u2fold.utils import get_tag_group
from u2fold.utils.ensure_loaded import ensure_loaded
from u2fold.utils.track import get_from_tag

from .cli_argument import CLIArgument


def build_parser() -> ArgumentParser:
    ensure_loaded("u2fold.cli_parsing")

    common_parser = ArgumentParser(
        description="U2FOLD: Underwater image UnFOLDing. Process subaquatic \
        images with an unfolding-based algorithm.",
        add_help=True,
    )

    __add_mode_subparsers(common_parser, ["train", "exec"])

    __add_cliarguments_to_subparser(common_parser, "common")

    return common_parser


def __add_mode_subparsers(parser: ArgumentParser, modes: list[str]) -> None:
    mode_subparsers = parser.add_subparsers(
        help="Either train a model or execute one.", required=True, dest="mode"
    )

    for mode in modes:
        mode_parser = mode_subparsers.add_parser(
            mode,
            help=f"{__format_mode(mode)} a model.",
            description=__format_description(mode, "given"),
        )

        __add_models_as_subparsers(mode_parser, mode)

        __add_cliarguments_to_subparser(mode_parser, mode)


def __add_models_as_subparsers(parser: ArgumentParser, mode: str) -> None:
    model_names = sorted(get_tag_group("model").keys())

    model_subparsers = parser.add_subparsers(
        help="Specify the model to be used.", required=True, dest="model"
    )

    for model in model_names:
        model_parser = model_subparsers.add_parser(
            model,
            help=get_from_tag(f"model/{model}").__doc__,
            description=__format_description(mode, model),
        )

        model_config = cast(ModelConfig, get_from_tag(f"config/model/{model}"))

        model_fields = fields(model_config)

        config_fields = {}
        for cli_mode in [mode, "model"]:
            config_fields[cli_mode] = [
                field
                for field in model_fields
                if field.metadata.get("cli_mode", "model") == cli_mode
            ]

        __add_model_args_to_parser(model_parser, *(config_fields["model"]))
        __add_model_args_to_parser(parser, *(config_fields[mode]))


def __add_model_args_to_parser(parser: ArgumentParser, *fields: Field) -> None:
    for field in fields:
        field_type = field.type
        annotation_origin = get_origin(field_type)
        annotation_args = get_args(field_type)

        is_list = annotation_origin is not None and issubclass(
            annotation_origin, list | tuple
        )

        param_type = annotation_args[0] if is_list else field_type
        long_name = f"--{field.name.replace('_', '-')}"

        parser.add_argument(
            long_name,
            help=field.metadata.get("desc", "No help available"),
            type=param_type,
            nargs="+" if is_list else None,
            required=True,
            action="extend" if is_list else "store",
        )


def __format_description(mode: str, model: str) -> str:
    input = "dataset" if mode == "train" else "input image"
    trail = f"with the specified hyperparameters and {input}."
    action = __format_mode(mode)

    return f"{action} a {model} model {trail}"


def __format_mode(mode: str) -> str:
    return "Train" if mode == "train" else "Execute"


def __add_cliarguments_to_subparser(parser: ArgumentParser, mode: str) -> None:
    for arg_class in get_tag_group(f"cli_argument/{mode}").values():
        arg = cast(CLIArgument, arg_class())
        arg.add_to_parser(parser)
