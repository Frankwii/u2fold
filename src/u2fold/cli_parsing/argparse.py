from argparse import ArgumentParser
from typing import cast

from u2fold.cli_parsing.cli_argument import CLIArgument
from u2fold.utils import get_tag_group
from u2fold.utils.ensure_loaded import ensure_loaded


def add_tagged_args_to_parser(parser: ArgumentParser, tag: str) -> None:
    for arg_class in get_tag_group(tag).values():
        arg = cast(CLIArgument, arg_class())
        arg.add_to_parser(parser)

def build_parser() -> ArgumentParser:
    ensure_loaded("u2fold.cli_parsing")

    # Common args
    common_parser = ArgumentParser(
        description="U2FOLD: Underwater image UnFOLDing. Process subaquatic \
        images with an unfolding-based algorithm.",
        add_help=True
    )

    add_tagged_args_to_parser(common_parser, "cli_argument/common")

    subparsers = common_parser.add_subparsers(
        help="Either train a model or execute one.",
        required=True
    )

    # Train args
    train_parser = subparsers.add_parser(
        "train",
        parents=[common_parser],
        add_help=False,
    )

    add_tagged_args_to_parser(train_parser, "cli_argument/train")

    # Exec args
    exec_parser = subparsers.add_parser(
        "exec",
        parents=[common_parser],
        add_help=False
    )

    add_tagged_args_to_parser(exec_parser, "cli_argument/exec")

    return common_parser
