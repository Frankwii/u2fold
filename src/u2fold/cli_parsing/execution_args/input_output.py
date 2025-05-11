from pathlib import Path

from u2fold.cli_parsing.cli_argument import CLIArgument
from u2fold.utils.track import track


@track(tag="cli_argument/exec/input_image")
class InputPath(CLIArgument):
    def short_name(self) -> str:
        return "-i"

    def long_name(self) -> str:
        return "--input"

    def metavar(self) -> str:
        return "INPUT_PATH"

    def help(self) -> str:
        return "Input file path. \
            TODO: figure out whether this is relative or absolute."

    def value_type(self) -> type:
        return Path


@track(tag="cli_argument/exec/output_image")
class OutputPath(CLIArgument):
    def short_name(self) -> str:
        return "-o"

    def long_name(self) -> str:
        return "--output"

    def metavar(self) -> str:
        return "OUTPUT_PATH"

    def help(self) -> str:
        return "Output file path. \
        TODO: figure out whether this is relative or absolute."

    def value_type(self) -> type:
        return Path
