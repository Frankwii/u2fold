
from u2fold.cli_parsing.cli_argument import FileCLIArgument
from u2fold.utils.track import tag


@tag("cli_argument/exec/input_image")
class InputPath(FileCLIArgument):
    def short_name(self) -> str:
        return "-i"

    def _name(self) -> str:
        return "input"

    def help(self) -> str:
        return "Input file path. \
            TODO: figure out whether this is relative or absolute."

@tag("cli_argument/exec/output_image")
class OutputPath(FileCLIArgument):
    def short_name(self) -> str:
        return "-o"

    def _name(self) -> str:
        return "output"

    def help(self) -> str:
        return "Output file path. \
        TODO: figure out whether this is relative or absolute."
