from pathlib import Path
from PIL import Image, UnidentifiedImageError

from u2fold.cli_parsing.cli_argument import FileCLIArgument
from u2fold.utils.track import tag


@tag("cli_argument/exec/input_image")
class InputPath(FileCLIArgument):
    def short_name(self) -> str:
        return "-i"

    def _name(self) -> str:
        return "input"

    def help(self) -> str:
        return "Input file path."

    def _validate_value(self, value: Path) -> None:
        super()._validate_value(value)

        if value.suffix not in Image.registered_extensions():
            errmsg = f"Unsupported image file format for {value}."
            raise UnidentifiedImageError(errmsg)


@tag("cli_argument/exec/output_image")
class OutputPath(FileCLIArgument):
    def short_name(self) -> str:
        return "-o"

    def _name(self) -> str:
        return "output"

    def help(self) -> str:
        return "Output file path."

    def _validate_value(self, value: Path) -> None:
        if value.is_dir():
            raise IsADirectoryError(
                f"Specified output file {value} is a directory!"
            )
