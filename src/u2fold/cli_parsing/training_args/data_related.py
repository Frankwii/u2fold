from pathlib import Path

from u2fold.cli_parsing.cli_argument import CLIArgument, DirectoryCLIArgument
from u2fold.utils.track import tag


@tag("cli_argument/train/dataset_dir")
class DatasetDir(DirectoryCLIArgument):
    def short_name(self) -> str:
        return "-d"

    def _name(self) -> str:
        return "dataset"

    def help(self) -> str:
        return """
        Path of the dataset to be used for traning. Splitting \
        into train, validation and testing subsets is handled by the program."
        """
