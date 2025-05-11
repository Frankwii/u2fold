from pathlib import Path

from u2fold.cli_parsing.cli_argument import CLIArgument
from u2fold.utils.track import track


@track(tag="cli_argument/train/dataset_dir")
class DatasetDir(CLIArgument):
    def short_name(self) -> str:
        return "-d"

    def long_name(self) -> str:
        return "--dataset-dir"

    def metavar(self) -> str:
        return "DATASET_PATH"

    def value_type(self) -> type:
        return Path

    def help(self) -> str:
        return "Path of the dataset to be used for traning. Splitting \
        into train, validation and testing subsets is handled by the program."
