from u2fold.cli_parsing.cli_argument import CLIArgument, DirectoryCLIArgument
from u2fold.utils.track import tag


@tag("cli_argument/train/dataset_dir")
class DatasetDir(DirectoryCLIArgument):
    def short_name(self) -> str:
        return "-d"

    def _name(self) -> str:
        return "dataset"

    def help(self) -> str:
        return (
            "Path of the dataset to be used for traning. Splitting into train,"
            " validation and testing subsets is handled by the program."
        )


@tag("cli_argument/train/n_epochs")
class NumberOfEpochs(CLIArgument[int]):
    def short_name(self) -> str:
        return "-n"

    def long_name(self) -> str:
        return "--n-epochs"

    def help(self) -> str:
        return "Number of epochs to train for."

    def metavar(self) -> str:
        return "N_EPOCHS"

    def _validate_value(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Number of training epochs should be at least 1")
        elif value > 1000:
            raise ValueError("Too many training epochs!")


@tag("cli_argument/train/batch_size")
class BatchSize(CLIArgument[int]):
    def short_name(self) -> str:
        return "-b"

    def long_name(self) -> str:
        return "--batch-size"

    def help(self) -> str:
        return "Batch size for loading the data."

    def metavar(self) -> str:
        return "BATCH_SIZE"

    def _validate_value(self, value: int) -> None:
        if value <= 0:
            raise ValueError(
                f"Batch size must be a positive integer. Value: {value}."
            )
        elif value > 1024:
            raise ValueError(
                f"Batch size too large. Must be at most 1024. Value: {value}."
            )
