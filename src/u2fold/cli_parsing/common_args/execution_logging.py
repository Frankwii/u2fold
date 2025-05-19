from u2fold.cli_parsing.cli_argument import CLIArgument
from u2fold.utils import tag


@tag("cli_argument/common/log_level")
class LogLevel(CLIArgument):
    def short_name(self) -> str:
        return "-L"

    def long_name(self) -> str:
        return "--log-level"

    def help(self) -> str:
        return """
        Log level of execution-related tasks. \
        This is for debugging purposes mostly.
        """

    def value_type(self) -> type:
        return str

    def choices(self) -> list[str]:
        return ["debug", "info", "warning", "error", "critical"]

    def default(self) -> str:
        return "debug"

    def required(self) -> bool:
        return False

@tag("cli_argument/common/log_dir")
class LogDir(CLIArgument):
    def short_name(self) -> str:
        return "-l"

    def long_name(self) -> str:
        return "--log-dir"

    def help(self) -> str:
        return """
        Path to directory for logging. Can be either relative to the current \
        directory or absolute.
        The directory will be created if it did not exist and emptied if it \
        did exist.
        """

    def value_type(self) -> type:
        return str

    def default(self) -> str:
        return "debug"

    def required(self) -> bool:
        return False
