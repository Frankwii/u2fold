from u2fold.cli_parsing.cli_argument import CLIArgument, DirectoryCLIArgument
from u2fold.utils import tag


@tag("cli_argument/common/log_level")
class LogLevel(CLIArgument[str]):
    def short_name(self) -> str:
        return "-L"

    def long_name(self) -> str:
        return "--log-level"

    def help(self) -> str:
        return (
            "Log level of execution-related tasks."
            " This is for debugging purposes mostly."
        )

    def choices(self) -> list[str]:
        return ["debug", "info", "warning", "error", "critical"]

    def default(self) -> str:
        return "debug"

    def required(self) -> bool:
        return False

    def _validate_value(self, value: str) -> None: ...


@tag("cli_argument/common/log_dir")
class LogDir(DirectoryCLIArgument):
    def short_name(self) -> str:
        return "-l"

    def help(self) -> str:
        return (
            "Path to directory for logging. The directory will be created"
            " if it did not exist."
        )

    def _name(self) -> str:
        return "log"
