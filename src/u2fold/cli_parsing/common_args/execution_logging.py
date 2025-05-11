from u2fold.cli_parsing.cli_argument import CLIArgument
from u2fold.utils import track


@track(tag="cli_argument/common/log_level")
class LogLevel(CLIArgument):
    def short_name(self) -> str:
        return "-l"

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
        return ["debug", "info", "warn", "error", "critical"]

    def default(self) -> str:
        return "debug"
