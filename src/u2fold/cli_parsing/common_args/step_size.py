from u2fold.cli_parsing.cli_argument import CLIArgument
from u2fold.utils import tag


@tag("cli_argument/common/step_size")
class StepSize(CLIArgument[float]):
    def long_name(self) -> str:
        return "--step-size"

    def help(self) -> str:
        return (
            "Step size of the analytical proximity operator. That is, the"
            " step size used for the proximity operator associated to the "
            " dual variable."
        )

    def default(self) -> float:
        return 0.01

    def required(self) -> bool:
        return False

    def _validate_value(self, value: float) -> None:
        if not 0 < value <= 1:
            raise ValueError(
                "Step size must be greater than 0 and less than or equal to 1."
            )
