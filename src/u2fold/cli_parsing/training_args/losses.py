import u2fold.models.supported_components as components
from u2fold.cli_parsing.cli_argument import CLIArgument
from u2fold.utils.track import tag


@tag("cli_argument/train/loss_strategy")
class LossStrategy(CLIArgument[str]):
    def short_name(self) -> str:
        return "-l"

    def long_name(self) -> str:
        return "--loss-strategy"

    def help(self) -> str:
        return ("Which strategy to use for computing the loss of a forward pass"
                " of the full algorithm (not only for a single model).")

    def choices(self) -> set[str]:
        return components.get_supported_loss_strategies()

    def required(self) -> bool:
        return True

    def _validate_value(self, value: str) -> None:
        pass
