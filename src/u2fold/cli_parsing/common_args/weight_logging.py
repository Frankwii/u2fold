from u2fold.cli_parsing.cli_argument import DirectoryCLIArgument
from u2fold.utils import tag


@tag("cli_argument/common/weight_dir")
class WeightDir(DirectoryCLIArgument):
    def short_name(self) -> str:
        return "-w"

    def _name(self) -> str:
        return "weight"

    def help(self) -> str:
        return "Path to directory containing model weights."
