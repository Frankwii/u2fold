import torch

from u2fold.cli_parsing.cli_argument import CLIArgument
from u2fold.utils.track import get_tag_group, tag


@tag("cli_argument/train/model_name")
class ModelName(CLIArgument):
    def short_name(self) -> str:
        return "-m"

    def long_name(self) -> str:
        return "--model"

    def value_type(self) -> type:
        return torch.nn.Module

    def help(self) -> str:
        return """
        Which neural network to use for unfolding.
        """

    def choices(self) -> list[str]:
        return list(get_tag_group("model").keys())
