from typing import NamedTuple

import torch


class GreedyIterationModels(NamedTuple):
    """
    The set of models involved in a "greedy penalty" iteration.
    """
    image: list[torch.nn.Module]
    kernel: list[torch.nn.Module]
