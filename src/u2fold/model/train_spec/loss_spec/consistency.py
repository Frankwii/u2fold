import itertools
from typing import Literal

import torch
from torch import Tensor

from u2fold.model.common_namespaces import ForwardPassResult
from u2fold.model.train_spec.loss_spec.generic import BaseLossSpec, BaseLossModule


class ConsistencyModule(BaseLossModule):
    @classmethod
    def _forward(cls, result: ForwardPassResult, ground_truth: Tensor) -> Tensor:
        losses = itertools.starmap(
            torch.nn.functional.mse_loss,
            itertools.pairwise(result.primal_variable_history),
        )

        return torch.mean(torch.stack(tuple(losses)))


class ConsistencyLossSpec(BaseLossSpec):
    loss: Literal["consistency"]

    def instantiate(self) -> ConsistencyModule:
        return ConsistencyModule(weight=self.weight)
