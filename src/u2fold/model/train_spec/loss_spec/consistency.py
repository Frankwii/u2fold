import itertools
from typing import Literal, final, override

import torch
from torch import Tensor

from u2fold.model.common_namespaces import ForwardPassResult
from u2fold.model.train_spec.loss_spec.generic import BaseLossSpec, BaseLossModule
from u2fold.neural_networks.metrics_and_losses import mse

@final
class ConsistencyModule(BaseLossModule):
    # NOTE: It's impossible to calibrate this one before training, so I'll just leave it as is.
    calibration_average = 1.0 
    @override
    @classmethod
    def _forward(cls, result: ForwardPassResult, ground_truth: Tensor) -> Tensor:
        losses = itertools.starmap(
            mse,
            itertools.pairwise(result.primal_variable_history),
        )

        return torch.sum(torch.stack(list(losses)))


class ConsistencyLossSpec(BaseLossSpec):
    loss: Literal["consistency"]

    @override
    def instantiate(self) -> ConsistencyModule:
        return ConsistencyModule(weight=self.weight)
