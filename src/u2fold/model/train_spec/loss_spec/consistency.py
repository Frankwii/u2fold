import itertools
from typing import Literal

import torch
from torch import Tensor

from u2fold.model.common_namespaces import ForwardPassResult
from u2fold.model.train_spec.loss_spec.generic import BaseLoss


class ConsistencyLoss(BaseLoss):
    loss: Literal["consistency"]

    def forward(self, result: ForwardPassResult, ground_truth: Tensor) -> Tensor:
        losses = itertools.starmap(
            torch.nn.functional.mse_loss, itertools.pairwise(result.primal_variable_history)
        )

        return self.weight * torch.mean(torch.stack(tuple(losses)))
