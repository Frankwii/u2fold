from typing import Literal

import torch
from torch import Tensor

from u2fold.model.common_namespaces import ForwardPassResult

from .generic import BaseLossSpec, BaseLossModule


class GroundTruthModule(BaseLossModule):
    @classmethod
    def _forward(
        cls,
        result: ForwardPassResult,
        ground_truth: Tensor,
    ) -> Tensor:
        return torch.nn.functional.mse_loss(result.radiance, ground_truth)


class GroundTruthLossSpec(BaseLossSpec):
    loss: Literal["ground_truth"]

    def instantiate(self) -> GroundTruthModule:
        return GroundTruthModule(weight=self.weight)
