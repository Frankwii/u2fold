from typing import Literal
import torch
from torch import Tensor

from u2fold.model.common_namespaces import ForwardPassResult

from .generic import BaseLoss


class GroundTruthLoss(BaseLoss):
    loss: Literal["ground_truth"]

    def ground_truth_loss(
        self,
        result: ForwardPassResult,
        ground_truth: Tensor,
    ) -> Tensor:
        return self.weight * torch.nn.functional.mse_loss(result.radiance, ground_truth)
