from typing import Literal

import torch
from pydantic import NonNegativeFloat
from torch import Tensor

from u2fold.model.common_namespaces import ForwardPassResult
from .generic import BaseLoss


class ColorCosineSimilarityLoss(BaseLoss):
    loss: Literal["color_cosine"]
    weight: NonNegativeFloat

    def forward(self, result: ForwardPassResult, ground_truth: Tensor) -> Tensor:
        return self.weight * (
            1 - torch.cosine_similarity(result.radiance, ground_truth, dim=1).mean()
        )
