from typing import Literal

import torch
from torch import Tensor

from u2fold.model.common_namespaces import ForwardPassResult

from .generic import BaseLossSpec, BaseLossModule


class ColorCosineSimilarityModule(BaseLossModule):
    @classmethod
    def _forward(cls, result: ForwardPassResult, ground_truth: Tensor) -> Tensor:
        return 1 - torch.cosine_similarity(result.radiance, ground_truth, dim=1).mean()


class ColorCosineSimilarityLossSpec(BaseLossSpec):
    loss: Literal["color_cosine_similarity"]

    def instantiate(self) -> ColorCosineSimilarityModule:
        return ColorCosineSimilarityModule(weight=self.weight)
