from typing import Literal, final, override

from torch import Tensor

from u2fold.model.common_namespaces import ForwardPassResult
from u2fold.neural_networks.metrics_and_losses import color_minimizable

from .generic import BaseLossSpec, BaseLossModule

@final
class ColorCosineSimilarityModule(BaseLossModule):
    calibration_average = 0.035978469997644424

    @override
    @classmethod
    def _forward(cls, result: ForwardPassResult, ground_truth: Tensor) -> Tensor:
        return color_minimizable(result.radiance, ground_truth)


class ColorCosineSimilarityLossSpec(BaseLossSpec):
    loss: Literal["color_cosine_similarity"]

    @override
    def instantiate(self) -> ColorCosineSimilarityModule:
        return ColorCosineSimilarityModule(weight=self.weight)
