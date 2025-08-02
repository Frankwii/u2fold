from typing import Literal, final, override

from torch import Tensor

from u2fold.model.common_namespaces import ForwardPassResult
from u2fold.neural_networks.metrics_and_losses import mse

from .generic import BaseLossSpec, BaseLossModule


@final
class GroundTruthModule(BaseLossModule):
    calibration_average = 0.025612633675336838

    @override
    @classmethod
    def _forward(
        cls,
        result: ForwardPassResult,
        ground_truth: Tensor,
    ) -> Tensor:
        return mse(result.radiance, ground_truth)


class GroundTruthLossSpec(BaseLossSpec):
    loss: Literal["ground_truth"]

    @override
    def instantiate(self) -> GroundTruthModule:
        return GroundTruthModule(weight=self.weight)
