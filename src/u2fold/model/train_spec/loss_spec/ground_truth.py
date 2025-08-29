from typing import Literal, final, override

from torch import Tensor
import torch

from u2fold.math.rescale_image import rescale_color
from u2fold.model.common_namespaces import ForwardPassResult
from u2fold.neural_networks.metrics_and_losses import mse

from .generic import BaseLossModule, BaseLossSpec


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
        clamped_transmission_map = result.deterministic_components.transmission_map.clamp(0.1)
        return torch.stack([
            mse(rescale_color(primal_variable / clamped_transmission_map), ground_truth)
            for primal_variable in result.primal_variable_history[1:]
        ]).mean()


class GroundTruthLossSpec(BaseLossSpec):
    loss: Literal["ground_truth"]

    @override
    def instantiate(self) -> GroundTruthModule:
        return GroundTruthModule(weight=self.weight)
