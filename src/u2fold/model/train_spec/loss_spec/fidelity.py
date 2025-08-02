from typing import Literal, final, override

from torch import Tensor

from u2fold.model.common_namespaces import ForwardPassResult
from u2fold.math.convolution import convolve
from u2fold.neural_networks.metrics_and_losses import mse

from .generic import BaseLossSpec, BaseLossModule

@final
class FidelityModule(BaseLossModule):
    calibration_average = 0.03710788115859032
    @override
    @classmethod
    def _forward(cls, result: ForwardPassResult, ground_truth: Tensor) -> Tensor:
        last_primal_variable = result.primal_variable_history[-1]
        last_kernel = result.kernel_history[-1]

        return mse(
            convolve(last_kernel, last_primal_variable),
            result.deterministic_components.fidelity,
        )


class FidelityLossSpec(BaseLossSpec):
    loss: Literal["fidelity"]

    @override
    def instantiate(self) -> FidelityModule:
        return FidelityModule(weight=self.weight)
