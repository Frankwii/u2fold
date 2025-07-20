from typing import Literal

import torch
from torch import Tensor

from u2fold.model.common_namespaces import ForwardPassResult
from u2fold.math.convolution import convolve

from .generic import BaseLossSpec, BaseLossModule

class FidelityModule(BaseLossModule):
    @classmethod
    def _forward(cls, result: ForwardPassResult, ground_truth: Tensor) -> Tensor:
        last_primal_variable = result.primal_variable_history[-1]
        last_kernel = result.kernel_history[-1]

        return torch.nn.functional.mse_loss(
            convolve(last_kernel, last_primal_variable),
            result.deterministic_components.fidelity,
        )


class FidelityLossSpec(BaseLossSpec):
    loss: Literal["fidelity"]

    def instantiate(self) -> FidelityModule:
        return FidelityModule(weight=self.weight)
