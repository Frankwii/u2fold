from typing import Literal
from torch import Tensor
import torch

from u2fold.model.common_namespaces import ForwardPassResult
from .generic import BaseLoss


class FidelityLoss(BaseLoss):
    loss: Literal["fidelity"]

    def fidelity_loss(self, result: ForwardPassResult, ground_truth: Tensor) -> Tensor:
        last_primal_variable = result.primal_variable_history[-1]
        last_kernel = result.kernel_history[-1]

        return self.weight * torch.nn.functional.mse_loss(
            convolution.conv(last_kernel, last_primal_variable),
            result.deterministic_components.fidelity
        )
