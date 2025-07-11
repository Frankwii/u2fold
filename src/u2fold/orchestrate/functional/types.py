from typing import Callable, NamedTuple, Sequence

from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer

from u2fold.math.primal_dual import PrimalDualSchema


class DeterministicComponents(NamedTuple):
    fidelity: Tensor
    transmission_map: Tensor
    background_light: Tensor


class KernelBundle(NamedTuple):
    preimage: Parameter
    preimage_to_kernel_mapping: Callable[[Tensor], Tensor]
    optimizer: Optimizer

    def compute_kernel(self):
        return self.preimage_to_kernel_mapping(self.preimage)


class PrimalDualBundle(NamedTuple):
    schema: PrimalDualSchema
    primal_variable: Tensor
    dual_variable: Tensor


class ForwardPassResult(NamedTuple):
    primal_variable_history: Sequence[Tensor]
    kernel_history: Sequence[Tensor]
    deterministic_components: DeterministicComponents
