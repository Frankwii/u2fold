from dataclasses import dataclass
from typing import Callable, NamedTuple, final

from torch import Tensor

from torch.nn import Parameter
from torch.optim import Optimizer

from u2fold.math.primal_dual import PrimalDualScheme
from u2fold.math.rescale_image import rescale_color

@dataclass
class KernelBundle:
    preimage: Parameter
    preimage_to_kernel_mapping: Callable[[Tensor], Tensor]
    optimizer: Optimizer

    def compute_kernel(self):
        return self.preimage_to_kernel_mapping(self.preimage)


@dataclass
class PrimalDualBundle:
    scheme: PrimalDualScheme  # pyright: ignore[reportMissingTypeArgument]
    primal_variable: Tensor
    dual_variable: Tensor



@dataclass
class DeterministicComponents:
    fidelity: Tensor
    transmission_map: Tensor
    background_light: Tensor

class EpochMetricData(NamedTuple):
    overall_loss: float
    granular_loss: dict[str, float]
    metrics: dict[str, float]


def compute_radiance(primal_variable: Tensor, clamped_transmission_map: Tensor) -> Tensor:
    return rescale_color(primal_variable / clamped_transmission_map)

@final
class ForwardPassResult:
    def __init__(
        self,
        primal_variable_history: list[Tensor],
        kernel_history: list[Tensor],
        deterministic_components: DeterministicComponents,
    ) -> None:
        self.primal_variable_history = primal_variable_history
        self.kernel_history = kernel_history
        self.deterministic_components = deterministic_components

        self.radiance = compute_radiance(
            self.primal_variable_history[-1],
            self.deterministic_components.transmission_map.clamp(0.1)
        )
