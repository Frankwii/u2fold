from dataclasses import dataclass

from torch import Tensor


@dataclass
class DeterministicComponents:
    fidelity: Tensor
    transmission_map: Tensor
    background_light: Tensor


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

        self.radiance = (
            self.primal_variable_history[-1] /
            self.deterministic_components.transmission_map.clamp(0.1)
        )
