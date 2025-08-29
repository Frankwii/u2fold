from abc import ABC, abstractmethod
from itertools import chain, repeat
from logging import getLogger
from typing import Any, Iterable

import torch
from torch import Tensor

import u2fold.orchestrate.functional as F
from u2fold.math import proximities
from u2fold.math.convolution import convolve, double_flip
from u2fold.model.common_namespaces import (
    ForwardPassResult,
    KernelBundle,
    PrimalDualBundle,
)
from u2fold.model.neural_network_spec import NeuralNetworkSpec
from u2fold.model.spec import U2FoldSpec
from u2fold.neural_networks.generic import NeuralNetwork
from u2fold.neural_networks.weight_handling.generic import (
    ModelInitBundle,
    WeightHandler,
)
from u2fold.orchestrate.functional.initialization import (
    initialize_square_matrix_with_square_distances_to_center,
)
from u2fold.utils.get_device import get_device
from u2fold.utils.track import get_from_tag


@torch.enable_grad
def optimize_kernel(
    kernel_bundle: KernelBundle,
    primal_variable: Tensor,
    fidelity: Tensor,
    n_iters: int,
) -> Tensor:
    for _ in range(n_iters):
        kernel_bundle.optimizer.zero_grad()

        kernel = kernel_bundle.compute_kernel()

        approximation = convolve(kernel, primal_variable.detach())

        fidelity_loss = torch.nn.functional.mse_loss(approximation, fidelity)

        fidelity_loss.backward()

        kernel_bundle.optimizer.step()

    return kernel_bundle.compute_kernel()


def optimize_primal_dual[Spec: NeuralNetworkSpec](
    primal_dual_bundle: PrimalDualBundle,
    greedy_iteration_models: Iterable[NeuralNetwork[Spec]],
    current_primal_variable: Tensor,
    current_dual_variable: Tensor,
) -> tuple[list[Tensor], list[Tensor]]:
    primal_variable_history = []
    dual_variable_history = []
    for model in greedy_iteration_models:
        primal_variable, dual_variable = (
            primal_dual_bundle.scheme.with_primal_proximity(model, False).run(
                current_primal_variable, current_dual_variable, 1
            )
        )

        primal_variable_history.append(primal_variable)
        dual_variable_history.append(dual_variable)

    return primal_variable_history, dual_variable_history


class Orchestrator[W: WeightHandler](ABC):
    def __init__(self, spec: U2FoldSpec[Any], weigth_handler: W) -> None:
        self._logger = getLogger(__name__)
        self._logger.info("Initializing orchestrator.")
        self._spec = spec
        self._device = get_device()
        self._weight_handler = weigth_handler
        image_bundle = ModelInitBundle(
            spec.neural_network_spec,
            get_from_tag(f"model/{spec.neural_network_spec.name}"),
            device=self._device,
        )

        self._models = self._weight_handler.load_models(image_bundle)

        self.__kernel_size = 7
        self.__kernel_square_distances_to_center = (
            initialize_square_matrix_with_square_distances_to_center(
                self.__kernel_size, self._device
            )
        )

    def forward_pass(
        self,
        input: Tensor,
    ) -> ForwardPassResult:
        algorithmic_spec = self._spec.algorithmic_spec
        nn_spec = self._spec.neural_network_spec
        deterministic_components = F.compute_deterministic_components(
            input,
            algorithmic_spec.guided_filter_patch_radius,
            algorithmic_spec.transmission_map_patch_radius,
            algorithmic_spec.transmission_map_saturation_coefficient,
            algorithmic_spec.guided_filter_regularization_coefficient,
        )

        kernel_bundle = F.initialize_gaussian_kernel(
            batch_size=input.size(0),
            device=get_device(),
            square_distances_to_center=self.__kernel_square_distances_to_center,
            learning_rate=algorithmic_spec.step_size,
        )

        # silence "possibly unbound" type-checker complaints
        kernel = kernel_bundle.compute_kernel()
        kernel_iterations = chain((20,), repeat(10))

        primal_variable = deterministic_components.fidelity
        dual_variable = torch.zeros_like(primal_variable)

        primal_variable_history = [primal_variable]
        kernel_history = [kernel]
        unfolded_step_size = nn_spec.unfolded_step_size
        step_size = algorithmic_spec.step_size
        for greedy_iter_models, n_kernel_iters in zip(self._models, kernel_iterations):
            # Fix image, estimate kernel
            kernel = optimize_kernel(
                kernel_bundle,
                primal_variable,
                deterministic_components.fidelity,
                n_kernel_iters,
            ).detach()
            kernel_history.append(kernel)

            # Fix kernel, estimate image
            flipped_kernel = double_flip(kernel)
            for model in greedy_iter_models:
                tmp = primal_variable
                primal_variable = model(
                    primal_variable
                    - unfolded_step_size
                    * convolve(kernel=flipped_kernel, input=dual_variable)
                )
                overrelaxed_primal_variable = 2 * primal_variable - tmp
                dual_variable = proximities.conjugate_shifted_square_L2_norm(
                    input=dual_variable + step_size * convolve(kernel=kernel, input=overrelaxed_primal_variable),
                    step_size=step_size,
                    shift=deterministic_components.fidelity,
                )

                primal_variable_history.append(primal_variable)

        return ForwardPassResult(
            primal_variable_history, kernel_history, deterministic_components
        )

    @abstractmethod
    def run(self) -> float | None: ...
