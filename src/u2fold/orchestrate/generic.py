from abc import ABC, abstractmethod
from itertools import chain, repeat
from logging import getLogger
from typing import Iterable, cast

import torch
from torch import Tensor
from torch.optim import Adam

import u2fold.orchestrate.functional as F
from u2fold.config_parsing.config_dataclasses import (
    U2FoldConfig,
)
from u2fold.math import convolution
from u2fold.models.generic import Model
from u2fold.models.weight_handling.generic import ModelInitBundle, WeightHandler
from u2fold.orchestrate.functional.initialization import (
    initialize_square_matrix_with_square_distances_to_center,
)
from u2fold.utils.track import get_from_tag

from .functional.types import (
    ForwardPassResult,
    KernelBundle,
    PrimalDualBundle,
)


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

        approximation = convolution.conv(kernel, primal_variable.detach())

        fidelity_loss = torch.nn.functional.mse_loss(approximation, fidelity)

        fidelity_loss.backward()

        kernel_bundle.optimizer.step()

    return kernel_bundle.compute_kernel()


@torch.compile
def optimize_primal_dual(
    primal_dual_bundle: PrimalDualBundle,
    greedy_iteration_models: Iterable[Model],
    current_primal_variable: Tensor,
    current_dual_variable: Tensor,
) -> tuple[list[Tensor], list[Tensor]]:
    primal_variable_history = []
    dual_variable_history = []
    for model in greedy_iteration_models:
        primal_variable, dual_variable = (
            primal_dual_bundle.schema.with_primal_proximity(model, False).run(
                current_primal_variable, current_dual_variable, 1
            )
        )

        primal_variable_history.append(primal_variable)
        dual_variable_history.append(dual_variable)

    return primal_variable_history, dual_variable_history


class Orchestrator[T: U2FoldConfig, W: WeightHandler](ABC):
    def __init__(self, config: T, weigth_handler: W) -> None:
        self._logger = getLogger(__name__)
        self._logger.info(f"Initializing orchestrator.")
        self._config = config
        self._weight_handler = weigth_handler
        image_bundle = ModelInitBundle(
            config.model_config,
            get_from_tag(f"model/{config.model_name}"),
            device=config.device,
        )

        self._models = self._weight_handler.load_models(image_bundle)

        self._model_optimizer = Adam(
            torch.nn.ModuleList(chain.from_iterable(self._models)).parameters()
        )

        self.__kernel_size = 7
        self.__kernel_square_distances_to_center = (
            initialize_square_matrix_with_square_distances_to_center(
                self.__kernel_size, self._config.device
            )
        )

    def forward_pass(
        self,
        input: Tensor,
    ) -> ForwardPassResult:
        deterministic_components = F.compute_deterministic_components(
            input, self._config.transmission_map_estimation_config
        )

        primal_dual_bundle = F.initialize_primal_dual(
            deterministic_components.fidelity,
            self._config.unfolded_step_size,
            self._config.step_size,
        )

        kernel_bundle = F.initialize_gaussian_kernel(
            input.size(0),
            self._config.device,
            self.__kernel_square_distances_to_center,
            self._config.step_size,
        )

        # silence "possibly unbound" type-checker complaints
        kernel = cast(Tensor, None)
        kernel_iterations = chain((20,), repeat(10))

        primal_variable = primal_dual_bundle.primal_variable
        dual_variable = primal_dual_bundle.dual_variable

        primal_variable_history = []
        kernel_history = []
        for greedy_iter_models, n_iters in zip(self._models, kernel_iterations):
            # Fix image, estimate kernel
            kernel = optimize_kernel(
                kernel_bundle,
                primal_variable,
                deterministic_components.fidelity,
                n_iters,
            )
            kernel_history.append(kernel)

            # Fix kernel, estimate image
            primal_dual_bundle.schema.with_linear_argument(kernel.detach())
            primal_variable_subhistory, dual_variable_subhistory = (
                optimize_primal_dual(
                    primal_dual_bundle,
                    greedy_iter_models,
                    primal_variable,
                    dual_variable,
                )
            )

            primal_variable_history.extend(primal_variable_subhistory)
            primal_variable = primal_variable_subhistory[-1]
            dual_variable = dual_variable_subhistory[-1]

        return ForwardPassResult(
            primal_variable_history, kernel_history, deterministic_components
        )

    @abstractmethod
    def run(self) -> None: ...
