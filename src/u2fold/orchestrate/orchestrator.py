from abc import ABC, abstractmethod
from logging import getLogger
from typing import cast

import u2fold
import torch
from torch import Tensor
from u2fold.config_parsing.config_dataclasses import (
    ExecConfig,
    TrainConfig,
    U2FoldConfig,
)
from u2fold.math import convolution, proximities
from u2fold.math.background_light_estimation import estimate_background_light
from u2fold.models.weight_handling.generic import ModelInitBundle, WeightHandler
from u2fold.models.weight_handling.train import TrainWeightHandler
from u2fold.utils.track import get_from_tag
from u2fold.data import get_dataloaders

from u2fold.math.compute_first_term import compute_I2, estimate_radiance_and_transmission_map


class Orchestrator[T: U2FoldConfig](ABC):
    def __init__(
        self,
        config: T,
        weigth_handler: WeightHandler
    ) -> None: ...

    @abstractmethod
    def run(self) -> None: ...

class TrainOrchestrator(Orchestrator):
    def __init__(self, config: TrainConfig) -> None:
        self.__logger = getLogger(__name__)
        self.__config = config
        self.__weight_handler = TrainWeightHandler(
            config.weight_dir
        )

        image_bundle = kernel_bundle = ModelInitBundle(
            config.model_config,
            get_from_tag(f"model/{config.model_name}"),
            device=config.device
        )

        self.__models = self.__weight_handler.load_models(
            image_bundle, kernel_bundle
        )

        self.__dataloaders = get_dataloaders(
            "uieb", # TODO: Add as CLIArgument
            config.dataset_dir,
            config.batch_size,
            config.device
        )

    def forward_pass(
        self,
        input: Tensor, # (B, C, H, W)
    ) -> Tensor: # (B, C, H, W)
        background_lights = estimate_background_light(input)

        image, transmission_maps = estimate_radiance_and_transmission_map(
            input,
            background_lights,
            self.__config.transmission_map_estimation_config
        ) # (J_0, t); ((B, C, H, W), (B, 1, H, W))

        second_terms = compute_I2(
            input, image, transmission_maps, background_lights
        ) # I2; (B, C, H, W)

        kernel = torch.zeros(self.__config.batch_size, 3, 20, 20)
        dual_image = torch.zeros_like(input)
        dual_kernel = torch.zeros_like(kernel)

        tau = self.__config.unfolded_step_size
        sigma = self.__config.step_size
        sigma_I2 = sigma * second_terms
        for greedy_iter, greedy_iter_models in enumerate(self.__models):
            scaled_flipped_image = convolution.double_flip(image) * tau

            for stage, model in enumerate(greedy_iter_models.image):
                tmp = cast(Tensor, model(kernel - convolution.conv(dual_kernel, scaled_flipped_image)))
                kernel = 2 * tmp - kernel
                dual_kernel = (
                    dual_kernel + sigma * convolution.conv(kernel, image) - sigma_I2
                ) / (1 + sigma)

            scaled_flipped_kernel = convolution.double_flip(kernel) * tau
            for stage, model in enumerate(greedy_iter_models.kernel):
                tmp = cast(Tensor, model(image - convolution.conv(dual_image, scaled_flipped_kernel)))

                image = 2 * tmp - image
                dual_image = (
                    dual_image + sigma * convolution.conv(image, kernel) - sigma_I2
                ) / (1 + sigma)

        return image / torch.max(transmission_maps, torch.tensor(0.1))



    def run_train_epoch(self):
        for input, ground_truth in self.__dataloaders.training:
            analytical_estimation = compute_I2(
                input,
                self.__config.transmission_map_estimation_config
            )

            # Chambolle-Pock primal-dual algorithm
            images = analytical_estimation
            kernels = torch.zeros(self.__config.batch_size, 3, 20, 20)

            for greedy_iteration_models in self.__models:
                image_dual_variables = torch.randn_like(images)
                kernel_dual_variables = torch.randn_like(kernels)

                # Fix kernel, estimate image
                for model in greedy_iteration_models.image:
                    images: Tensor = model(
                        images - self.__config.unfolded_step_size * convolution.convex_conjugate(image_dual_variables, kernels)
                    )

                    image_dual_variables = proximities.shifted_square_L2_norm_conjugate(
                        images,
                        analytical_estimation,
                        self.__config.step_size
                    )

                # Fix image, estimate kernel
                for model in greedy_iteration_models.kernel:
                    kernels: Tensor = model(
                        kernels - self.__config.unfolded_step_size * convolution.convex_conjugate(kernels, image_dual_variables)
                    )


    def run_greedy_iteration(
        self,
        image: Tensor,
        kernel: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Fix kernel, estimate image
        for model in self.__models



class ExecOrchestrator(Orchestrator):
    def __init__(self, config: ExecConfig) -> None: ...

    def run(self) -> None: ...
