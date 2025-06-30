from abc import ABC, abstractmethod
from logging import getLogger
from typing import cast

from torch.nn import MSELoss

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
from u2fold.models.weight_handling import TrainWeightHandler, ExecWeightHandler
from u2fold.utils.track import get_from_tag
from u2fold.data import get_dataloaders

from u2fold.math.compute_first_term import compute_I2, estimate_radiance_and_transmission_map

class Orchestrator[T: U2FoldConfig, W: WeightHandler](ABC):
    def __init__(
        self,
        config: T,
        weigth_handler: W
    ) -> None:
        self._logger = getLogger(__name__)
        self._logger.info(f"Initializing orchestrator.")
        self._config = config
        self.__weight_handler = weigth_handler
        image_bundle = kernel_bundle = ModelInitBundle(
            config.model_config,
            get_from_tag(f"model/{config.model_name}"),
            device=config.device
        )

        self._models = self.__weight_handler.load_models(
            image_bundle, kernel_bundle
        )
        self._logger.info(f"Finished initializing orchestrator!")

    def forward_pass(
        self,
        input: Tensor, # (B, C, H, W)
    ) -> Tensor: # (B, C, H, W)
        background_lights = estimate_background_light(input)

        image, transmission_maps = estimate_radiance_and_transmission_map(
            input,
            background_lights,
            self._config.transmission_map_estimation_config
        ) # (J_0, t); ((B, C, H, W), (B, 1, H, W))

        second_terms = compute_I2(
            input, image, transmission_maps, background_lights
        ) # I2; (B, C, H, W)

        kernel = torch.zeros(
            getattr(self._config, "batch_size", 1), # TODO: 
            3,
            20,
            20
        )
        dual_image = torch.zeros_like(input)
        dual_kernel = torch.zeros_like(kernel)

        tau = self._config.unfolded_step_size
        sigma = self._config.step_size
        sigma_I2 = sigma * second_terms
        # TODO: Add logging
        for greedy_iter, greedy_iter_models in enumerate(self._models):

            # Fix image, estimate kernel
            scaled_flipped_image = convolution.double_flip(image) * tau
            for stage, model in enumerate(greedy_iter_models.image):
                arg = kernel - convolution.conv(dual_kernel, scaled_flipped_image)
                tmp = cast(Tensor, model(arg))
                kernel = 2 * tmp - kernel
                dual_kernel = (
                    dual_kernel + sigma * convolution.conv(kernel, image) - sigma_I2
                ) / (1 + sigma)

            # Fix kernel, estimate image
            scaled_flipped_kernel = convolution.double_flip(kernel) * tau
            for stage, model in enumerate(greedy_iter_models.kernel):
                arg = image - convolution.conv(dual_image, scaled_flipped_kernel)
                print(f"{arg.shape=}")
                tmp = cast(Tensor, model(arg))

                image = 2 * tmp - image
                dual_image = (
                    dual_image + sigma * convolution.conv(image, kernel) - sigma_I2
                ) / (1 + sigma)

        return image / torch.max(transmission_maps, torch.tensor(0.1))

    @abstractmethod
    def run(self) -> None: ...

class TrainOrchestrator(Orchestrator[TrainConfig, TrainWeightHandler]):
    def __init__(
        self,
        config: TrainConfig,
        weigth_handler: TrainWeightHandler
    ) -> None:
        super().__init__(config, weigth_handler)
        self.__dataloaders = get_dataloaders(
            "uieb", # TODO: Add as CLIArgument
            config.dataset_dir,
            config.batch_size,
            config.device
        )

        self.__loss = MSELoss()


    def run(self):
        for epoch in range(self._config.n_epochs):
            self.run_train_epoch()

    def run_train_epoch(self):
        for input, ground_truth in self.__dataloaders.training:
            output = self.forward_pass(input)

            loss = self.__loss(output, ground_truth)
            
            print(f"LOSS:\n{loss=}, {type(loss)=}")

class ExecOrchestrator(Orchestrator[ExecConfig, ExecWeightHandler]):
    def run(self) -> None:
        raise NotImplementedError("Hehehe")

def get_orchestrator(config: U2FoldConfig) -> Orchestrator:
    if isinstance(config, TrainConfig):
        return TrainOrchestrator(
            config,
            TrainWeightHandler(config.weight_dir)
        )
    elif isinstance(config, ExecConfig):
        return ExecOrchestrator(
            config,
            ExecWeightHandler(config.weight_dir)
        )
    else:
        raise TypeError(f"Invalid config class.")
