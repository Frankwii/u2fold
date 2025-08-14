import math
from pathlib import Path

import torch
from .generic import (
    ModelInitBundle,
    WeightHandler,
    WeightTreeStructure,
)

from u2fold.neural_networks.generic import NeuralNetwork
from u2fold.model.neural_network_spec import NeuralNetworkSpec


class TrainWeightHandler(WeightHandler):
    def __init__(
        self,
        model_weight_dir: Path,
        default_greedy_iterations: int = 3,
        default_stages: int = 3,
    ) -> None:
        self.__default_greedy_iterations = default_greedy_iterations
        self.__default_stages = default_stages
        model_weight_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(model_weight_dir)

    @staticmethod
    def __generate_ranging_file_names(min: int, max: int) -> list[str]:
        max_digits = int(math.log10(max)) + 1
        return [str(i).rjust(max_digits, "0") for i in range(min, max + 1)]

    def _handle_no_greedy_iter_dirs(self, root_dir: Path) -> None:
        self._logger.debug(
            f"Found no greedy iteration weight directories under {root_dir}"
            f"when parsing weight tree. Creating "
            f"{self.__default_greedy_iterations} of them."
        )

        dir_names = self.__generate_ranging_file_names(
            1, self.__default_greedy_iterations
        )

        for name in dir_names:
            root_dir.joinpath(name).mkdir()

    def _handle_empty_stage_dir(self, stage_dir: Path) -> list[Path]:
        self._logger.debug(
            f"Found no weight files at stage level in {stage_dir}. Creating"
            f" {self.__default_stages} entries in the filetree."
        )

        file_names = self.__generate_ranging_file_names(
            1, self.__default_stages
        )

        return [stage_dir.joinpath(f"{name}.pt") for name in file_names]

    def _handle_nonexisting_weight_file[C: NeuralNetworkSpec](
        self, weight_file: Path, model_bundle: ModelInitBundle[C]
    ) -> NeuralNetwork[C]:
        self._logger.debug(
            f"Found no weight file at {weight_file}. Initializing"
            " model weights randomly."
        )

        return model_bundle.class_(model_bundle.config, model_bundle.device)

    def __save_weights[C: NeuralNetworkSpec](self, weight_file: Path, model: NeuralNetwork[C]) -> None:
        self._logger.debug(
            f"Saving state dict for model {type(model).__name__}"
            f" at {weight_file}."
        )
        state_dict = model.state_dict()
        torch.save(state_dict, weight_file)

    def save_models[C: NeuralNetworkSpec](self, models: WeightTreeStructure[NeuralNetwork[C]]) -> None:
        assert len(models) == len(self._filetree), (
            "Models and filetree not paired"
        )
        for stage_models, dir in zip(models, self._filetree):
            assert len(stage_models) == len(dir), (
                "Models and filetree not paired"
            )

        for stage_models, dir in zip(models, self._filetree):
            for model, file in zip(stage_models, dir):
                self.__save_weights(file, model)


