import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from u2fold.model.neural_network_spec import NeuralNetworkSpec
from u2fold.neural_networks.generic import NeuralNetwork

type WeightTreeStructure[T] = tuple[tuple[T, ...], ...]
type WeightFileTree = WeightTreeStructure[Path]


@dataclass
class ModelInitBundle[C: NeuralNetworkSpec]:
    spec: C
    class_: type[NeuralNetwork[C]]
    device: str | None


class WeightHandler(ABC):
    """For loading and saving model weights, or first instantiating models."""

    def __init__(
        self,
        model_weight_dir: Path,
        share_weights: bool,
        greedy_iterations: int,
        stages: int
    ) -> None:

        self._greedy_iterations = greedy_iterations
        self._stages = stages
        self._share_weights = share_weights
        self._logger = logging.getLogger(__name__)

        self._logger.info(
            f"Attempting to parse weight tree under {model_weight_dir}."
        )

        self._filetree = self.__build_filetree(model_weight_dir)

        self._logger.info(
            f"Succesfully parsed weight tree under {model_weight_dir}."
            f" Number of greedy iterations: {len(self._filetree)}."
        )

    @abstractmethod
    def _handle_no_greedy_iter_dirs(self, root_dir: Path) -> None: ...

    @abstractmethod
    def _handle_empty_stage_dir(self, stage_dir: Path) -> tuple[Path, ...]: ...

    @abstractmethod
    def _handle_nonexisting_weight_file[C: NeuralNetworkSpec](
        self, weight_file: Path, model_bundle: ModelInitBundle[C]
    ) -> NeuralNetwork[C]: ...

    def __build_filetree(self, model_weight_dir: Path) -> WeightFileTree:
        sorted_greedy_iteration_dirs = sorted(model_weight_dir.iterdir())

        if len(sorted_greedy_iteration_dirs) == 0:
            self._handle_no_greedy_iter_dirs(model_weight_dir)
            return self.__build_filetree(model_weight_dir)

        return tuple(
            self.__resolve_greedy_iteration(greedy_iter_dir)
            for greedy_iter_dir in sorted_greedy_iteration_dirs
        )

    def __resolve_greedy_iteration(self, dir: Path) -> tuple[Path, ...]:
        stage_weight_files = tuple(sorted(dir.iterdir()))
        if len(stage_weight_files) == 0:
            return self._handle_empty_stage_dir(dir)

        return stage_weight_files

    def __load_model_no_check[C: NeuralNetworkSpec](
        self, weight_file: Path, model_init_bundle: ModelInitBundle[C]
    ) -> NeuralNetwork[C]:
        model = torch.nn.utils.skip_init(
            model_init_bundle.class_, spec=model_init_bundle.spec
        ).to(model_init_bundle.device)
        try:
            state_dict = torch.load(weight_file, weights_only=True)
        except EOFError as e:
            errmsg = f"Corrupt state dict file at {weight_file}. Error:\n{e}"
            raise EOFError(errmsg)

        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            name = model_init_bundle.class_.__name__
            errmsg = (
                f"Invalid state dict file for model {name} at {weight_file}."
                f" Error:\n{e}"
            )
            raise RuntimeError(errmsg)
        # Cheating here since pytorch doesn't know about custom abstractions
        return cast(NeuralNetwork[C], model)

    def _load_model[C: NeuralNetworkSpec](
        self, weight_file: Path, model_bundle: ModelInitBundle[C]
    ) -> NeuralNetwork[C]:
        if weight_file.exists():
            return self.__load_model_no_check(weight_file, model_bundle)

        return self._handle_nonexisting_weight_file(weight_file, model_bundle)

    def load_models[C: NeuralNetworkSpec](
        self,
        model_bundle: ModelInitBundle[C],
    ) -> WeightTreeStructure[NeuralNetwork[C]]:
        if self._share_weights:
            model = self._load_model(self._filetree[0][0], model_bundle)

            return ((model,) * self._stages,) * self._greedy_iterations
        return tuple(
            tuple(
                self._load_model(weight_file, model_bundle)
                for weight_file in greedy_iter_dir
            )
            for greedy_iter_dir in self._filetree
        )
