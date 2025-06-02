import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, cast

import torch

from u2fold.models.generic import Model, ModelConfig


class U2FoldWeightTuple[T, U](NamedTuple):
    image: list[T]
    kernel: list[U]


type WeightTreeStructure[T, U] = list[U2FoldWeightTuple[T, U]]
type WeightFileTree = WeightTreeStructure[Path, Path]


@dataclass
class ModelInitBundle[C: ModelConfig]:
    config: C
    class_: type[Model[C]]
    device: Optional[str]


class WeightHandler(ABC):
    """For loading and saving model weights, or first instantiating models."""

    def __init__(self, model_weight_dir: Path) -> None:
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
    def _handle_nonexisting_subdir(self, subdir: Path) -> None: ...

    @abstractmethod
    def _handle_empty_stage_dir(self, stage_dir: Path) -> list[Path]: ...

    def _resolve_stages(self, dir: Path) -> list[Path]:
        stage_weight_files = sorted(dir.iterdir())
        if len(stage_weight_files) == 0:
            return self._handle_empty_stage_dir(dir)

        return stage_weight_files

    def __resolve_greedy_iteration(
        self, dir: Path
    ) -> U2FoldWeightTuple[Path, Path]:
        expected_subdirs = [dir.joinpath("image"), dir.joinpath("kernel")]

        for subdir in expected_subdirs:
            if not subdir.exists():
                self._handle_nonexisting_subdir(subdir)

        return U2FoldWeightTuple(
            image=self._resolve_stages(expected_subdirs[0]),
            kernel=self._resolve_stages(expected_subdirs[1]),
        )

    def __build_filetree(self, model_weight_dir: Path) -> WeightFileTree:
        sorted_greedy_iteration_dirs = sorted(model_weight_dir.iterdir())

        if len(sorted_greedy_iteration_dirs) == 0:
            self._handle_no_greedy_iter_dirs(model_weight_dir)
            return self.__build_filetree(model_weight_dir)

        return [
            self.__resolve_greedy_iteration(greedy_iter_dir)
            for greedy_iter_dir in sorted_greedy_iteration_dirs
        ]

    def __load_model_no_check[C: ModelConfig](
        self, weight_file: Path, model_init_bundle: ModelInitBundle[C]
    ) -> Model[C]:
        model = torch.nn.utils.skip_init(
            model_init_bundle.class_, config=model_init_bundle.config
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
        # Cheating here since Pytorch doesn't know about custom abstractions
        return cast(Model[C], model)

    @abstractmethod
    def _handle_nonexisting_weight_file[C: ModelConfig](
        self, weight_file: Path, model_bundle: ModelInitBundle[C]
    ) -> Model[C]: ...

    def _load_model[C: ModelConfig](
        self, weight_file: Path, model_bundle: ModelInitBundle[C]
    ) -> Model[C]:
        if weight_file.exists():
            return self.__load_model_no_check(weight_file, model_bundle)

        return self._handle_nonexisting_weight_file(weight_file, model_bundle)

    def load_models[C: ModelConfig, D: ModelConfig](
        self,
        image_model_bundle: ModelInitBundle[C],
        kernel_model_bundle: ModelInitBundle[D],
    ) -> WeightTreeStructure[Model[C], Model[D]]:
        return [
            U2FoldWeightTuple(
                image=[
                    self._load_model(weight_file, image_model_bundle)
                    for weight_file in greedy_iter_dir.image
                ],
                kernel=[
                    self._load_model(weight_file, kernel_model_bundle)
                    for weight_file in greedy_iter_dir.kernel
                ],
            )
            for greedy_iter_dir in self._filetree
        ]
