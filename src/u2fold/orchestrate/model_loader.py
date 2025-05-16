import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, cast

import torch

from u2fold.models.generic import Model, ModelConfig
from u2fold.utils.nested_map import nested_map


class WeightHandler(ABC):
    """For loading and saving model weights, or first instantiating models."""

    def __init__(self, model_weight_dir: Path) -> None:
        self._filetree = self.__build_filetree(model_weight_dir)

    @abstractmethod
    def _handle_empty_root_dir(self, dir: Path) -> None: ...

    @abstractmethod
    def _handle_nonexisting_subdir(self, dir: Path) -> None: ...

    @abstractmethod
    def _handle_empty_weight_dir(self, dir: Path) -> list[Path]: ...

    @abstractmethod
    def _handle_empty_weight_file[C: ModelConfig](
        self,
        weight_file: Path,
        model_class: type[Model[C]],
        config: C,
        device: Optional[str],
    ) -> Model[C]: ...

    def _resolve_stages(self, dir: Path) -> list[Path]:
        stage_weight_files = sorted(dir.iterdir())
        if len(stage_weight_files) == 0:
            return self._handle_empty_weight_dir(dir)

        return stage_weight_files

    def __resolve_greedy_iteration(
        self, dir: Path
    ) -> tuple[list[Path], list[Path]]:
        expected_subdirs = [dir.joinpath("image"), dir.joinpath("kernel")]

        for subdir in expected_subdirs:
            if not subdir.exists:
                self._handle_nonexisting_subdir(subdir)

        return (
            self._resolve_stages(expected_subdirs[0]),
            self._resolve_stages(expected_subdirs[1]),
        )

    def __build_filetree(
        self, model_weight_dir: Path
    ) -> list[tuple[list[Path], list[Path]]]:
        sorted_greedy_iteration_dirs = sorted(model_weight_dir.iterdir())

        if len(sorted_greedy_iteration_dirs) == 0:
            self._handle_empty_root_dir(model_weight_dir)

        return [
            self.__resolve_greedy_iteration(greedy_iter_dir)
            for greedy_iter_dir in sorted_greedy_iteration_dirs
        ]

    def _load_model[C: ModelConfig](
        self,
        weight_file: Path,
        model_class: type[Model[C]],
        config: C,
        device: Optional[str],
    ) -> Model[C]:
        if weight_file.exists:
            model = torch.nn.utils.skip_init(model_class, config, device)
            state_dict = torch.load(weight_file, weights_only=True)
            model.load_state_dict(state_dict)
            # Cheating here since Pytorch doesn't know about custom abstractions
            return cast(Model[C], model)
        else:
            return self._handle_empty_weight_file(
                weight_file, model_class, config, device
            )

    def load_models[C: ModelConfig](
        self,
        model_class: type[Model[C]],
        config: C,
        device: Optional[str]
    ) -> list[tuple[list[Model[C]], list[Model[C]]]]:

        result = nested_map(
            lambda weight_file: self._load_model(
                weight_file, model_class, config, device
            ),
            self._filetree
        )

        return cast(list[tuple[list[Model[C]], list[Model[C]]]], result)


class TrainWeightHandler(WeightHandler):
    def __init__(
        self,
        model_weight_dir: Path,
        default_greedy_iterations: int = 3,
        default_stages: int = 3,
    ) -> None:
        self.__default_greedy_iterations = default_greedy_iterations
        self.__default_stages = default_stages
        super().__init__(model_weight_dir)

    @staticmethod
    def __number_of_digits(n: float) -> int:
        return math.ceil(math.log10(n))

    def _handle_empty_root_dir(self, dir: Path) -> None:
        max_digit_number = self.__number_of_digits(
            self.__default_greedy_iterations
        )

        for i in range(self.__default_greedy_iterations):
            dir.joinpath(str(i).ljust(max_digit_number, "0")).mkdir()

    def _handle_nonexisting_subdir(self, dir: Path) -> None:
        dir.mkdir()

    def _handle_empty_weight_dir(self, dir: Path) -> list[Path]:
        max_digit_number = self.__number_of_digits(self.__default_stages)

        return [
            dir.joinpath(f"{str(i).ljust(max_digit_number, '0')}.pt")
            for i in range(self.__default_stages)
        ]

    def _handle_empty_weight_file[C: ModelConfig](
        self,
        weight_file: Path,
        model_class: type[Model[C]],
        config: C,
        device: Optional[str],
    ) -> Model[C]:
        weight_file.parent.mkdir(parents=True, exist_ok=True)
        weight_file.touch(exist_ok=False)

        return model_class(config, device)

    def save_weights(self, weight_file: Path, model: Model) -> None:
        state_dict = model.state_dict()
        torch.save(state_dict, weight_file)

class ExecWeightHandler(WeightHandler):
    def _handle_empty_root_dir(self, dir: Path) -> None:
        errmsg = f"Empty global weight directory: {dir}"
        raise FileNotFoundError(errmsg)

    def _handle_nonexisting_subdir(self, dir: Path) -> None:
        errmsg = f"Empty weight subdirectory: {dir}"
        raise FileNotFoundError(errmsg)

    def _handle_empty_weight_dir(self, dir: Path) -> list[Path]:
        errmsg = f"Empty model weight directory: {dir}"
        raise FileNotFoundError(errmsg)

    def _handle_empty_weight_file[C: ModelConfig](
        self,
        weight_file: Path,
        model_class: type[Model[C]],
        config: C,
        device: Optional[str],
    ) -> Model[C]:
        errmsg = f"Empty model weight file: {dir}"
        raise FileNotFoundError(errmsg)
