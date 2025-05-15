from abc import ABC
from pathlib import Path

import torch

from u2fold.cli_parsing.config import TrainingConfig
from u2fold.orchestrate.greedy_iteration_models import GreedyIterationModels
from u2fold.utils.track import tag

from .loading_models import load_models


class SharedOrchestrator(ABC):
    def _load_models(
            self,
            model_name: str,
            weight_files_dir: Path
        ) -> list[GreedyIterationModels]:
        """
        Load the given model based on the contents of the specified directory
        in `weight_files_dir`.


        Args:
            model_name: The name of the model, as specified in its tag.
                The f-string f"model/{model_name}" should evaluate to said tag.
            weight_files_dir: The path to the weights directory. This should
                contain only other directories, and those, in turn, two other
                directories which contain the actual weight files (in total,
                a 3-deep tree, with the root being `weight_files_dir`).
                The number of subdirectories in `weight_files_dir` corresponds
                to the number of "greedy penalty" iterations to be performed.
                Each subdirectory at this depth should contain exactly two other
                subdirectories: one called "image" and the other "kernel".
                Finally, these subdirectories contain the actual weight files:
                one for each stage in that iteration.

        Returns:
            models: A list containing all of the loaded models, in an order
                similar to `weight_files_dir`: first axis corresponds to the
                "greedy penalty" iteration; second, to the estimated function
                (either image or kernel) and the third axis corresponds to the
                stage iteration.

                In this way, the natural order of iterating through `models` is
                the order of the algorithm.
        """

        return load_models(model_name, weight_files_dir)


@tag("orchestrate/train")
class TrainingOrchestrator(SharedOrchestrator):
    def __init__(self, config: TrainingConfig) -> None:
        self.__execution_log_dir = config.execution_log_dir
        self.__tensorboard_log_dir = config.tensorboard_log_dir
        self.__dataset_dir = config.dataset_dir
        self.__n_epochs = config.n_epochs

        self.__data_loader = self.__instantiate_dataloader(self.__dataset_dir)
        self.__models = self._load_models(
            config.model_name, config.weight_file_dir
        )

        # TODO: logger?

    def __instantiate_dataloader(
            self,
            dataset_dir: Path
        ) -> torch.utils.data.DataLoader:
        raise NotImplementedError("WIP")

    def train(self):
        for epoch in range(self.__n_epochs):
            for greedy_iteration in range(len(self.__n_epochs)):
                # Eval image
                raise NotImplementedError
