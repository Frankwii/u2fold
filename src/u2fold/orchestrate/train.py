from itertools import chain
import shlex
import shutil
import subprocess

import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from u2fold.math.rescale_image import rescale_color
import u2fold.orchestrate.functional as F
from u2fold.config_parsing.config_dataclasses import TrainConfig
from u2fold.data.get_dataloaders import get_dataloaders
from u2fold.models.weight_handling.train import TrainWeightHandler

from .generic import Orchestrator


class TrainOrchestrator(Orchestrator[TrainConfig, TrainWeightHandler]):
    def __init__(
        self, config: TrainConfig, weigth_handler: TrainWeightHandler
    ) -> None:
        super().__init__(config, weigth_handler)
        self.__dataloaders = get_dataloaders(
            "uieb",  # TODO: Add as CLIArgument
            config.dataset_dir,
            config.batch_size,
            config.device,
        )

        self._model_optimizer = Adam(
            torch.nn.ModuleList(chain.from_iterable(self._models)).parameters(), lr=0.1
        )

        self.__loss_function = F.loss
        self._tensorboard_logger = SummaryWriter(config.tensorboard_log_dir)
        self._model_scheduler = StepLR(
            self._model_optimizer,
            gamma = 0.5 ** 0.5,
            step_size=50
        )
        self._logger.info(f"Finished initializing orchestrator.")

    def run(self):
        print(f"Running!")
        if self._config.tensorboard_log_dir.exists():
            self._logger.warning(
                "Found an existing tensorboard log directory. Emptying it."
            )
            shutil.rmtree(self._config.tensorboard_log_dir)
            self._config.tensorboard_log_dir.mkdir()

        self._logger.debug("Starting tensorboard process")
        tensorboard_process = subprocess.Popen(
            shlex.split(
                f"tensorboard --port 6066 --logdir {self._config.tensorboard_log_dir}"
            )
        )

        self._logger.info("Starting training.")
        min_valiation_loss = torch.inf
        for epoch in tqdm(
            range(1, self._config.n_epochs + 1),
            total=self._config.n_epochs,
            desc="Training epochs",
        ):
            print(f"Epoch {epoch}")
            train_loss = self.run_train_epoch()
            validation_loss = self.run_validation_epoch()
            self._model_scheduler.step()

            if validation_loss < min_valiation_loss:
                min_valiation_loss = validation_loss
                self._logger.info(
                    f"Validation loss minimum achieved at epoch {epoch}."
                    " saving new weights to disk."
                )
                self._weight_handler.save_models(self._models)

            test_loss = self.run_test_epoch(epoch)

            self.tensorboard_log_loss(train_loss, "Train loss", epoch)
            self.tensorboard_log_loss(validation_loss, "Validation loss", epoch)
            self.tensorboard_log_loss(test_loss, "Test loss", epoch)
        msg = (
            "Traning has finished, but the tensorboard process will be kept"
            " running."
        )
        self._logger.warning(msg)
        print(msg + " Please kill the process to stop it.")

        tensorboard_process.wait()

    def run_test_epoch(self, epoch: int) -> float:
        cumulative_loss = 0.0

        with torch.no_grad():
            test_iter = iter(
                tqdm(
                    self.__dataloaders.test,
                    desc="Test",
                    total=len(self.__dataloaders.test),
                )
            )

            first_input, first_ground_truth = next(test_iter)

            output = self.forward_pass(first_input)

            loss = self.__loss_function(output, first_ground_truth)
            cumulative_loss += loss.detach().item()

            restored_image = rescale_color(
                output.primal_variable_history[-1]
                / output.deterministic_components.transmission_map.clamp(
                    min=0.01
                )
            )
            self.tensorboard_log_image(restored_image, "Test/Output", epoch)
            self.tensorboard_log_image(
                output.kernel_history[-1], "Test/Kernel", epoch
            )

            if epoch == 1:
                radiance_estimation = rescale_color(
                    output.deterministic_components.fidelity
                    / output.deterministic_components.transmission_map.clamp(
                        min=0.01
                    )
                )

                self.tensorboard_log_image(first_input, "Test/Input", epoch)
                self.tensorboard_log_image(
                    radiance_estimation,
                    "Test/First_radiance",
                    epoch,
                )
                self.tensorboard_log_image(
                    first_ground_truth, "Test/Ground_truth", epoch
                )
                self.tensorboard_log_image(
                    output.deterministic_components.background_light,
                    "Test/Background_light",
                    epoch,
                )
                self.tensorboard_log_image(
                    output.deterministic_components.transmission_map,
                    "Test/Transmission_map",
                    epoch,
                )

            for input, ground_truth in test_iter:
                output = self.forward_pass(input)

                loss = self.__loss_function(output, ground_truth)
                cumulative_loss += loss.detach().item()

        return cumulative_loss / len(self.__dataloaders.test)

    def run_validation_epoch(self) -> float:
        cumulative_loss = 0.0

        with torch.no_grad():
            for input, ground_truth in tqdm(
                self.__dataloaders.validation,
                desc="Validation",
                total=len(self.__dataloaders.validation),
            ):
                output = self.forward_pass(input)

                loss = self.__loss_function(output, ground_truth)
                cumulative_loss += loss.detach().item()

        return cumulative_loss / len(self.__dataloaders.validation)

    def run_train_epoch(self) -> float:
        cumulative_loss = 0.0

        for input, ground_truth in tqdm(
            self.__dataloaders.training,
            desc="Training",
            total=len(self.__dataloaders.training),
        ):
            self._model_optimizer.zero_grad()
            output = self.forward_pass(input)

            loss = self.__loss_function(output, ground_truth)

            cumulative_loss += loss.detach().item()

            loss.backward()
            self._model_optimizer.step()

        return cumulative_loss / len(self.__dataloaders.training)

    def tensorboard_log_image(
        self, image: Tensor, tag: str, epoch: int
    ) -> None:
        self._tensorboard_logger.add_images(
            tag, image.detach(), global_step=epoch
        )

    def tensorboard_log_hist(self, image: Tensor, tag: str) -> None:
        img = image.detach()
        for i in range(image.size(0)):
            for c in range(image.size(1)):
                self._tensorboard_logger.add_histogram(
                    f"{tag}_channel{c}_image{i}",
                    img[i][c].flatten(),
                    global_step=i,
                )

    def tensorboard_log_loss(self, val: float, tag: str, epoch: int) -> None:
        return self._tensorboard_logger.add_scalar(tag, val, epoch)

    def tensorboard_log_text(self, text: str, tag: str) -> None:
        self._tensorboard_logger.add_text(tag, text)
