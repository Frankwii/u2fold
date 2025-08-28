from itertools import chain
from typing import cast, final, override

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from u2fold.data.get_dataloaders import get_dataloaders
from u2fold.math.rescale_image import rescale_color
from u2fold.model.spec import U2FoldSpec
from u2fold.model.train_spec.spec import TrainSpec
from u2fold.neural_networks.weight_handling.train import TrainWeightHandler
from u2fold.utils.get_device import get_device
from u2fold.utils.get_directories import get_tensorboard_log_directory

from .generic import Orchestrator

@final
class TrainOrchestrator(Orchestrator[TrainWeightHandler]):
    def __init__(self, spec: U2FoldSpec, weigth_handler: TrainWeightHandler) -> None:
        super().__init__(spec, weigth_handler)
        self.train_spec = cast(TrainSpec, spec.mode_spec)

        dataset_spec = self.train_spec.dataset_spec
        self._dataloaders = get_dataloaders(
            dataset=dataset_spec.name,
            dataset_path=dataset_spec.path,
            batch_size=dataset_spec.batch_size,
            device=get_device(),
        )

        self._model_optimizer = self.train_spec.optimizer_spec.instantiate(
            torch.nn.ModuleList(chain.from_iterable(self._models)).parameters()
        )

        self.__loss_function = self.train_spec.instantiate_loss()

        self._tensorboard_log_dir = get_tensorboard_log_directory(
            spec.neural_network_spec
        )
        self._tensorboard_logger = SummaryWriter(self._tensorboard_log_dir)
        self._model_scheduler = (
            self.train_spec.learning_rate_scheduler_spec.instantiate(
                self._model_optimizer
            )
        )
        self._logger.info(f"Finished initializing orchestrator.")

    @override
    def run(self) -> float | None:
        self._logger.info("Starting training.")
        min_valiation_loss = torch.inf
        test_loss = None
        for epoch in tqdm(
            range(1, self.train_spec.dataset_spec.n_epochs + 1),
            total=self.train_spec.dataset_spec.n_epochs,
            desc="Training epochs",
        ):
            train_loss = self.run_train_epoch()
            validation_loss = self.run_validation_epoch()
            self._model_scheduler.step(validation_loss)

            if validation_loss < min_valiation_loss:
                min_valiation_loss = validation_loss
                self._logger.info(
                    f"Validation loss minimum achieved at epoch {epoch}."
                    " saving new weights to disk."
                )
                self._weight_handler.save_models(self._models)

            test_loss = self.run_test_epoch(epoch).item()

            self.tensorboard_log_loss(train_loss, "Train loss", epoch)
            self.tensorboard_log_loss(validation_loss, "Validation loss", epoch)
            self.tensorboard_log_loss(test_loss, "Test loss", epoch)

        return test_loss

    def run_test_epoch(self, epoch: int) -> Tensor:
        cumulative_loss = torch.tensor(0.0)

        with torch.no_grad():
            test_iter = iter(
                tqdm(
                    self._dataloaders.test,
                    desc="Test",
                    total=len(self._dataloaders.test),
                )
            )

            first_input, first_ground_truth = next(test_iter)

            output = self.forward_pass(first_input)

            loss = self.__loss_function(output, first_ground_truth)
            cumulative_loss += loss.detach().item()

            for iter, (primal_variable, kernel) in enumerate(zip(output.primal_variable_history, output.kernel_history), start=1):
                restored_image = rescale_color(
                    primal_variable / output.deterministic_components.transmission_map.clamp(min=0.01)
                )
                self.tensorboard_log_image(restored_image, f"Test/Output/{iter}", epoch)
                self.tensorboard_log_image(kernel, f"Test/Kernel/{iter}", epoch)

            if epoch == 1:
                radiance_estimation = rescale_color(
                    output.deterministic_components.fidelity
                    / output.deterministic_components.transmission_map.clamp(min=0.01)
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

        return cumulative_loss / len(self._dataloaders.test)

    def run_validation_epoch(self) -> Tensor:
        cumulative_loss = torch.tensor(0.0)

        with torch.no_grad():
            for input, ground_truth in tqdm(
                self._dataloaders.validation,
                desc="Validation",
                total=len(self._dataloaders.validation),
            ):
                output = self.forward_pass(input)

                loss = self.__loss_function(output, ground_truth)
                cumulative_loss += loss.detach().item()

        return cumulative_loss / len(self._dataloaders.validation)

    def run_train_epoch(self) -> Tensor:
        cumulative_loss = torch.tensor(0.0)

        for input, ground_truth in tqdm(
            self._dataloaders.training,
            desc="Training",
            total=len(self._dataloaders.training),
        ):
            self._model_optimizer.zero_grad()
            output = self.forward_pass(input)

            loss = self.__loss_function(output, ground_truth)

            cumulative_loss += loss.detach().item()

            loss.backward()
            self._model_optimizer.step()

        return cumulative_loss / len(self._dataloaders.training)

    def tensorboard_log_image(self, image: Tensor, tag: str, epoch: int) -> None:
        self._tensorboard_logger.add_images(
            tag,
            image.detach(),
            global_step=epoch
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

    def tensorboard_log_loss(self, val: float | Tensor, tag: str, epoch: int) -> None:
        if isinstance(val, Tensor):
            val = val.item()
        last_losses = self.__loss_function.last_losses
        for loss_name, value in last_losses.items():
            self._tensorboard_logger.add_scalar(f"{tag}/{loss_name}", value, epoch)
        self._tensorboard_logger.add_scalar(tag, val, epoch)

    def tensorboard_log_text(self, text: str, tag: str) -> None:
        self._tensorboard_logger.add_text(tag, text)
