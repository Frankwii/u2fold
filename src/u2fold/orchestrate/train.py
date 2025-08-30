from functools import partial
from itertools import chain
from typing import Any, Callable, Literal, NamedTuple, Self, cast, final, override

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from u2fold.data.dataloader_generics.base import U2FoldDataLoader
from u2fold.data.get_dataloaders import get_dataloaders
from u2fold.math.rescale_image import rescale_color
from u2fold.model.common_namespaces import (
    DeterministicComponents,
    ForwardPassResult,
    compute_radiance,
)
from u2fold.model.spec import U2FoldSpec
from u2fold.model.train_spec.spec import TrainSpec
from u2fold.neural_networks.weight_handling.train import TrainWeightHandler
from u2fold.utils.dict_utils import merge_sum, shallow_dict_map
from u2fold.utils.func_utils import chain_calls
from u2fold.utils.get_device import get_device
from u2fold.utils.get_directories import get_tensorboard_log_directory

from .generic import Orchestrator


def no_op(
    *_: Any, **__: Any
): ...  # do nothing  # pyright: ignore[reportAny, reportUnusedParameter]


class LossRegister(NamedTuple):
    overall_loss: Tensor
    granular_loss: dict[str, float]


class EpochLossData(NamedTuple):
    overall_loss: float
    granular_loss: dict[str, float]


class LossAccumulator:
    def __init__(self):
        self.__cumulative_overall_loss: float = 0.0
        self.__cumulative_granular_loss: dict[str, float] = {}
        self.__counter = 0

    def add_register(self, register: LossRegister) -> None:
        self.__counter += 1
        self.__cumulative_overall_loss += register.overall_loss.detach().mean().item()
        self.__cumulative_granular_loss = merge_sum(
            self.__cumulative_granular_loss, register.granular_loss
        )

    def average(self) -> EpochLossData:
        if self.__counter == 0:
            raise ValueError("Add at least one register first!")
        return EpochLossData(
            self.__cumulative_overall_loss / self.__counter,
            shallow_dict_map(
                lambda n: n / self.__counter, self.__cumulative_granular_loss
            ),
        )


class LossComputingData(NamedTuple):
    input: Tensor
    output: ForwardPassResult
    ground_truth: Tensor


@final
class TrainOrchestrator(Orchestrator[TrainWeightHandler]):
    def __init__(
        self, spec: U2FoldSpec[Any], weigth_handler: TrainWeightHandler
    ) -> None:
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

        self._loss_function = self.train_spec.instantiate_loss()

        self._tensorboard_log_dir = get_tensorboard_log_directory(spec)
        self._tensorboard_logger = SummaryWriter(self._tensorboard_log_dir)
        self._model_scheduler = (
            self.train_spec.learning_rate_scheduler_spec.instantiate(
                self._model_optimizer
            )
        )
        self._logger.info("Finished initializing orchestrator.")

    @override
    def run(self) -> float | None:
        self._logger.info("Starting training.")
        min_valiation_loss = torch.inf
        test_loss = cast(EpochLossData, None)  # pyright: ignore[reportInvalidCast]
        for epoch in tqdm(
            range(1, self.train_spec.dataset_spec.n_epochs + 1),
            total=self.train_spec.dataset_spec.n_epochs,
            desc="Epochs",
        ):
            train_loss = self.run_train_epoch(epoch)
            self.tensorboard_log_loss(train_loss, "Train loss", epoch)
            validation_loss_data = self.run_validation_epoch(epoch)
            self._model_scheduler.step(validation_loss_data.overall_loss)
            self.tensorboard_log_loss(validation_loss_data, "Validation loss", epoch)

            if validation_loss_data.overall_loss < min_valiation_loss:
                min_valiation_loss = validation_loss_data.overall_loss
                self._logger.info(
                    f"Validation loss minimum achieved at epoch {epoch}."
                    " saving new weights to disk."
                )
                self._weight_handler.save_models(self._models)

            test_loss = self.run_test_epoch(epoch)
            self.tensorboard_log_loss(test_loss, "Test loss", epoch)

        return test_loss.overall_loss

    def run_test_epoch(self, epoch: int) -> EpochLossData:
        with torch.no_grad():
            return self.__run_epoch(
                phase="test",
                pre_computing_loss_hook=no_op,
                post_computing_loss_hook=no_op,
                epoch=epoch,
            )

    def run_validation_epoch(
        self,
        epoch: int,
    ) -> EpochLossData:
        with torch.no_grad():
            return self.__run_epoch(
                phase="validation",
                pre_computing_loss_hook=no_op,
                post_computing_loss_hook=no_op,
                epoch=epoch,
            )

    def run_train_epoch(self, epoch: int) -> EpochLossData:
        def refresh_gradients():
            self._model_optimizer.zero_grad()

        def update_weights_and_training_state(loss: Tensor):
            loss.backward()
            self._model_optimizer.step()

        return self.__run_epoch(
            phase="training",
            epoch=epoch,
            pre_computing_loss_hook=refresh_gradients,
            post_computing_loss_hook=update_weights_and_training_state,
        )

    def __run_epoch(
        self,
        phase: Literal["training", "validation", "test"],
        epoch: int,
        pre_computing_loss_hook: Callable[[], None],
        post_computing_loss_hook: Callable[[Tensor], None],
    ) -> EpochLossData:
        dataloader: U2FoldDataLoader[Any, Any, *tuple[Any, ...]] = getattr(
            self._dataloaders, phase
        )
        n_elements = len(dataloader)
        loss_accumulator = LossAccumulator()
        process_batch = partial(
            self.__process_batch,
            pre_computing_loss_hook=pre_computing_loss_hook,
            post_computing_loss_hook=post_computing_loss_hook,
            loss_accumulator=loss_accumulator,
        )

        formatted_phase = phase.capitalize()
        dataloader_iter = iter(dataloader)
        first_input, first_ground_truth = next(dataloader_iter)

        def log_results(loss_computing_data: LossComputingData) -> None:
            return self.tensorboard_log_result(
                loss_computing_data.output, formatted_phase=formatted_phase, epoch=epoch
            )

        log_first_epoch_components = (
            partial(
                self.tensorboard_log_first_epoch_components,
                formatted_phase=formatted_phase,
            )
            if epoch == 1
            else no_op
        )

        process_batch(
            input=first_input,
            ground_truth=first_ground_truth,
            output_hook=chain_calls(log_results, log_first_epoch_components),
        )

        for input, ground_truth in tqdm(
            dataloader_iter, desc=formatted_phase, initial=1, total=n_elements
        ):
            process_batch(input=input, ground_truth=ground_truth, output_hook=no_op)

        return loss_accumulator.average()

    def __process_batch(
        self,
        input: Tensor,
        ground_truth: Tensor,
        loss_accumulator: LossAccumulator,
        pre_computing_loss_hook: Callable[[], None],
        post_computing_loss_hook: Callable[[Tensor], None],
        output_hook: Callable[[LossComputingData], None],
    ) -> None:
        pre_computing_loss_hook()

        output = self.forward_pass(input)
        output_hook(LossComputingData(input, output, ground_truth))
        loss_register = self.__compute_batch_loss(output, ground_truth)
        loss_accumulator.add_register(loss_register)

        post_computing_loss_hook(loss_register.overall_loss)

    def __compute_batch_loss(
        self, output: ForwardPassResult, ground_truth: Tensor
    ) -> LossRegister:
        loss = self._loss_function(output, ground_truth)
        granular_loss = self._loss_function.get_last_losses()

        return LossRegister(loss, granular_loss)

    def tensorboard_log_first_epoch_components(
        self, loss_computing_data: LossComputingData, formatted_phase: str
    ) -> None:
        self.tensorboard_log_image(
            loss_computing_data.input, f"{formatted_phase}/Input", 1
        )

        first_radiance_estimation = compute_radiance(
            loss_computing_data.output.primal_variable_history[0],
            loss_computing_data.output.deterministic_components.transmission_map.clamp(
                0.1
            ),
        )

        self.tensorboard_log_image(
            first_radiance_estimation,
            f"{formatted_phase}/First_radiance",
            1,
        )
        self.tensorboard_log_image(
            loss_computing_data.ground_truth, f"{formatted_phase}/Ground_truth", 1
        )
        self.tensorboard_log_image(
            loss_computing_data.output.deterministic_components.background_light,
            f"{formatted_phase}/Background_light",
            1,
        )
        self.tensorboard_log_image(
            loss_computing_data.output.deterministic_components.transmission_map,
            f"{formatted_phase}/Transmission_map",
            1,
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

    def tensorboard_log_loss(
        self, loss_log: EpochLossData, tag: str, epoch: int
    ) -> None:
        for loss_name, value in loss_log.granular_loss.items():
            self._tensorboard_logger.add_scalar(f"{tag}/{loss_name}", value, epoch)
        self._tensorboard_logger.add_scalar(tag, loss_log.overall_loss, epoch)

    def tensorboard_log_result(
        self, result: ForwardPassResult, formatted_phase: str, epoch: int
    ) -> None:
        clamped_tm = result.deterministic_components.transmission_map.clamp(min=0.1)
        for iteration, primal_variable in enumerate(
            result.primal_variable_history[1:], start=1
        ):
            restored_image = compute_radiance(primal_variable, clamped_tm)
            self.tensorboard_log_image(
                restored_image, f"{formatted_phase}/Output/{iteration}", epoch
            )

        for iteration, kernel in enumerate(result.kernel_history):
            self.tensorboard_log_image(
                kernel, f"{formatted_phase}/Kernel/{iteration}", epoch
            )

    def tensorboard_log_image(self, image: Tensor, tag: str, epoch: int) -> None:
        self._tensorboard_logger.add_images(tag, image.detach(), global_step=epoch)

    def tensorboard_log_text(self, text: str, tag: str) -> None:
        self._tensorboard_logger.add_text(tag, text)
