from functools import partial
from itertools import chain
from typing import Any, Callable, Literal, cast, final, override

import torch
from torch import Tensor
from tqdm import tqdm

from u2fold.data.dataloader_generics.base import U2FoldDataLoader
from u2fold.data.get_dataloaders import get_dataloaders
from u2fold.model.spec import U2FoldSpec
from u2fold.model.train_spec.spec import TrainSpec
from u2fold.neural_networks.weight_handling.train import TrainWeightHandler
from .metric_log_computing import compute_metrics_to_log
from u2fold.orchestrate.train.tensorboard_logger import TensorboardLogger
from .auxiliary_methods_and_classes import LossComputingData, MetricAccumulator, MetricRegister, no_op
from u2fold.model.common_namespaces import EpochMetricData
from u2fold.utils.func_utils import chain_calls
from u2fold.utils.get_device import get_device
from u2fold.utils.get_directories import get_tensorboard_log_directory

from u2fold.orchestrate.generic import Orchestrator

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
        self._tensorboard_logger = TensorboardLogger(self._tensorboard_log_dir)
        self._model_scheduler = (
            self.train_spec.learning_rate_scheduler_spec.instantiate(
                self._model_optimizer
            )
        )
        self._logger.info("Finished initializing orchestrator.")

    @override
    def run(self) -> EpochMetricData:
        self._logger.info("Starting training.")
        min_valiation_loss = torch.inf
        test_metrics = cast(EpochMetricData, None)  # pyright: ignore[reportInvalidCast]
        for epoch in tqdm(
            range(1, self.train_spec.dataset_spec.n_epochs + 1),
            total=self.train_spec.dataset_spec.n_epochs,
            desc="Epochs",
        ):
            train_metrics = self.run_train_epoch(epoch)
            self._tensorboard_logger.log_metrics(train_metrics, "Train loss", epoch)
            validation_metrics = self.run_validation_epoch(epoch)
            self._model_scheduler.step(validation_metrics.overall_loss)
            self._tensorboard_logger.log_metrics(validation_metrics, "Validation loss", epoch)

            if validation_metrics.overall_loss < min_valiation_loss:
                min_valiation_loss = validation_metrics.overall_loss
                self._logger.info(
                    f"Validation loss minimum achieved at epoch {epoch}."
                    " Saving new weights to disk."
                )
                self._weight_handler.save_models(self._models)

            test_metrics = self.run_test_epoch(epoch)
            self._tensorboard_logger.log_metrics(test_metrics, "Test loss", epoch)

        return test_metrics

    def run_train_epoch(self, epoch: int) -> EpochMetricData:
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

    def run_validation_epoch(
        self,
        epoch: int,
    ) -> EpochMetricData:
        with torch.no_grad():
            return self.__run_epoch(
                phase="validation",
                pre_computing_loss_hook=no_op,
                post_computing_loss_hook=no_op,
                epoch=epoch,
            )

    def run_test_epoch(self, epoch: int) -> EpochMetricData:
        with torch.no_grad():
            return self.__run_epoch(
                phase="test",
                pre_computing_loss_hook=no_op,
                post_computing_loss_hook=no_op,
                epoch=epoch,
            )

    def __run_epoch(
        self,
        phase: Literal["training", "validation", "test"],
        epoch: int,
        pre_computing_loss_hook: Callable[[], None],
        post_computing_loss_hook: Callable[[Tensor], None],
    ) -> EpochMetricData:
        dataloader: U2FoldDataLoader[Any, Any, *tuple[Any, ...]] = getattr(
            self._dataloaders, phase
        )
        n_elements = len(dataloader)
        loss_accumulator = MetricAccumulator()
        process_batch = partial(
            self.__process_batch,
            pre_computing_loss_hook=pre_computing_loss_hook,
            post_computing_loss_hook=post_computing_loss_hook,
            loss_accumulator=loss_accumulator,
        )

        formatted_phase = phase.capitalize()
        dataloader_iter = iter(dataloader)
        first_input, first_ground_truth = next(dataloader_iter)

        def log_results(loss_computing_data: LossComputingData, /) -> None:
            return self._tensorboard_logger.log_result(
                loss_computing_data.output, formatted_phase=formatted_phase, epoch=epoch
            )

        log_first_epoch_components = (
            partial(
                self._tensorboard_logger.log_first_epoch_components,
                formatted_phase=formatted_phase,
            )
        )
        output_hook = log_results if epoch > 1 else chain_calls(log_results, log_first_epoch_components)

        process_batch(
            input=first_input,
            ground_truth=first_ground_truth,
            output_hook=output_hook
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
        loss_accumulator: MetricAccumulator,
        pre_computing_loss_hook: Callable[[], None],
        post_computing_loss_hook: Callable[[Tensor], None],
        output_hook: Callable[[LossComputingData], None],
    ) -> None:
        pre_computing_loss_hook()

        output = self.forward_pass(input)
        loss_computing_data = LossComputingData(input, output, ground_truth)
        output_hook(loss_computing_data)
        loss_register = self.__compute_batch_loss(loss_computing_data)
        loss_accumulator.add_register(loss_register)

        post_computing_loss_hook(loss_register.overall_loss)

    def __compute_batch_loss(
        self, data: LossComputingData
    ) -> MetricRegister:
        loss = self._loss_function(data.output, data.ground_truth)
        granular_loss = self._loss_function.get_last_losses()
        metrics = compute_metrics_to_log(data)

        return MetricRegister(loss, granular_loss, metrics)
