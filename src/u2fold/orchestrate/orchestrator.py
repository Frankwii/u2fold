import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from logging import getLogger
from typing import Callable, NamedTuple, Optional, cast

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Adam, Optimizer
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image, to_tensor
from tqdm import tqdm

import u2fold.math.linear_operators as lin
import u2fold.math.proximities as prox
from u2fold.config_parsing.config_dataclasses import (
    ExecConfig,
    TrainConfig,
    U2FoldConfig,
)
from u2fold.data import get_dataloaders
from u2fold.math import convolution
from u2fold.math.background_light_estimation import estimate_background_light
from u2fold.math.compute_first_term import (
    estimate_fidelity_and_transmission_map,
)
from u2fold.math.primal_dual import PrimalDualSchema
from u2fold.models.weight_handling import ExecWeightHandler, TrainWeightHandler
from u2fold.models.weight_handling.generic import ModelInitBundle, WeightHandler
from u2fold.utils.track import get_from_tag


class ForwardPassResult(NamedTuple):
    primal_variable: Tensor
    kernel: Tensor
    transmission_map: Tensor
    fidelity: Tensor


class KernelBundle(NamedTuple):
    preimage: Parameter
    preimage_to_kernel_mapping: Callable[[Tensor], Tensor]
    optimizer: Optimizer

    def compute_kernel(self):
        return self.preimage_to_kernel_mapping(self.preimage)


class PrimalDualBundle(NamedTuple):
    schema: PrimalDualSchema
    primal_variable: Tensor
    dual_variable: Tensor


class DeterministicComponents(NamedTuple):
    fidelity: Tensor
    transmission_map: Tensor


def unroll_kernel(kernel_preimage: Tensor, kernel_size: int) -> Tensor:
    return F.softmax(kernel_preimage, dim=-1).reshape(
        kernel_preimage.size(0),
        kernel_preimage.size(1),
        kernel_size,
        kernel_size,
    )


class Orchestrator[T: U2FoldConfig, W: WeightHandler](ABC):
    def __init__(self, config: T, weigth_handler: W) -> None:
        self._logger = getLogger(__name__)
        self._logger.info(f"Initializing orchestrator.")
        self._config = config
        self._weight_handler = weigth_handler
        image_bundle = ModelInitBundle(
            config.model_config,
            get_from_tag(f"model/{config.model_name}"),
            device=config.device,
        )

        self._models = self._weight_handler.load_models(image_bundle)

        self._model_optimizer = Adam(
            torch.nn.ModuleList(chain.from_iterable(self._models)).parameters()
        )

        self.__kernel_size = 7
        self.__kernel_centrality_weight = 1e-4
        self.__kernel_centrality_penalty = (
            self.initialize_kernel_centrality_penalty(
                self.__kernel_size, self.__kernel_centrality_weight
            )
        )

    def compute_deterministic_components(
        self, input: Tensor
    ) -> DeterministicComponents:
        with torch.no_grad():
            background_light = estimate_background_light(input)

            transmission_map_config = (
                self._config.transmission_map_estimation_config
            )
            fidelity, transmission_map = estimate_fidelity_and_transmission_map(
                input,
                background_light,
                transmission_map_config.patch_radius,
                transmission_map_config.saturation_coefficient,
                transmission_map_config.regularization_coefficient,
            )

        return DeterministicComponents(fidelity, transmission_map)

    def initialize_primal_dual(self, fidelity: Tensor) -> PrimalDualBundle:
        primal_dual_schema = (
            PrimalDualSchema()
            .with_primal_proximity(prox.identity, True)
            .with_primal_argument(0)
            .with_dual_proximity(prox.conjugate_shifted_square_L2_norm, True)
            .with_dual_argument(fidelity)
            .with_step_sizes(
                self._config.unfolded_step_size, self._config.step_size
            )
            .with_linear_operator(lin.convolution, lin.conjugate_convolution)
        )

        return PrimalDualBundle(
            primal_dual_schema,
            fidelity.clone(),
            fidelity.clone(),
        )

    def initialize_kernel(
        self,
        batch_size: int,
        kernel_size: int,
        learning_rate: Optional[float] = None,
    ) -> KernelBundle:
        kernel_preimage = torch.full(
            (batch_size, 3, kernel_size, kernel_size), -1e3
        )
        kernel_preimage[..., kernel_size // 2, kernel_size // 2] = 0

        kernel_preimage = Parameter(
            kernel_preimage.reshape(batch_size, 3, -1).to(self._config.device),
            requires_grad=True,
        )

        optimizer_kwargs = {}
        if learning_rate is not None:
            optimizer_kwargs["lr"] = learning_rate

        return KernelBundle(
            preimage=kernel_preimage,
            preimage_to_kernel_mapping=partial(
                unroll_kernel, kernel_size=kernel_size
            ),
            optimizer=Adam([kernel_preimage], **optimizer_kwargs),
        )

    def initialize_kernel_centrality_penalty(
        self, size: int, weight: float
    ) -> Tensor:
        center = size // 2

        row = col = torch.arange(size).unsqueeze(1)
        col = col.transpose(0, 1)

        distances_to_center = (row - center) ** 2 + (col - center) ** 2

        return weight * distances_to_center.reshape(1, 1, size, size).to(
            self._config.device,
        )

    @torch.enable_grad
    def optimize_kernel(
        self,
        kernel_bundle: KernelBundle,
        primal_variable: Tensor,
        fidelity: Tensor,
        centrality_penalty_matrix: Tensor,
        n_iters: int,
    ) -> Tensor:
        for _ in range(n_iters):
            kernel_bundle.optimizer.zero_grad()

            kernel = kernel_bundle.compute_kernel()

            approximation = convolution.conv(primal_variable.detach(), kernel)

            fidelity_loss = torch.nn.functional.mse_loss(
                approximation, fidelity
            )

            centrality_loss = torch.mean(
                torch.sum(kernel * centrality_penalty_matrix, dim=(-2, -1))
            )

            (fidelity_loss + centrality_loss).backward()

            kernel_bundle.optimizer.step()

        return kernel_bundle.compute_kernel()

    def tensorboard_log_image(
        self, image: Tensor, tag: str, epoch: int
    ) -> None: ...

    def tensorboard_log_scalar(self, value: float, tag: str) -> None: ...

    def tensorboard_log_hist(self, image: Tensor, tag: str) -> None: ...

    def tensorboard_log_text(self, text: str, tag: str) -> None: ...

    def tensorboard_log_loss(
        self, val: float, tag: str, epoch: int
    ) -> None: ...

    def forward_pass(
        self,
        input: Tensor,
    ) -> ForwardPassResult:
        deterministic_components = self.compute_deterministic_components(input)

        primal_dual_bundle = self.initialize_primal_dual(
            deterministic_components.fidelity
        )

        kernel_bundle = self.initialize_kernel(
            input.size(0), self.__kernel_size, self._config.step_size
        )

        # silence "possibly unbound" type-checker complaints
        kernel = cast(Tensor, None)
        n_iters = 150

        primal_variable = primal_dual_bundle.primal_variable
        dual_variable = primal_dual_bundle.dual_variable

        for greedy_iter, greedy_iter_models in enumerate(self._models, start=1):
            # Fix image, estimate kernel
            kernel = self.optimize_kernel(
                kernel_bundle,
                primal_dual_bundle.primal_variable,
                deterministic_components.fidelity,
                self.__kernel_centrality_penalty,
                n_iters,
            )

            primal_dual_bundle.schema.with_linear_argument(
                kernel.detach().clone()
            )

            # Fix kernel, estimate image
            for stage_n, model in enumerate(greedy_iter_models, start=1):
                # backward_checkpoint = checkpoint(
                #     primal_dual_bundle.schema.with_primal_proximity(
                #         model, False
                #     ).run,
                #     primal_variable,
                #     dual_variable,
                #     1,
                #     use_reentrant=False,
                # )
                #
                # primal_variable, dual_variable = cast(
                #     tuple[Tensor, Tensor], backward_checkpoint
                # )

                primal_variable, dual_variable = (
                    primal_dual_bundle.schema.with_primal_proximity(
                        model, False
                    ).run(primal_variable, dual_variable, 1)
                )

        return ForwardPassResult(
            primal_variable,
            kernel,
            deterministic_components.transmission_map,
            deterministic_components.fidelity,
        )

    @abstractmethod
    def run(self) -> None: ...


type Loss = Tensor


def train_loss(output: ForwardPassResult, ground_truth: Tensor) -> Loss:
    fidelity_term = torch.nn.functional.mse_loss(
        convolution.conv(output.primal_variable, output.kernel), output.fidelity
    )
    restored_image = output.primal_variable / output.transmission_map.clamp(
        min=1e-4
    )
    ground_truth_term = torch.nn.functional.mse_loss(
        restored_image, ground_truth
    )

    return fidelity_term + ground_truth_term


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

        self.__loss_function = train_loss
        self._tensorboard_logger = SummaryWriter(config.tensorboard_log_dir)
        self._logger.info(f"Finished initializing orchestrator!")

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
            " running. Please kill the process to stop it."
        )
        self._logger.warning(msg)
        print(msg)

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

            restored_image = (
                output.primal_variable / output.transmission_map.clamp(min=1e-4)
            )
            self.tensorboard_log_image(restored_image, "Test/Output", epoch)
            self.tensorboard_log_image(output.kernel, "Test/Kernel", epoch)

            radiance_estimation = (
                output.fidelity / output.transmission_map.clamp(min=1e-4)
            )

            if epoch == 1:
                self.tensorboard_log_image(first_input, "Test/Input", epoch)
                self.tensorboard_log_image(
                    radiance_estimation,
                    "Test/First_radiance",
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


class ExecOrchestrator(Orchestrator[ExecConfig, ExecWeightHandler]):
    def run(self) -> None:
        self._logger.info(f"Executing model on {self._config.input}")

        input_image = Image.open(self._config.input).convert("RGB")
        input_tensor = (
            to_tensor(input_image).unsqueeze(0).to(self._config.device)
        )

        with torch.no_grad():
            output = self.forward_pass(input_tensor)

        restored_image_tensor = (
            (output.primal_variable / output.transmission_map.clamp(min=1e-4))
            .squeeze(0)
            .clamp(0, 1)
        )

        restored_image = to_pil_image(restored_image_tensor.cpu())
        restored_image.save(self._config.output)
        self._logger.info(f"Saved output to {self._config.output}")


def get_orchestrator(config: U2FoldConfig) -> Orchestrator:
    if isinstance(config, TrainConfig):
        return TrainOrchestrator(config, TrainWeightHandler(config.weight_dir))
    elif isinstance(config, ExecConfig):
        return ExecOrchestrator(config, ExecWeightHandler(config.weight_dir))
    else:
        raise TypeError(f"Invalid config class.")
