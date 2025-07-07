import shutil
import subprocess
from abc import ABC, abstractmethod
from functools import partial
from logging import getLogger
from typing import Callable, NamedTuple, Optional, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import MSELoss, Parameter
from torch.optim import Adam, Optimizer
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import u2fold.math.linear_operators as lin
import u2fold.math.proximities as prox
from u2fold.config_parsing.config_dataclasses import (
    ExecConfig,
    TrainConfig,
    U2FoldConfig,
)
from u2fold.data import get_dataloaders
from u2fold.data.dataloader_generics.base import U2FoldDataLoader
from u2fold.math import convolution, proximities
from u2fold.math.background_light_estimation import estimate_background_light
from u2fold.math.compute_first_term import (
    compute_I2,
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
        self.__weight_handler = weigth_handler
        image_bundle = kernel_bundle = ModelInitBundle(
            config.model_config,
            get_from_tag(f"model/{config.model_name}"),
            device=config.device,
        )

        self._models = self.__weight_handler.load_models(
            image_bundle, kernel_bundle
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

            fidelity, transmission_map = estimate_fidelity_and_transmission_map(
                input,
                background_light,
                self._config.transmission_map_estimation_config,
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

    @torch.compile
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

    def tensorboard_log_image(self, image: Tensor, tag: str) -> None: ...

    def tensorboard_log_scalar(self, value: float, tag: str) -> None: ...

    def tensorboard_log_hist(self, image: Tensor, tag: str) -> None: ...

    def tensorboard_log_text(self, text: str, tag: str) -> None: ...

    def tensorboard_log_loss(self, val: float) -> None: ...

    def forward_pass(
        self,
        input: Tensor,
    ) -> ForwardPassResult:
        self.tensorboard_log_image(input, "Input")

        deterministic_components = self.compute_deterministic_components(input)

        primal_dual_bundle = self.initialize_primal_dual(
            deterministic_components.fidelity
        )

        kernel_bundle = self.initialize_kernel(
            input.size(0), self.__kernel_size, self._config.step_size
        )

        # silence "possibly unbound" type-checker complaints
        kernel = cast(Tensor, None)
        N_ITERS = 150

        for greedy_iter, greedy_iter_models in tqdm(
            enumerate(self._models, start=1), desc="Greedy iterations"
        ):
            # Fix image, estimate kernel
            kernel = self.optimize_kernel(
                kernel_bundle,
                primal_dual_bundle.primal_variable,
                deterministic_components.fidelity,
                self.__kernel_centrality_penalty,
                N_ITERS,
            )

            primal_dual_bundle.schema.with_linear_argument(
                kernel.detach().clone()
            )

            # Fix kernel, estimate image
            model = next(iter(greedy_iter_models.image))
            for stage_n, _ in enumerate(greedy_iter_models.image, start=1):
                primal_variable, dual_variable = (
                    primal_dual_bundle.schema.with_primal_proximity(
                        model, False
                    ).run(
                        primal_dual_bundle.primal_variable,
                        primal_dual_bundle.dual_variable,
                        N_ITERS,
                    )
                )

                backward_checkpoint = checkpoint(
                    primal_dual_bundle.schema.run,
                    primal_variable,
                    dual_variable,
                    N_ITERS,
                    use_reentrant=False,
                )

                primal_variable, dual_variable = cast(
                    tuple[Tensor, Tensor], backward_checkpoint
                )

            self.tensorboard_log_image(
                kernel, f"Kernel; greedy iteration {greedy_iter}"
            )
            self.tensorboard_log_image(
                primal_dual_bundle.primal_variable
                / deterministic_components.transmission_map.clamp(0.1),
                f"Radiance estimation; greedy iteration {greedy_iter}",
            )

        return ForwardPassResult(
            primal_dual_bundle.primal_variable,
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
    ground_truth_term = torch.nn.functional.mse_loss(
        output.primal_variable, ground_truth
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
        self._tensorboard_logger = SummaryWriter(
            config.tensorboard_log_dir,
            flush_secs=2,  # TODO: remove this; this is for debugging only
        )
        self._logger.info(f"Finished initializing orchestrator!")

    def forward_pass(self, input: Tensor) -> ForwardPassResult:
        res = super().forward_pass(input)

        return res

    def run(self):
        # self.debug()
        print(f"Running!")
        if self._config.tensorboard_log_dir.exists():
            shutil.rmtree(self._config.tensorboard_log_dir)
            self._config.tensorboard_log_dir.mkdir()

        tensorboard_process = subprocess.Popen(
            ["tensorboard", "--logdir", self._config.tensorboard_log_dir]
        )

        min_valiation_loss = torch.inf
        for epoch in tqdm(
            range(1, self._config.n_epochs + 1), desc="Training epochs"
        ):
            print(f"Epoch {epoch}")
            train_loss = self.run_train_epoch()
            validation_loss = self.run_validation_epoch()

            if validation_loss < min_valiation_loss:
                min_valiation_loss = validation_loss
                print(f"Min loss @ epoch {epoch}")
            test_loss = self.run_validation_epoch()

            self.tensorboard_log_loss(train_loss, "Train loss", epoch)
            self.tensorboard_log_loss(validation_loss, "Validation loss", epoch)
            self.tensorboard_log_loss(test_loss, "Validation loss", epoch)

        print(
            "Traning has finished, but the tensorboard process will be kept"
            " running. Please kill the process to stop it."
        )
        tensorboard_process.wait()

    def run_test_epoch(self) -> float:
        cumulative_loss = 0.0

        with torch.no_grad():
            test_iter = iter(self.__dataloaders.test)

            first_input, first_ground_truth = next(test_iter)
            output = self.forward_pass(first_input)

            loss = self.__loss_function(output, first_ground_truth)
            cumulative_loss += loss.item()

            # self.tensorboard_log_image(first_input, "Input")
            for input, ground_truth in test_iter:
                output = self.forward_pass(input)

                loss = self.__loss_function(output, ground_truth)
                cumulative_loss += loss.item()

        return cumulative_loss / len(self.__dataloaders.validation)

    def run_validation_epoch(self) -> float:
        cumulative_loss = 0.0

        with torch.no_grad():
            for input, ground_truth in self.__dataloaders.validation:
                output = self.forward_pass(input)

                loss = self.__loss_function(output, ground_truth)
                cumulative_loss += loss.item()

        return cumulative_loss / len(self.__dataloaders.validation)

    def run_train_epoch(self) -> float:
        cumulative_loss = 0.0

        for input, ground_truth in self.__dataloaders.training:
            output = self.forward_pass(input)

            loss = self.__loss_function(output, ground_truth)

            cumulative_loss += loss.item()

            loss.backward()

        return cumulative_loss / len(self.__dataloaders.training)

    def tensorboard_log_image(self, image: Tensor, tag: str) -> None:
        self._tensorboard_logger.add_images(
            tag,
            image,
        )

    def tensorboard_log_hist(self, image: Tensor, tag: str) -> None:
        for i in range(image.size(0)):
            for c in range(image.size(1)):
                self._tensorboard_logger.add_histogram(
                    f"{tag}_channel{c}_image{i}",
                    image[i][c].flatten(),
                    global_step=i,
                )

    def tensorboard_log_loss(self, val: float, tag: str, epoch: int) -> None:
        return self._tensorboard_logger.add_scalar(tag, val, epoch)

    def tensorboard_log_text(self, text: str, tag: str) -> None:
        self._tensorboard_logger.add_text(tag, text)


class ExecOrchestrator(Orchestrator[ExecConfig, ExecWeightHandler]):
    def run(self) -> None:
        raise NotImplementedError("Hehehe")


def get_orchestrator(config: U2FoldConfig) -> Orchestrator:
    if isinstance(config, TrainConfig):
        return TrainOrchestrator(config, TrainWeightHandler(config.weight_dir))
    elif isinstance(config, ExecConfig):
        return ExecOrchestrator(config, ExecWeightHandler(config.weight_dir))
    else:
        raise TypeError(f"Invalid config class.")
