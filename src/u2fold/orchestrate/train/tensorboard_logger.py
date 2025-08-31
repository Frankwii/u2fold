from pathlib import Path

from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter

from u2fold.model.common_namespaces import ForwardPassResult, compute_radiance
from .auxiliary_methods_and_classes import EpochMetricData, LossComputingData


class TensorboardLogger:
    def __init__(self, log_dir: Path) -> None:
        self._tensorboard_logger = SummaryWriter(log_dir)

    def log_first_epoch_components(
        self, loss_computing_data: LossComputingData, formatted_phase: str
    ) -> None:
        self.log_image(
            loss_computing_data.input, f"{formatted_phase}/Input", 1
        )

        first_radiance_estimation = compute_radiance(
            loss_computing_data.output.primal_variable_history[0],
            loss_computing_data.output.deterministic_components.transmission_map.clamp(
                0.1
            ),
        )

        self.log_image(
            first_radiance_estimation,
            f"{formatted_phase}/First_radiance",
            1,
        )
        self.log_image(
            loss_computing_data.ground_truth, f"{formatted_phase}/Ground_truth", 1
        )
        self.log_image(
            loss_computing_data.output.deterministic_components.background_light,
            f"{formatted_phase}/Background_light",
            1,
        )
        self.log_image(
            loss_computing_data.output.deterministic_components.transmission_map,
            f"{formatted_phase}/Transmission_map",
            1,
        )

    def log_histogram(self, image: Tensor, tag: str) -> None:
        img = image.detach()
        for i in range(image.size(0)):
            for c in range(image.size(1)):
                self._tensorboard_logger.add_histogram(
                    f"{tag}_channel{c}_image{i}",
                    img[i][c].flatten(),
                    global_step=i,
                )

    def log_metrics(
        self, data: EpochMetricData, tag: str, epoch: int
    ) -> None:
        for metric_name, value in (data.granular_loss | data.metrics).items():
            self._tensorboard_logger.add_scalar(f"{tag}/{metric_name}", value, epoch)
        self._tensorboard_logger.add_scalar(tag, data.overall_loss, epoch)

    def log_result(
        self, result: ForwardPassResult, formatted_phase: str, epoch: int
    ) -> None:
        clamped_tm = result.deterministic_components.transmission_map.clamp(min=0.1)
        for iteration, primal_variable in enumerate(
            result.primal_variable_history[1:], start=1
        ):
            restored_image = compute_radiance(primal_variable, clamped_tm)
            self.log_image(
                restored_image, f"{formatted_phase}/Output/{iteration}", epoch
            )

        for iteration, kernel in enumerate(result.kernel_history):
            self.log_image(
                kernel, f"{formatted_phase}/Kernel/{iteration}", epoch
            )

    def log_image(self, image: Tensor, tag: str, epoch: int) -> None:
        self._tensorboard_logger.add_images(tag, image.detach(), epoch)

    def log_text(self, text: str, tag: str) -> None:
        self._tensorboard_logger.add_text(tag, text)
