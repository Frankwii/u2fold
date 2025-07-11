import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from u2fold.config_parsing.config_dataclasses import ExecConfig
from u2fold.models.weight_handling.exec import ExecWeightHandler

from .generic import Orchestrator


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
            (
                output.primal_variable_history[-1]
                / output.deterministic_components.transmission_map.clamp(
                    min=0.1
                )
            )
            .squeeze(0)
            .clamp(0, 1)
        )

        restored_image = to_pil_image(restored_image_tensor.cpu())
        restored_image.save(self._config.output)
        self._logger.info(f"Saved output to {self._config.output}")
