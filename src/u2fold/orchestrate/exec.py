from typing import cast
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from u2fold.model.exec_spec.spec import ExecSpec
from u2fold.neural_networks.weight_handling.exec import ExecWeightHandler
from u2fold.utils.get_device import get_device
from u2fold.utils.path import compute_output_paths

from .generic import Orchestrator


class ExecOrchestrator(Orchestrator[ExecWeightHandler]):
    def run(self) -> None:
        exec_spec = cast(ExecSpec, self._spec.mode_spec)

        self._logger.info(f"Executing model on {exec_spec.input}")

        input_tensor = torch.stack([
            to_tensor(Image.open(img).convert("RGB")).to(get_device())
            for img in exec_spec.input
        ])

        with torch.no_grad():
            output = self.forward_pass(input_tensor)

        restored_image_tensor = (
            output.radiance
            .clamp(0, 1)
            .cpu()
        )

        output_paths = compute_output_paths(exec_spec.output_dir, *exec_spec.input)
        for output_path, restored_image in zip(output_paths, restored_image_tensor):
            output_path.parent.mkdir(exist_ok=True, parents=True)
            to_pil_image(restored_image).save(output_path)
            self._logger.info(f"Saved output to {output_path}")
