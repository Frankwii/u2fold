from typing import cast
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from u2fold.model.exec_spec.spec import ExecSpec
from u2fold.neural_networks.weight_handling.exec import ExecWeightHandler
from u2fold.utils.get_device import get_device
from u2fold.utils.path import compute_output_paths

from .generic import Orchestrator

def compute_closest_multiple(n: int, m: int) -> int:
    """Computes a minimally close number to n that is a multiple of m.

    Returns the smallest possible number that satisfies the property,
    meaning that if there is a tie (e.g. with n=6, m=4), it "rounds below"
    (outputs 4 in the example).
    """

    r = n % m
    if r <= (m // 2):
        d = -r
    else:
        d = m - r
    return n + d

def resize_image_to_closest_multiple(img: Image.Image, m: int) -> Image.Image:
    width = img.width
    height = img.height
    new_width = compute_closest_multiple(width, m)
    new_height = compute_closest_multiple(height, m)

    return img.resize((new_width, new_height))


class ExecOrchestrator(Orchestrator[ExecWeightHandler]):
    def run(self) -> None:
        exec_spec = cast(ExecSpec, self._spec.mode_spec)

        self._logger.info(f"Executing model on {exec_spec.input}")

        m = 2 ** len(self._spec.neural_network_spec.channels_per_layer)

        input_tensors = [
            to_tensor(resize_image_to_closest_multiple(Image.open(img).convert("RGB"), m)).to(get_device()).unsqueeze(0)
            for img in exec_spec.input
        ]

        with torch.no_grad():
            output_tensors = [self.forward_pass(t) for t in input_tensors]

        restored_image_tensors = [
            output.radiance
            .clamp(0, 1)
            .cpu()

            for output in output_tensors
        ]

        output_paths = compute_output_paths(exec_spec.output_dir, *exec_spec.input)
        for output_path, restored_image in zip(output_paths, restored_image_tensors):
            output_path.parent.mkdir(exist_ok=True, parents=True)
            to_pil_image(restored_image.squeeze(0)).save(output_path)
            self._logger.info(f"Saved output to {output_path}")
