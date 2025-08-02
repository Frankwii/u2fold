import torch
from torch import Tensor

from u2fold.math.background_light_estimation import estimate_background_light
from u2fold.math.compute_first_term import estimate_fidelity_and_transmission_map
from u2fold.model.common_namespaces import DeterministicComponents


def compute_deterministic_components(
    input: Tensor,
    guided_filter_patch_radius: int,
    transmission_map_patch_radius: int,
    saturation_coefficient: float,
    regularization_coefficient: float,
) -> DeterministicComponents:
    with torch.no_grad():
        background_light = estimate_background_light(input)

        fidelity, transmission_map = estimate_fidelity_and_transmission_map(
            input,
            background_light,
            guided_filter_patch_radius,
            transmission_map_patch_radius,
            saturation_coefficient,
            regularization_coefficient,
        )

    return DeterministicComponents(fidelity, transmission_map, background_light)
