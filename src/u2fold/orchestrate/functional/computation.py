import torch
from torch import Tensor

from u2fold.config_parsing.config_dataclasses import (
    TransmissionMapEstimationConfig,
)
from u2fold.math import convolution
from u2fold.math.background_light_estimation import estimate_background_light
from u2fold.math.compute_first_term import (
    estimate_fidelity_and_transmission_map,
)
from u2fold.math.rescale_image import rescale_color
from u2fold.utils.sliding_window import sliding_window

from .types import DeterministicComponents, ForwardPassResult


@torch.compile
def compute_deterministic_components(
    input: Tensor, transmission_map_config: TransmissionMapEstimationConfig
) -> DeterministicComponents:
    with torch.no_grad():
        background_light = estimate_background_light(input)

        fidelity, transmission_map = estimate_fidelity_and_transmission_map(
            input,
            background_light,
            transmission_map_config.patch_radius,
            transmission_map_config.saturation_coefficient,
            transmission_map_config.regularization_coefficient,
        )

    return DeterministicComponents(fidelity, transmission_map, background_light)


@torch.compile
def loss(output: ForwardPassResult, ground_truth: Tensor) -> Tensor:
    final_primal_variable = output.primal_variable_history[-1]
    final_kernel = output.kernel_history[-1]
    radiance = rescale_color(
        final_primal_variable
        / output.deterministic_components.transmission_map.clamp(min=0.1)
    )
    pairwise_primal_variables = list(
        sliding_window(output.primal_variable_history)
    )

    fidelity_term = torch.nn.functional.mse_loss(
        convolution.conv(final_kernel, final_primal_variable),
        output.deterministic_components.fidelity,
    )
    ground_truth_term = torch.nn.functional.mse_loss(radiance, ground_truth)

    consistency_term = sum(
        torch.nn.functional.mse_loss(previous_output, next_output)
        for (previous_output, next_output) in pairwise_primal_variables
    ) / len(pairwise_primal_variables)

    tv_loss = (
        torch.abs(radiance[..., 1:] - radiance[..., :-1]).mean()
        + torch.abs(radiance[..., 1:, :] - radiance[..., :-1, :]).mean()
    )

    color_similarity_term = (
        1 - torch.cosine_similarity(radiance, ground_truth, dim=1).mean()
    )

    return (
        ground_truth_term
        + 0.1 * fidelity_term
        + 0.1 * consistency_term
        + 0.01 * tv_loss
        + 0.1 * color_similarity_term
    )
