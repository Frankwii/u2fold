from functools import partial

import torch
from torch import Tensor
from torch.optim import Adam

import u2fold.math.linear_operators as linear_operators
import u2fold.math.proximities as proximities
from u2fold.math.primal_dual import PrimalDualScheme
from u2fold.model.common_namespaces import KernelBundle, PrimalDualBundle
from torch.nn import Parameter


def initialize_square_matrix_with_square_distances_to_center(
    size: int, device: str
) -> Tensor:
    center = size // 2

    row = col = torch.arange(size).unsqueeze(1)
    col = col.transpose(0, 1)

    distances_to_center = (row - center) ** 2 + (col - center) ** 2

    return distances_to_center.reshape(1, 1, size, size).to(device)


def initialize_primal_dual(
    fidelity: Tensor, unfolded_step_size: float, step_size: float
) -> PrimalDualBundle:
    primal_dual_schema = (
        PrimalDualScheme()
        .with_primal_proximity(proximities.identity, True)
        .with_primal_argument(0)
        .with_dual_proximity(proximities.conjugate_shifted_square_L2_norm, True)
        .with_dual_argument(fidelity)
        .with_step_sizes(unfolded_step_size, step_size)
        .with_linear_operator(
            linear_operators.convolution, linear_operators.conjugate_convolution
        )
    )

    return PrimalDualBundle(
        primal_dual_schema,
        fidelity.clone(),
        fidelity.clone(),
    )


def get_gaussian_kernel(
    standard_deviations: Tensor,  # (B, C, 1, 1)
    square_distances_to_center: Tensor,  # (B, C, 1, 1)
) -> Tensor:
    return torch.exp(-square_distances_to_center / (2 * standard_deviations**2))


def initialize_gaussian_kernel(
    batch_size: int,
    device: str,
    square_distances_to_center: Tensor,  # Pass as argument to avoid recomputing
    learning_rate: float,
) -> KernelBundle:
    standard_deviations = torch.ones(batch_size, 3, 1, 1)

    standard_deviations = Parameter(standard_deviations.to(device), requires_grad=True)

    return KernelBundle(
        preimage=standard_deviations,
        preimage_to_kernel_mapping=partial(
            get_gaussian_kernel,
            square_distances_to_center=square_distances_to_center,
        ),
        optimizer=Adam([standard_deviations], lr=learning_rate),
    )
