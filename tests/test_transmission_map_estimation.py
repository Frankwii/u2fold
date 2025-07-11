import torch

from u2fold.math.background_light_estimation import estimate_background_light
from u2fold.math.transmission_map_estimation import (
    compute_saturation_map,
    estimate_coarse_red_transmission_map,
)


def test_compute_saturation_map():
    B = 2
    C = 3
    H = W = 5

    eps = 1e-3
    input = torch.full((B, C, H, W), eps)

    input[0, 0, 0, 0] = 1  # top-left pixel is red

    expected = torch.zeros(B, 1, H, W)
    expected[0, 0, 0, 0] = 1 - eps

    output = compute_saturation_map(input)

    if not torch.allclose(output, expected):
        raise ValueError(f"Unequal tensors!\n{output == expected}")


def test_estimate_coarse_transmission_map():
    B = 1
    C = 3
    H = W = 4

    input = torch.zeros(B, C, H, W)
    input[0, 1:, :2, :2] = 1  # top-left corner is cyan

    expected_background_lights = torch.zeros(B, 3, 1, 1)
    expected_background_lights[0, 1:, 0, 0] = 1

    background_lights = estimate_background_light(input)

    assert background_lights.shape == (B, C, 1, 1)
    assert torch.equal(background_lights, expected_background_lights)

    patch_radius = 1

    expected_coarse_tm = torch.ones(B, 1, H, W)
    expected_coarse_tm[0, 0, 0, 0] = 0

    output_coarse_tm = estimate_coarse_red_transmission_map(
        input, background_lights, patch_radius, 1
    )

    if not torch.allclose(output_coarse_tm, expected_coarse_tm):
        raise ValueError(
            f"Unequal tensors!\n{output_coarse_tm}\n{expected_coarse_tm}"
        )
