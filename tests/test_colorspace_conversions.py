"""Colorspace conversion tests.

Based on the very useful first answer of https://stackoverflow.com/questions/59169261/lossless-srgb-lab-conversion
"""

import torch

from u2fold.neural_networks.metrics_and_losses.colorspace_utilities import (
    linear_rgb_to_xyz,
    rgb_to_lab,
    rgb_to_linear_rgb,
    xyz_to_lab,
)


def assert_colorspace_conversions(
    rgb: torch.Tensor,
    linear_rgb: torch.Tensor,
    xyz: torch.Tensor,
    lab: torch.Tensor,
):
    linear_rgb_output = rgb_to_linear_rgb(rgb)
    assert linear_rgb_output.allclose(linear_rgb, atol=0.01)
    xyz_output = linear_rgb_to_xyz(linear_rgb_output)
    assert xyz_output.allclose(xyz, atol=0.01)
    lab_output = xyz_to_lab(xyz_output)
    assert lab_output.allclose(lab, atol=0.01)

    lab_direct_output = rgb_to_lab(rgb)
    assert lab_direct_output.allclose(lab, atol=0.01)


def test_rgb_to_lab_black():
    black = torch.zeros(1, 3, 1, 1)
    assert_colorspace_conversions(black, black, black, black)


def test_rgb_to_lab_white():
    rgb_white = torch.ones(1, 3, 1, 1)
    xyz_white = torch.tensor((0.9505, 1.0, 1.0888)).reshape_as(rgb_white)
    lab_white = torch.tensor((100.0, 0.0, 0.0)).reshape_as(rgb_white)

    assert_colorspace_conversions(rgb_white, rgb_white.clone(), xyz_white, lab_white)


def test_rgb_to_lab_gray():
    gray = torch.full((1, 3, 1, 1), 119 / 255)

    linear_rgb_gray = torch.full_like(gray, 0.1845)
    xyz_gray = torch.tensor((0.1753, 0.1845, 0.2009)).reshape_as(gray)
    lab_gray = torch.tensor((50.03, 0.0, 0.0)).reshape_as(gray)

    assert_colorspace_conversions(gray, linear_rgb_gray, xyz_gray, lab_gray)
