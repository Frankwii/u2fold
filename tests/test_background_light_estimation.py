import torch
from torch import Tensor

from u2fold.math.background_light_estimation import (
    _assign_scores,
    _linear_search_background_light,
    _split_image_batch_into_quadrants,
    estimate_background_light,
)


def test_assign_scores():
    patch = Tensor([1, 2, 3, 4]).reshape(1, 1, 2, 2)
    expected_score = Tensor([2.5]) - Tensor([1, 2, 3, 4]).std()

    score = _assign_scores(patch)

    assert score.shape == (1,)
    assert torch.allclose(score, expected_score)


def test_linear_search_background_light_per_image():
    B, C, H, W = 2, 3, 2, 2
    patches = torch.rand(B, C, H, W) / 2

    patches[0, :, 0, 0] = 1.0
    patches[1, :, 1, 1] = 1.0

    expected = Tensor([[1, 1, 1], [1, 1, 1]]).reshape(2, 3, 1, 1)
    output = _linear_search_background_light(patches)

    assert output.shape == (B, C, 1, 1)
    assert torch.allclose(output, expected)


def test_split_image_into_quadrants_even_dims():
    B, C, H, W = 1, 1, 4, 4
    image = torch.arange(H * W, dtype=torch.float32).reshape(B, C, H, W)

    upper_left_quadrant = Tensor([[0, 1], [4, 5]])
    upper_right_quadrant = Tensor([[2, 3], [6, 7]])
    lower_left_quadrant = Tensor([[8, 9], [12, 13]])
    lower_right_quadrant = Tensor([[10, 11], [14, 15]])

    expected_quadrants = torch.stack(
        [
            upper_left_quadrant,
            upper_right_quadrant,
            lower_left_quadrant,
            lower_right_quadrant,
        ]
    ).reshape(4, 1, 1, 2, 2)

    output_quadrants = _split_image_batch_into_quadrants(image, C, H, W)

    assert output_quadrants.shape == (4, B, C, 2, 2)
    assert torch.equal(output_quadrants, expected_quadrants)


def test_split_image_into_quadrants_odd_dims():
    B, C, H, W = 1, 1, 5, 5
    image = torch.arange(H * W, dtype=torch.float32).reshape(B, C, H, W)

    upper_left_quadrant = Tensor([[0, 1], [5, 6]])
    upper_right_quadrant = Tensor([[2, 3], [7, 8]])
    lower_right_quadrant = Tensor([[12, 13], [17, 18]])
    lower_left_quadrant = Tensor([[10, 11], [15, 16]])

    expected_quadrants = torch.stack(
        [
            upper_left_quadrant,
            upper_right_quadrant,
            lower_left_quadrant,
            lower_right_quadrant,
        ]
    ).reshape(4, 1, 1, 2, 2)

    quadrants = _split_image_batch_into_quadrants(image, C, H, W)

    assert quadrants.shape == (4, B, C, H // 2, W // 2)
    assert torch.equal(quadrants, expected_quadrants)


def test_estimate_background_light1():
    B, C, H, W = 1, 3, 8, 8
    images = torch.rand(B, C, H, W) / 2

    # Optimal quadrant
    images[:, :, 0 : 4, 4 : W] = 0.95

    expected = Tensor([0.95, 0.95, 0.95]).reshape(B, C, 1, 1)

    output = estimate_background_light(images)

    assert output.shape == (B, C, 1, 1)
    assert torch.allclose(output, expected)


def test_estimate_background_light2():
    B, C, H, W = 1, 3, 4, 4
    images = torch.zeros(B, C, H, W)
    images[0, :, :2, :2] = 0.95 # "Optimal quadrant"
    images[:, :, 3, 3] = 1  # Optimal pixel, outside of the quadrant

    # The program should not keep splitting into quadrants
    # but rather go straight into the optimal pixel
    expected = Tensor([1, 1, 1]).reshape(B, C, 1, 1)
    output = estimate_background_light(images)

    assert output.shape == (B, C, 1, 1)
    assert torch.allclose(output, expected)
