import torch
from torch import Tensor

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LINEAR_RGB_TO_CIEXYZ_MATRIX = torch.Tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
).to(_DEVICE)
CIEXYZ_TO_LINEAR_RGB_MATRIX = LINEAR_RGB_TO_CIEXYZ_MATRIX.inverse()

# X_n, Y_n, Z_n
CIEXYZ_AND_CIELAB_COEFFICIENTS = torch.Tensor(
    [95.0489, 100, 108.8840], device=_DEVICE
).reshape(1, 3, 1, 1)

CIEXYZ_AND_CIELAB_DELTA = 6 / 29

CIEXYZ_TO_CIELAB_MATRIX = torch.Tensor(
    [
        [0.0, 116.0, 0.0],
        [500.0, -500.0, 0.0],
        [0.0, 200.0, -200.0],
    ],
).to(_DEVICE)

CIELAB_TO_CIEXYZ_MATRIX = CIEXYZ_TO_CIELAB_MATRIX.inverse()
CIEXYZ_TO_CIELAB_BIAS = torch.Tensor([-16, 0, 0]).reshape(
    1, 3, 1, 1
).to(_DEVICE)
CIELAB_TO_CIEXYZ_BIAS = -CIEXYZ_TO_CIELAB_BIAS


def signed_pow(t: Tensor, pow: float) -> Tensor:
    return t.sign() * t.abs().pow(pow)


def XYZ_AND_LAB_f(t: Tensor) -> Tensor:
    return torch.where(
        t > CIEXYZ_AND_CIELAB_DELTA**3,
        signed_pow(t, 1 / 3),
        1 / 3 * t * CIEXYZ_AND_CIELAB_DELTA ** (-2) + 4 / 29,
    )


def XYZ_AND_LAB_f_inv(t: Tensor) -> Tensor:
    return torch.where(
        t > CIEXYZ_AND_CIELAB_DELTA,
        signed_pow(t, 3),
        3 * CIEXYZ_AND_CIELAB_DELTA**2 * (t - 4 / 29),
    )


def channelwise_matrix_multiplication(
    image: Tensor,  # (B, C, H, W),
    matrix: Tensor,  # (C, C)
) -> Tensor:  # matrix @ image, over dim=1; (B, C, H, W)
    return torch.einsum("bchw, cd -> bdhw", image, matrix)


def xyz_to_lab(image: Tensor) -> Tensor:
    return (
        channelwise_matrix_multiplication(
            XYZ_AND_LAB_f(image / CIEXYZ_AND_CIELAB_COEFFICIENTS),
            CIEXYZ_TO_CIELAB_MATRIX,
        )
        + CIEXYZ_TO_CIELAB_BIAS
    )


def lab_to_xyz(image: Tensor) -> Tensor:
    return CIEXYZ_AND_CIELAB_COEFFICIENTS * XYZ_AND_LAB_f_inv(
        channelwise_matrix_multiplication(
            image + CIELAB_TO_CIEXYZ_BIAS, CIELAB_TO_CIEXYZ_MATRIX
        )
    )


def linear_rgb_to_xyz(image: Tensor) -> Tensor:
    return channelwise_matrix_multiplication(image, LINEAR_RGB_TO_CIEXYZ_MATRIX)


def xyz_to_linear_rgb(image: Tensor) -> Tensor:
    return channelwise_matrix_multiplication(image, CIEXYZ_TO_LINEAR_RGB_MATRIX)


def linear_rgb_to_lab(image: Tensor) -> Tensor:
    return xyz_to_lab(linear_rgb_to_xyz(image))


def lab_to_linear_rgb(image: Tensor) -> Tensor:
    return xyz_to_linear_rgb(lab_to_xyz(image))


def rgb_to_linear_rgb(image: Tensor) -> Tensor:
    return torch.where(
        image > 0.04045, torch.pow((image + 0.055) / 1.055, 2.4), image / 12.92
    )


def linear_rgb_to_rgb(image: Tensor) -> Tensor:
    return torch.where(
        image > 0.0031308,
        1.055 * torch.pow(image, 1 / 2.4) - 0.055,
        12.92 * image,
    )


def rgb_to_lab(image: Tensor) -> Tensor:
    return linear_rgb_to_lab(rgb_to_linear_rgb(image))


def lab_to_rgb(image: Tensor) -> Tensor:
    return lab_to_linear_rgb(linear_rgb_to_rgb(image))
