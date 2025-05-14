import torch

from u2fold.math.convolution import conv


def square_norm(t: torch.Tensor):

    return torch.sum(t ** 2)

mock_f = torch.rand((16, 3, 10, 10))

def test_conv_shape(mock_f = mock_f):

    mock_g = torch.rand((3, 3, 3))

    assert conv(mock_f, mock_g).shape == torch.Size([16, 3, 10, 10])

def test_conv_zeros(mock_f = mock_f):

    mock_g = torch.zeros((3, 3, 3))

    assert torch.sum(conv(mock_f, mock_g) ** 2) < 10e-9

def test_conv_identity(mock_f = mock_f):

    kernel_size = 5
    mock_dirac_delta = torch.zeros((3, kernel_size, kernel_size))

    middle_point = kernel_size // 2
    for i in range(3):
        mock_dirac_delta[i, middle_point, middle_point] = 1

    res = conv(mock_f, mock_dirac_delta)

    assert square_norm(res - mock_f) < 10e-9
