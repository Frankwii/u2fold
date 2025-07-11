import torch

from u2fold.math.convolution import conv


def square_norm(t: torch.Tensor):
    return torch.sum(t**2)


mock_f = torch.rand((16, 3, 11, 11))


def test_conv_shape(mock_f=mock_f):
    mock_g = torch.rand((1, 3, 3, 3))

    assert conv(mock_g, mock_f).shape == mock_f.shape

def test_conv_shape2(mock_f=mock_f):
    mock_g = torch.rand((16, 3, 3, 3))

    assert conv(mock_f, mock_g).shape == mock_g.shape

def test_conv_zeros(mock_f=mock_f):
    mock_g = torch.zeros((1, 3, 3, 3))

    assert square_norm(conv(mock_g, mock_f)) < 1e-9


def test_conv_identity(mock_f=mock_f):
    kernel_size = 5
    mock_dirac_delta = torch.zeros((1, 3, kernel_size, kernel_size))

    middle_point = kernel_size // 2
    for i in range(3):
        mock_dirac_delta[0, i, middle_point, middle_point] = 1

    res = conv(mock_dirac_delta, mock_f)

    assert square_norm(res - mock_f) < 1e-9
