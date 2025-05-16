import torch

from u2fold.utils.track import get_tag_group, tag


def test_cli_common_command_tracking():

    args = get_tag_group("cli_argument/common")

    expected_args = {"log_level", "log_dir"}

    assert expected_args == set(args.keys())

def test_cli_command_tracking():

    args = get_tag_group("cli_argument")

    expected_args = {"common", "exec", "train"}

    print(args)
    assert expected_args == set(args.keys())

@tag("model/mock")
class MockModel(torch.nn.Module):
    """My documentation"""
    ...

def test_mock_model_tracking():

    models = get_tag_group("model")

    assert "mock" in models.keys()

def test_models_tracking():

    models = get_tag_group("model")

    assert {"unet-like"}.issubset(set(models.keys()))

def test_wrapping():

    assert MockModel.__doc__ == "My documentation"
