import torch

from u2fold.utils.track import get_tag_group, tag


def test_cli_common_command_tracking():

    args = get_tag_group("cli_argument/common")

    expected_args = {"log_level", "log_dir", "weight_dir"}

    assert expected_args == set(args.keys())

def test_cli_command_tracking():

    args = get_tag_group("cli_argument")

    expected_args = {"common", "exec", "train"}

    assert expected_args == set(args.keys())


def test_mock_model_tracking():
    @tag("model/mock")
    class MockModel(torch.nn.Module):
        """My documentation"""
        ...

    models = get_tag_group("model")

    assert "mock" in models.keys()

    from u2fold.utils.track import _TRACKED
    _TRACKED["model"].pop("mock")

def test_models_tracking():

    models = get_tag_group("model")

    assert {"unet"}.issubset(set(models.keys()))

def test_wrapping():
    @tag("model/mock")
    class MockModel(torch.nn.Module):
        """My documentation"""
        ...

    assert MockModel.__doc__ == "My documentation"
    from u2fold.utils.track import _TRACKED
    _TRACKED["model"].pop("mock")


