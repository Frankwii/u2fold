import torch

from u2fold.utils.track import get_tracked_group, track


def test_cli_common_command_tracking():

    args = get_tracked_group("cli_argument/common")

    expected_args = {"log_level"}

    assert expected_args == set(args.keys())

def test_cli_command_tracking():

    args = get_tracked_group("cli_argument")

    expected_args = {"common", "exec", "train"}

    print(args)
    assert expected_args == set(args.keys())

@track(tag="model/mock")
class MockModel(torch.nn.Module):
    ...

def test_mock_model_tracking():

    models = get_tracked_group("model")

    assert "mock" in models.keys()

