import os
from pathlib import Path

from u2fold.model.algorithmic_spec.spec import AlgorithmicSpec
from u2fold.model.neural_network_spec import NeuralNetworkSpec
from u2fold.model.spec import U2FoldSpec

USER_HOME = os.getenv("HOME")
PROJECT_HOME = os.getenv("U2FOLD_HOME", f"{USER_HOME}/.local/share/u2fold")


def get_project_home() -> Path:
    global PROJECT_HOME

    project_home = Path(PROJECT_HOME)

    project_home.mkdir(exist_ok=True)

    return project_home

def get_algorithmic_spec_subdir(algorithmic_spec: AlgorithmicSpec) -> str:
    if algorithmic_spec.share_network_weights:
        return "shared"
    else:
        return f"greedyIterations_{algorithmic_spec.greedy_iterations}__stages_{algorithmic_spec.stages}"


def get_tensorboard_log_directory[S: NeuralNetworkSpec](spec: U2FoldSpec[S]) -> Path:
    model_spec = spec.neural_network_spec
    return (
        get_project_home()
        / "log"
        / "tensorboard"
        / model_spec.name
        / model_spec.format_self()
        / get_algorithmic_spec_subdir(spec.algorithmic_spec)
    )


def get_weight_directory[S: NeuralNetworkSpec](spec: U2FoldSpec[S]) -> Path:
    model_spec = spec.neural_network_spec
    return (
        get_project_home()
        / "weight"
        / model_spec.name
        / model_spec.format_self()
        / get_algorithmic_spec_subdir(spec.algorithmic_spec)
    )
