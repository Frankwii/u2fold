import os
from pathlib import Path
from u2fold.model.neural_network_spec import NeuralNetworkSpec

USER_HOME = os.getenv("HOME")
PROJECT_HOME = os.getenv("U2FOLD_HOME", f"{USER_HOME}/.local/share/u2fold")

def get_project_home() -> Path:
    global PROJECT_HOME

    project_home = Path(PROJECT_HOME)

    project_home.mkdir(exist_ok=True)

    return project_home

def get_tensorboard_log_directory(model_spec: NeuralNetworkSpec) -> Path:
    return (
    get_project_home() 
        / "log" 
        / "tensorboard" 
        / model_spec.name
        / model_spec.format_self()
    )

def get_weight_directory(model_spec: NeuralNetworkSpec) -> Path:
    return (
        get_project_home()
        / "weight"
        / model_spec.name
        / model_spec.format_self()
    )
