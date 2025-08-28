import os
from pathlib import Path

from u2fold.model.spec import U2FoldSpec

USER_HOME = os.getenv("HOME")
PROJECT_HOME = os.getenv("U2FOLD_HOME", f"{USER_HOME}/.local/share/u2fold")


def get_project_home() -> Path:
    global PROJECT_HOME

    project_home = Path(PROJECT_HOME)

    project_home.mkdir(exist_ok=True)

    return project_home


def get_tensorboard_log_directory(spec: U2FoldSpec) -> Path:
    model_spec = spec.neural_network_spec
    scheme_str = f"greedyIterations_{spec.algorithmic_spec.greedy_iterations}__stages_{spec.algorithmic_spec.stages}"
    return (
        get_project_home()
        / "log"
        / "tensorboard"
        / model_spec.name
        / model_spec.format_self()
        / scheme_str
    )


def get_weight_directory(spec: U2FoldSpec) -> Path:
    model_spec = spec.neural_network_spec
    scheme_str = f"greedyIterations_{spec.algorithmic_spec.greedy_iterations}__stages_{spec.algorithmic_spec.stages}"
    return (
        get_project_home()
        / "weight"
        / model_spec.name
        / model_spec.format_self()
        / scheme_str
    )
