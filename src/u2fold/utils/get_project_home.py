import os
from pathlib import Path

USER_HOME = os.getenv("HOME")
PROJECT_HOME = os.getenv("U2FOLD_HOME", f"{USER_HOME}/.local/share/u2fold")

def get_project_home() -> Path:
    global PROJECT_HOME

    project_home = Path(PROJECT_HOME)

    project_home.mkdir(exist_ok=True)

    return project_home
