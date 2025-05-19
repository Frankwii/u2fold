import os
from pathlib import Path

USER_HOME = os.getenv("HOME")
PROJECT_HOME = os.getenv("U2FOLD_HOME", f"{USER_HOME}/.local/share")

def get_project_home() -> Path:
    global PROJECT_HOME

    return Path(PROJECT_HOME)
