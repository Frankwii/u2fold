from pathlib import Path


class EmptyDirectoryError(Exception):
    def __init__(self, dir: Path):
        errmsg = f"Directory {dir} is empty."

        super().__init__(errmsg)
