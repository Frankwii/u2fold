import os
from pathlib import Path
import shutil


class TmpFiles:
    TEST_DIR = Path("/tmp/u2fold/test/")
    first_instantiated = False

    def __init__(self, *files: Path) -> None:
        if not self.first_instantiated:
            self.TEST_DIR.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(self.TEST_DIR)
            self.first_instantiated = True

        self.files = [self.TEST_DIR / file for file in files]

    def __enter__(self) -> list[Path]:
        for file in self.files:
            file.parent.mkdir(exist_ok=True, parents=True)

        return self.files

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for file in self.files:
            if file.exists():
                os.remove(file)

    def __iter__(self):
        yield from self.files
