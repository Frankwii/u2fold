import shutil
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Collection

import torch


class TemporaryDir(AbstractContextManager):
    def __init__(self, chosen_name: str):
        """
        Initializes the context manager.
        Args:
            chosen_name: The name for the subdirectory inside /tmp/u2fold/.
        """
        if not bool(chosen_name) or "/" in chosen_name or "." in chosen_name:
            raise ValueError(
                "chosen_name must be a valid, simple directory name."
            )

        self.__path = Path(f"/tmp/u2fold/{chosen_name}")

    def __enter__(self) -> Path:
        self.__path.mkdir(parents=True, exist_ok=True)
        return self.__path

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        shutil.rmtree(self.__path, ignore_errors=True)


def check_numeric_tensor_equality(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    return ((t1 - t2) ** 2).sum().item() < 1e-9


def check_numeric_tensor_collection_equality(
    col1: Collection[torch.Tensor], col2: Collection[torch.Tensor]
) -> bool:
    if len(col1) != len(col2):
        return False

    for t1, t2 in zip(col1, col2):
        if not check_numeric_tensor_equality(t1, t2):
            return False

    return True
