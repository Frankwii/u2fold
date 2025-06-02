from abc import ABC
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from u2fold.models.generic import ModelConfig
from u2fold.utils.track import tag


@dataclass
class SharedConfig(ABC):
    model_name: str
    log_dir: Path
    execution_log_dir: Path


@tag("config/train")
@dataclass
class TrainingConfig(SharedConfig):
    # TODO: Add (admittedly complex) validation logic for this directory.
    # See SharedOrchestrator._load_models for details.
    weight_dir: Path
    dataset_dir: Path
    n_epochs: int
    # TODO: Check how to add this automatically with post_init
    tensorboard_log_dir: Path

    def __post_init__(self):
        ...


@tag("config/exec")
@dataclass
class ExecConfig(SharedConfig):
    model_name: str
    model_config: ModelConfig
    weight_dir: Path


class U2FoldConfig(ABC):
    def __init__(self, args: Namespace) -> None:
        ...

class TrainConfig(U2FoldConfig): ...


# class ExecConfig(U2FoldConfig): ...
