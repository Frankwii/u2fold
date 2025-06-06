from abc import ABC
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from u2fold.models.generic import ModelConfig
from u2fold.utils.track import tag


@dataclass
class U2FoldConfig(ABC):
    log_level: Literal["debug", "info", "warning", "error", "critical"]
    step_size: float
    unfolded_step_size: float
    weight_dir: Path
    execution_log_dir: Path
    model_config: ModelConfig
    device: str


@tag("config/train")
@dataclass
class TrainConfig(U2FoldConfig):
    tensorboard_log_dir: Path
    loss_strategy: str
    n_epochs: int
    dataset_dir: Path
    # TODO: Add these two as CLIArguments
    # the logic responsible for ignoring these if there are already weights
    # should be inside WeightHandler.
    n_greedy_iterations: int = 3
    n_stages: int = 10


@tag("config/exec")
@dataclass
class ExecConfig(U2FoldConfig):
    input: Path
    output: Path
