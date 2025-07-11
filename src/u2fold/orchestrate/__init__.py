from u2fold.config_parsing.config_dataclasses import (
    ExecConfig,
    TrainConfig,
    U2FoldConfig,
)
from u2fold.models.weight_handling.exec import ExecWeightHandler
from u2fold.models.weight_handling.train import TrainWeightHandler

from .exec import ExecOrchestrator
from .generic import Orchestrator
from .train import TrainOrchestrator


def get_orchestrator(config: U2FoldConfig) -> Orchestrator:
    if isinstance(config, TrainConfig):
        return TrainOrchestrator(config, TrainWeightHandler(config.weight_dir))
    elif isinstance(config, ExecConfig):
        return ExecOrchestrator(config, ExecWeightHandler(config.weight_dir))
    else:
        raise TypeError(f"Invalid config class.")


__all__ = ["get_orchestrator", "Orchestrator"]
