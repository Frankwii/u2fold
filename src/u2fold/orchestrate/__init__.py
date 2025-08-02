from typing import Any
from u2fold.neural_networks.weight_handling.exec import ExecWeightHandler
from u2fold.neural_networks.weight_handling.train import TrainWeightHandler
from u2fold.model import U2FoldSpec, TrainSpec, ExecSpec
from u2fold.utils.get_directories import get_weight_directory

from .exec import ExecOrchestrator
from .generic import Orchestrator
from .train import TrainOrchestrator



def get_orchestrator(spec: U2FoldSpec) -> Orchestrator[Any]:
    weight_dir = get_weight_directory(spec.neural_network_spec)
    if isinstance(spec.mode_spec, TrainSpec):
        return TrainOrchestrator(spec, TrainWeightHandler(weight_dir))
    elif isinstance(spec.mode_spec, ExecSpec):
        return ExecOrchestrator(spec, ExecWeightHandler(weight_dir))
    else:
        raise TypeError(f"Invalid spec class.")


__all__ = ["get_orchestrator", "Orchestrator"]
