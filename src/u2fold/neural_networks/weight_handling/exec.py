from pathlib import Path

from u2fold.neural_networks.generic import NeuralNetworkSpec, NeuralNetwork

from .generic import (
    ModelInitBundle,
    WeightHandler,
)


class ExecWeightHandler(WeightHandler):
    def __init__(self, model_weight_dir: Path) -> None:
        super().__init__(model_weight_dir)

    def _handle_no_greedy_iter_dirs(self, root_dir: Path) -> None:
        errmsg = f"Empty global weight directory: {root_dir}"
        raise FileNotFoundError(errmsg)

    def _handle_empty_stage_dir(self, stage_dir: Path) -> list[Path]:
        errmsg = f"Empty model weight directory: {stage_dir}"
        raise FileNotFoundError(errmsg)

    def _handle_nonexisting_weight_file[C: NeuralNetworkSpec](
        self, weight_file: Path, model_bundle: ModelInitBundle[C]
    ) -> NeuralNetwork[C]:
        errmsg = f"Non-existing model weight file: {weight_file}"
        raise FileNotFoundError(errmsg)
