from typing import Literal
from pydantic import BaseModel, Field

from .neural_network_spec import NeuralNetworkSpec
from .train_spec import TrainSpec
from .exec_spec import ExecSpec
from .algorithmic_spec import AlgorithmicSpec

class U2FoldSpec[C: NeuralNetworkSpec](BaseModel):
    mode_spec: TrainSpec | ExecSpec = Field(
        title="Mode",
        discriminator="mode"
    )

    neural_network_spec: C = Field(title="Neural network specification")
    algorithmic_spec: AlgorithmicSpec = Field(title="Algorithmic specification")
    log_level: Literal["debug", "train", "warning", "error", "critical"] = Field(
        title="Log level"
    )
