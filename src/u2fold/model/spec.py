from types import NoneType
from typing import Literal
from pydantic import BaseModel, Field

from .neural_network_spec import NeuralNetworkSpec
from .train_spec.spec import TrainSpec
from .algorithmic_spec import AlgorithmicSpec

type ExecSpec = NoneType # TODO: Implment

class U2FoldSpec(BaseModel):
    mode_spec: TrainSpec | ExecSpec = Field(discriminator="mode")

    neural_network_spec: NeuralNetworkSpec
    algorithmic_spec: AlgorithmicSpec
    log_level: Literal["debug", "train", "warning", "error", "critical"]
