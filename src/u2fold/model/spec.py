from typing import Literal
from pydantic import BaseModel, Field

from .neural_network_spec import NeuralNetworkSpec
from .train_spec import TrainSpec
from .exec_spec import ExecSpec
from .algorithmic_spec import AlgorithmicSpec

class U2FoldSpec[C: NeuralNetworkSpec](BaseModel):
    """Specification for the whole program."""
    mode_spec: TrainSpec | ExecSpec = Field(
        title="Mode",
        description="Specify mode in which to execute the program (either 'train' or 'exec', for training a network or executing it with given images) and all related configuration (dataset or input images, loss function components...)",
        discriminator="mode"
    )
    neural_network_spec: C = Field(
        title="Neural network specification",
        description="Specify the neural network to train and all its architectural hyperparameters"
    )
    algorithmic_spec: AlgorithmicSpec = Field(
        title="Algorithmic specification",
        description="Specify the unfolding layout and all the parameters for the ``classical'' components."
    )
    log_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        title="Log level"
    )
