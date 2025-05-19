from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ModelConfig(ABC):
    """Configuration for a model.

    Models should specify a (concrete) subclass of this class via generic types
    (see Model for details and examples). A metadata attribute should be added
    to each field in subclasses. It should be a dictionary containing two keys::

        desc (str): A human-readable description of what the parameter controls.

        cli_mode (Literal["train", "exec", "model"]): How to register this
            parameter in the CLI. If "train" or "exec", it should be specified
            as a parameter only for the respective subparser (e.g., since
            `dropout` has a mode of "train", it should be specified like this:
            `u2fold train --dropout 0.2`). If "model", it has to be specified
            as a parameter of the respective model subparser.
    """
    dropout: float = field(
        metadata={
            "desc":"Dropout to use during traning",
            "cli_mode": "train"
        }
    )

    def __post_init__(self):
        if not 0 <= self.dropout <= 1:
            raise ValueError("Dropout must be between 0 and 1")

class Model[Config: ModelConfig](ABC, torch.nn.Module):
    r"""Common interface for models with a configuration class.

    When subclassing, make sure to follow the guidelines specified below for
    Config and device.

    Generics::

        Config: A ModelConfig subclass. It should contain all of the necessary
            hyperparameters for instantiating the neural network (e.g. number
            of layers and number of neurons on each layer, for a standard
            feed-forward neural network; see example below) as attributes.

    Args::

        config (Config): An instance of the specified Config.

        device: The device in which the model should be loaded. This is
            necessary so that it is possible to skip weight initialization
            (useful when loading pre-trained weights). This parameter should
            be passed to all submodules. For more details, see
            https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html

    Subclassing example (taken from the test suite)::

        @dataclass
        class FeedForwardConfig(ModelConfig):
            layer_dimensions: list[int]

            def __post_init__(self):
                assert len(self.layer_dimensions) >= 2

        class FeedForwardBlock(Model[FeedForwardConfig]):
            def __init__(
                self, config: FeedForwardConfig, device: Optional[str] = None
            ) -> None:
                torch.nn.Module.__init__(self)
                layer_dimensions = config.layer_dimensions

                dimension_pairs = (
                    (layer_dimensions[i], layer_dimensions[i + 1])
                    for i in range(len(layer_dimensions) - 1)
                )

                self.__layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(input_dim, output_dim, device=device)
                        for input_dim, output_dim in dimension_pairs
                    ]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for layer in self.__layers:
                    x = layer(x)
                return x

        config = FeedForwardConfig([10, 100, 5])

        model = FeedForwardBlock(config)
    """
    @abstractmethod
    def __init__(self, config: Config, device: Optional[str]) -> None:
        ...
