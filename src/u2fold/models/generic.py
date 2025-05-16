from dataclasses import dataclass

from abc import ABC, abstractmethod

import torch

@dataclass
class ModelConfig(ABC):
    """Configuration for a model. 

    Models should specify a (concrete) subclass of this class via generic types
    (see Model for details and examples).
    """
    ...

class Model[Config: ModelConfig](ABC, torch.nn.Module):
    r"""Common interface for models with a configuration class.

    Generics::

        Config: A ModelConfig subclass. It should provide all of the necessary
            hyperparameters for instantiating the neural network (e.g. number
            of layers and number of neurons on each layer, for a standard 
            feed-forward neural network; see example below).

    Args::

        config: Config. An instance of the specified Config, as described above.

    Subclassing example::

        @dataclass
        class FeedForwardConfig(ModelConfig):
            layer_dimensions: list[int]

            def __post_init__(self):
                assert len(self.layer_dimensions) >= 2

        class FeedForwardBlock(Model[FeedForwardConfig]):
            def __init__(self, config: FeedForwardConfig) -> None:
                torch.nn.Module.__init__(self)
                layer_dimensions = config.layer_dimensions

                dimension_pairs = (
                    (layer_dimensions[i], layer_dimensions[i+1])
                    for i in range(len(layer_dimensions) - 1)
                )

                self.__layers = torch.nn.ModuleList([
                    torch.nn.Linear(input_dim, output_dim)
                    for input_dim, output_dim in dimension_pairs
                ])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for layer in self.__layers:
                    x = layer(x)
                return x
    """
    @abstractmethod
    def __init__(self, config: Config) -> None:
        ...
