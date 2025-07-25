from abc import ABC, abstractmethod
from typing import Iterable, Optional

import torch

from u2fold.model.neural_network_spec import NeuralNetworkSpec

type Tree[A] = A | Iterable["Tree[A]"]


class NeuralNetwork[Spec: NeuralNetworkSpec](torch.nn.Module, ABC):
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
    def __init__(self, config: Spec, device: Optional[str]) -> None: ...

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
