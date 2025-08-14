from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch

from u2fold.model.neural_network_spec import NeuralNetworkSpec

type Tree[A] = A | Iterable["Tree[A]"]


class NeuralNetwork[Spec: NeuralNetworkSpec](torch.nn.Module, ABC):
    r"""Common interface for models with a configuration class.

    When subclassing, make sure to follow the guidelines specified below for
    Config and device.

    Generics::

        Spec: A NeuralNetworkSpec subclass. It should contain all of the necessary
            hyperparameters for instantiating the neural network (e.g. number
            of layers and number of neurons on each layer, for a standard
            feed-forward neural network; see example below) as attributes.

    Args::

        spec (Spec): An instance of the specified Spec.

        device: The device in which the model should be loaded. This is
            necessary so that it is possible to skip weight initialization
            (useful when loading pre-trained weights). This parameter should
            be passed to all submodules. For more details, see
            https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html

    Subclassing example::

        @dataclass
        class FeedForwardSpec(ModelSpec):
            layer_dimensions: list[int]

            def __post_init__(self):
                assert len(self.layer_dimensions) >= 2


        class FeedForwardBlock(Model[FeedForwardSpec]):
            def __init__(
                self, spec: FeedForwardSpec, device: Optional[str] = None
            ) -> None:
                torch.nn.Module.__init__(self)
                layer_dimensions = spec.layer_dimensions

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


        spec = FeedForwardSpec([10, 100, 5])

        model = FeedForwardBlock(spec)
    """

    @abstractmethod
    def __init__(self, spec: Spec, device: str | None) -> None: ...

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
