from u2fold.model.neural_network_spec.unet import UNetSpec
from .aunet import AUNetSpec

type NeuralNetworkSpec = AUNetSpec | UNetSpec

__all__ = ["AUNetSpec", "UNetSpec", "NeuralNetworkSpec"]
