from .gelu import GeLU
from .generic import BaseActivationSpec
from .relu import ReLU

type Activation = ReLU | GeLU

__all__ = ["Activation", "BaseActivationSpec", "GeLU", "ReLU"]
