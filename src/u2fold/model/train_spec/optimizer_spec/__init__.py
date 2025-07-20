from .adam import AdamSpec
from .sgd import SGDSpec

type OptimizerSpec = AdamSpec | SGDSpec

__all__ = ["AdamSpec", "SGDSpec"]
