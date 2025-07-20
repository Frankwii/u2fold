from .adam import AdamModel
from .sgd import SGDModel

type OptimizerSpec = AdamModel | SGDModel

__all__ = ["AdamModel", "SGDModel"]
