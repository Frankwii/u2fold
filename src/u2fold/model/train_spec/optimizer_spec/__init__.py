from .adam import AdamModel
from .sgd import SGDModel

type OptimizerModel = AdamModel | SGDModel

__all__ = ["AdamModel", "SGDModel"]
