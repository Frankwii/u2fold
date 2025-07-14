from .cosine_annealing_lr import CosineAneallingLRModel
from .reduce_lr_on_plateau import ReduceLROnPlateauModel
from .step_lr import StepLRModel

type LRSchedulerModel = StepLRModel | CosineAneallingLRModel | ReduceLROnPlateauModel

__all__ = [
    "CosineAneallingLRModel",
    "StepLRModel",
    "ReduceLROnPlateauModel",
    "LRSchedulerModel",
]
