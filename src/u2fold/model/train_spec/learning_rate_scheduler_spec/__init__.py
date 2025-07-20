from .cosine_annealing_lr import CosineAneallingLRSpec
from .reduce_lr_on_plateau import ReduceLROnPlateauSpec
from .step_lr import StepLRModel

type LRSchedulerSpec = StepLRModel | CosineAneallingLRSpec | ReduceLROnPlateauSpec

__all__ = [
    "CosineAneallingLRSpec",
    "StepLRModel",
    "ReduceLROnPlateauSpec",
    "LRSchedulerSpec",
]
