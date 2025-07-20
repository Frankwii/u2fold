from .color_cosine_similarity import ColorCosineSimilarityLoss
from .consistency import ConsistencyLoss
from .fidelity import FidelityLoss
from .ground_truth import GroundTruthLoss

type LossSpec = ColorCosineSimilarityLoss | ConsistencyLoss | FidelityLoss | GroundTruthLoss

__all__ = [
    "ColorCosineSimilarityLoss",
    "ConsistencyLoss",
    "FidelityLoss",
    "GroundTruthLoss",
    "LossSpec"
]
