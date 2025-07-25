from .color_cosine_similarity import ColorCosineSimilarityLossSpec
from .consistency import ConsistencyLossSpec
from .fidelity import FidelityLossSpec
from .ground_truth import GroundTruthLossSpec
from .generic import Loss

type LossSpec = ColorCosineSimilarityLossSpec | ConsistencyLossSpec | FidelityLossSpec | GroundTruthLossSpec

__all__ = [
    "ColorCosineSimilarityLossSpec",
    "ConsistencyLossSpec",
    "FidelityLossSpec",
    "GroundTruthLossSpec",
    "LossSpec",
    "Loss"
]
