from .avg import AvgPoolSpec
from .l2 import L2PoolSpec
from .max import MaxPoolSpec

type PoolSpec = MaxPoolSpec | AvgPoolSpec | L2PoolSpec

__all__ = ["AvgPoolSpec", "MaxPoolSpec", "L2PoolSpec", "PoolSpec"]
