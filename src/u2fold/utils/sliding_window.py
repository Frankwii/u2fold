from itertools import tee
from typing import Iterable


def sliding_window[A](iter: Iterable[A]) -> Iterable[tuple[A, A]]:
    i1, i2 = tee(iter)
    next(i2)

    return zip(i1, i2)
