from typing import Callable, Hashable, Iterable


def groupby[A, B: Hashable](iter: Iterable[A], key: Callable[[A], B]) -> dict[B, list[A]]:
    d = {}
    for item in iter:
        k = key(item)

        d[k] = d.get(k, []).append(item)

    return d
