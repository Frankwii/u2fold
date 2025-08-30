from collections.abc import Mapping
from typing import Callable, Hashable, Protocol, TypeVar

S = TypeVar("S")
class SupportsSum(Protocol):
    def __add__(self: S, value: S, /) -> S: ...

def merge_sum[T_key: Hashable, T_val: SupportsSum](d1: Mapping[T_key, T_val], d2: Mapping[T_key, T_val]) -> dict[T_key, T_val]:

    result = dict(d1)

    for k,v in d2.items():
        if k in d1:
            result[k] = result[k] + v
        else:
            result[k] = v

    return result

def shallow_dict_map[T_key: Hashable, T_val, T_res](f: Callable[[T_val], T_res], d: Mapping[T_key, T_val]) -> dict[T_key, T_res]:

    return {k:f(v) for k,v in d.items()}
