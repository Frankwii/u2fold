from typing import Callable

def chain_calls[T](*funcs: Callable[[T], None]) -> Callable[[T], None]:
    def res(arg: T, /) -> None:
        for f in funcs:
            f(arg)

    return res
