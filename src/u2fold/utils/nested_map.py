import inspect
from collections.abc import Iterable
from typing import Any, Callable, Generator, NamedTuple, cast

type MappableIterable[A] = list[A] | tuple[A] | dict[Any, A] | Generator[A]

type NestedIterable[A] = A | MappableIterable["NestedIterable[A]"]


def nested_map[A, B](
        f: Callable[[A], B], nested_structure: NestedIterable[A]
    ) -> NestedIterable[B]:
    """
    Takes a nested iterable structure and applies f to all of its nodes,
    returning them in the same structure.
    """

    structure = type(nested_structure)
    match nested_structure:
        case dict():
            return {
                k: nested_map(f, v)
                for k, v in nested_structure.items()
            }
        # NamedTuple
        case _ if hasattr(structure, "_fields") and isinstance(
                getattr(structure, "_fields"), tuple
            ):
            d = {
                k: nested_map(f, v) for
                k, v in cast(NamedTuple, nested_structure)._asdict().items()
            }
            return cast(NestedIterable[B], structure(**d))
        case _ if inspect.isgenerator(nested_structure):
            return ( nested_map(f, s) for s in nested_structure )
        case str():
            return f(nested_structure)
        case Iterable():
            structure = cast(type[list], structure)
            return structure(map(lambda s: nested_map(f, s), nested_structure))
        case _:
            return f(nested_structure)
