import inspect
from collections.abc import Iterable


def nested_map(f, nested_structure):
    """
    Takes a nested iterable structure and applies f to all of its nodes,
    returning them in the same structure.
    """

    structure = type(nested_structure)

    if issubclass(structure, dict):
        return {
            k: nested_map(f, v)
            for k, v in nested_structure.items()
        }
    elif inspect.isgenerator(nested_structure):
        return ( nested_map(f, s) for s in nested_structure )
    elif issubclass(structure, Iterable) and not issubclass(structure, str):
        return structure(map(lambda s: nested_map(f, s), nested_structure))
    else:
        return f(nested_structure)
