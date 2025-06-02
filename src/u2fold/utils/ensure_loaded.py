from importlib import import_module


def ensure_loaded(module: str):
    """
    Makes sure that the given module is loaded.
    """
    import_module(module)
