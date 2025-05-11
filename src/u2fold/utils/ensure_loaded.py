def ensure_loaded(module: str):
    """
    Makes sure that the given module is loaded.
    """
    __import__(module)
