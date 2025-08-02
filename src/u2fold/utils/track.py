from logging import getLogger
from typing import Callable, cast

logger = getLogger("tracking_logger")


NestedDict = dict[str, "NestedDict | type"]
_TRACKED: NestedDict = {}

def _split_group_and_level(tag: str) -> tuple[list[str], str]:
    parts = tag.split("/")

    return parts[:-1], parts[-1]

def _get_last_level(levels: list[str]) -> dict[str, type]:
    global _TRACKED

    logger.debug(f"Attempting to access tracking level {levels}.")

    curlevel = _TRACKED
    for level in levels:
        curlevel = cast(NestedDict, curlevel)  # pyright: ignore[reportUnnecessaryCast]
        curlevel = curlevel.get(level, {})

    logger.debug(f"Tracked at levels {levels}: {curlevel}.")

    return cast(dict[str, type], curlevel)

def _set_last_level(
        levels: list[str],
        name: str,
        cls: type
    ) -> None:
    global _TRACKED

    logger.debug(f"Setting tracking levels {levels} to {name}.")

    curlevel = _TRACKED
    for level in levels:
        curlevel = cast(NestedDict, curlevel)  # pyright: ignore[reportUnnecessaryCast]
        if level in curlevel.keys():
            curlevel = curlevel.get(level)
        else:
            curlevel[level] = {}
            curlevel = curlevel[level]

    curlevel = cast(NestedDict, curlevel)
    curlevel[name] = cls

    logger.debug(f"Finished setting tracking levels {levels} to {name}.")

def tag[C: type](*tags: str) -> Callable[[C], C]:
    global _TRACKED

    def decorator(cls: C):
        for tag in tags:
            levels, name = _split_group_and_level(tag)
            logger.info(f"Tracking element of group {levels}: {name}.")

            _set_last_level(levels, name, cls)

        return cls

    return decorator

def get_tag_group(group_tag: str) -> dict[str, type]:

    return _get_last_level(group_tag.split("/")).copy()

def get_from_tag(tag: str) -> type:

    return cast(type, _get_last_level(tag.split("/")))  # pyright: ignore[reportInvalidCast]
