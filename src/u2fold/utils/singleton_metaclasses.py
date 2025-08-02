from abc import ABCMeta
from typing import final, override


@final
class Singleton(type):
    _instances = {}

    @override
    def __call__(cls, *args, **kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        if cls not in cls._instances:  # pyright: ignore[reportUnknownMemberType]
            cls._instances[cls] = super(Singleton, cls).__call__(  # pyright: ignore[reportUnknownMemberType]
                *args, **kwargs
            )

        return cls._instances[cls]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


@final
class AbstractSingleton(ABCMeta):
    _instances = {}

    @override
    def __call__(cls, *args, **kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        if cls not in cls._instances:  # pyright: ignore[reportUnknownMemberType]
            cls._instances[cls] = super(AbstractSingleton, cls).__call__(  # pyright: ignore[reportUnknownMemberType]
                *args, **kwargs
            )

        return cls._instances[cls]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
