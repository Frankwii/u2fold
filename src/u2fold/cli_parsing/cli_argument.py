from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional, get_args, get_origin

from u2fold.utils.get_project_home import get_project_home
from u2fold.utils.singleton_metaclasses import AbstractSingleton


class CLIArgument[T: object](ABC, metaclass = AbstractSingleton):
    
    @classmethod
    def value_type(cls: type) -> type[T]:
        for superclass in cls.__mro__:
            # This basically gets the text used to define the class
            # as it was in the source code.
            # See PEP 560 "__mro_entries__" for details.
            for superclass_base in superclass.__orig_bases__:
                if get_origin(superclass_base) is CLIArgument:
                    return get_args(superclass_base)[0]

        errmsg = (f"Could not find annotated CLIArgument in the class"
                  f" hierarchy of {cls.__name__}. Make sure that subclassing"
                  f" of CLIArgument is done with a substituted TypeVar.")
        raise TypeError(errmsg)

    def validate_value(self, value: T) -> None:
        if not isinstance(value, t := self.value_type()):
            raise TypeError(
                f"Invalid type when parsing CLIArgument \
                `{type(self)}`. Expected `{t}`, got `{type(value)}`."
            )

        self._validate_value(value)

    def short_name(self) -> Optional[str]:
        return None

    def metavar(self) -> Optional[str]:
        return None

    def choices(self) -> Optional[list[T]]:
        return None

    def required(self) -> bool:
        return True

    def default(self) -> Optional[T]:
        return None

    def _help(self) -> str:
        return dedent(self.help()).strip()

    def add_to_parser(self, parser: ArgumentParser) -> None:
        possible_args = (
            self.short_name(),
            self.long_name()
        )

        possible_kwargs = {
            "type": self.value_type(),
            "metavar": self.metavar(),
            "help": self._help(),
            "choices": self.choices(),
            "required": self.required(),
            "default": self.default()
        }

        args = (arg for arg in possible_args if arg is not None)
        kwargs = {k:v for k, v in possible_kwargs.items() if v is not None}

        parser.add_argument(
            *args,
            **kwargs
        )

    @abstractmethod
    def _validate_value(self, value: T) -> None: ...

    @abstractmethod
    def long_name(self) -> str: ...

    @abstractmethod
    def help(self) -> str: ...


class PathCLIArgument(CLIArgument[Path], ABC):
    @abstractmethod
    def _name(self) -> str: ...

    def required(self) -> bool:
        return False

    def default(self) -> Path:
        return get_project_home() / self._name()

    def metavar(self) -> str:
        return f"{self._name().upper()}_PATH"

class FileCLIArgument(PathCLIArgument, ABC):
    def long_name(self) -> str:
        return f"--{self._name()}"

    def _validate_value(self, value: Path) -> None:
        if not value.exists():
            raise FileNotFoundError(f"File not found at {value}")
        elif value.is_dir():
            raise IsADirectoryError(f"Path {value} specified as a file is"
                                    " a directory")

class DirectoryCLIArgument(PathCLIArgument, ABC):
    def long_name(self) -> str:
        return f"--{self._name()}-dir"

    def _validate_value(self, value: Path) -> None:
        if not value.exists():
            value.mkdir()
        elif not value.is_dir():
            raise NotADirectoryError(f"Path {value} specified as a directory"
                                     "is a file.")
