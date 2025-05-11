from abc import ABC, abstractmethod
from argparse import ArgumentParser
from textwrap import dedent
from typing import Any, Optional

from u2fold.utils.singleton_metaclasses import AbstractSingleton


class CLIArgument(ABC, metaclass = AbstractSingleton):

    def short_name(self) -> Optional[str]:
        return None

    def metavar(self) -> Optional[str]:
        return None

    def choices(self) -> Optional[list[str]]:
        return None

    def required(self) -> bool:
        return True

    def default(self) -> Optional[str]:
        return None

    @abstractmethod
    def long_name(self) -> str:
        ...

    @abstractmethod
    def value_type(self) -> type:
        ...

    @abstractmethod
    def help(self) -> str:
        ...

    def _help(self) -> str:
        return dedent(self.help()).strip()

    def validate_value(self, value: Any) -> None:
        if not isinstance(value, t := self.value_type()):
            raise TypeError(
                f"Invalid type when parsing CLIArgument \
                `{type(self)}`. Expected `{t}`, got `{type(value)}`."
            )

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


