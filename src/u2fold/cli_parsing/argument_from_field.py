from argparse import ArgumentParser
from dataclasses import Field
from typing import Any, Collection, cast, get_args, get_origin

from u2fold.utils.name_conversions import snake_to_cli


class ArgumentFromField:
    def __init__(self, field: Field) -> None:
        self._type = None
        self._nargs = None
        self._choices = None
        self._required = True
        self._action = "store"
        self.__field = field

        self.__add_long_name()
        self.__add_help()
        self.__resolve_type()

    def __add_long_name(self) -> None:
        self._long_name = snake_to_cli(self.__field.name)

    def __add_help(self) -> None:
        self._help = self.__field.metadata.get("desc", "No help available.")

    def __resolve_collection(self, origin: Any, args: tuple) -> bool:
        if origin is not None and issubclass(origin, Collection):
            self._type = args[0]
            self._nargs = "+"
            self._action = "extend"
            return True

        return False

    def __resolve_choices(self) -> bool:
        metadata_choices = self.__field.metadata.get("choices")
        if metadata_choices is not None:
            self._type = str
            self._choices = metadata_choices

            return True
        return False

    def __resolve_type(self) -> None:
        field_type = self.__field.type
        annotation_origin = get_origin(field_type)
        annotation_args = get_args(field_type)

        if self.__resolve_collection(annotation_origin, annotation_args):
            return

        if self.__resolve_choices():
            return

        self._type = field_type

    def add_to_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            self._long_name,
            action=self._action,
            nargs=self._nargs,
            choices=self._choices,
            help=self._help,
            required=self._required,
            type=cast(type, self._type),
        )
