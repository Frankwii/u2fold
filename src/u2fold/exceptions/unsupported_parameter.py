from typing import Optional, Sequence


class UnsupportedParameter(Exception):
    def __init__[T: str](
        self, parameter: T, supported_options: Optional[Sequence[T]]
    ) -> None:
        errmsg = f"Parameter with value {parameter} is not supported."

        if supported_options is not None:
            errmsg += f"\nSupported options: {', '.join(supported_options)}"
        super().__init__(errmsg)
