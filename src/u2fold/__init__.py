import u2fold.models as models
from u2fold.cli_parsing import build_parser

__all__ = ["models", "build_parser"]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print(args)
