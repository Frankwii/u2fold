import logging
import inspect

from u2fold import build_parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    fmt = "{asctime} | [{levelname:<8}]@{name}(line {lineno:0>3}): {message}"
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        style="{",
        format=fmt,
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    print(args)


if __name__ == "__main__":
    main()
