from u2fold import build_parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print(args)


if __name__ == "__main__":
    main()
