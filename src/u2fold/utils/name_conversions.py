"""Naming convention utilities."""


def cli_to_snake(cli_long_name: str) -> str:
    """Formats a CLI "long" name into snake_case.

    Examples:
        >>> cli_to_snake("--cli-arg")
        cli_arg
    """
    return cli_long_name.lstrip("-").replace("-", "_")


def snake_to_cli(snake_case_name: str) -> str:
    """Formats a snake_case name into a "long" CLI name.

    Examples:
        >>> snake_to_cli("snake_case")
        --snake-case
    """
    return f"--{snake_case_name.replace('_', '-')}"
