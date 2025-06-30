import logging
from pathlib import Path

from u2fold import build_parser
from u2fold.config_parsing import parse_and_validate_config
from u2fold.config_parsing.config_dataclasses import U2FoldConfig
from u2fold.orchestrate.orchestrator import get_orchestrator


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = parse_and_validate_config(args)

    bootstrap_logger(config)

    orchestrator = get_orchestrator(config)

    orchestrator.run()


def bootstrap_logger(config: U2FoldConfig) -> None:
    fmt = "{asctime} | [{levelname:<8}]@{name}(line {lineno:0>3}): {message}"

    config.execution_log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        filename=config.execution_log_dir / "exec.log",
        style="{",
        format=fmt,
        datefmt="%Y/%m/%d %H:%M:%S",
    )


# def instantiate_orchestrator(
#     config: U2FoldConfig, logger: logging.Logger
# ) -> Orchestrator:
#     logger.info("Instantiating program orchestrator...")
#     if isinstance(config, TrainConfig):
#         orchestrator = TrainOrchestrator(config)
#         mode = "train"
#     elif isinstance(config, ExecConfig):
#         orchestrator = ExecOrchestrator(config)
#         mode = "exec"
#     else:
#         raise TypeError("Got invalid configuration type.")
#
#     logger.debug(
#         f"Successfully instantiated program orchestrator. Mode: {mode}"
#     )
#
#     return orchestrator


if __name__ == "__main__":
    main()
