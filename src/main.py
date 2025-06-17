import logging
import time
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt

from u2fold import build_parser
from u2fold.config_parsing.config_dataclasses import (
    ExecConfig,
    TrainConfig,
    U2FoldConfig,
)
from u2fold.data.uieb_handling import get_dataloaders
from u2fold.data.uieb_handling.dataset import UIEBDataset
from u2fold.orchestrate.orchestrator import (
    ExecOrchestrator,
    Orchestrator,
    TrainOrchestrator,
)


def main() -> None:

    # torch.multiprocessing.set_sharing_strategy("file_descriptor")
    parser = build_parser()
    args = parser.parse_args()

    bootstrap_logger(args)
    path = Path("/home/frank/TFM/code/u2fold/uieb/processed/")

    dataset = UIEBDataset(path)

    train, valid, test = get_dataloaders(path, 16, "cpu").to_tuple()

def bootstrap_logger(args: Namespace) -> None:
    fmt = "{asctime} | [{levelname:<8}]@{name}(line {lineno:0>3}): {message}"

    args.log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        filename=args.log_dir / "exec.log",
        style="{",
        format=fmt,
        datefmt="%Y/%m/%d %H:%M:%S",
    )



def build_config(args: Namespace, logger: logging.Logger) -> U2FoldConfig:
    logger.info("Successfully parsed CLI arguments. Instantiating config...")
    if args.mode == "train":
        conf = TrainConfig(args)
    elif args.mode == "exec":
        conf = ExecConfig(args)
    else:  # It should be impossible to end up here, but just in case.
        raise ValueError(
            "Invalid mode specified for the program. Mode should be either"
            ' "train" or "exec".'
        )

    logger.debug("Successfully instantiated configuration classes.")
    return conf


def instantiate_orchestrator(
    config: U2FoldConfig, logger: logging.Logger
) -> Orchestrator:
    logger.info("Instantiating program orchestrator...")
    if isinstance(config, TrainConfig):
        orchestrator = TrainOrchestrator(config)
        mode = "train"
    elif isinstance(config, ExecConfig):
        orchestrator = ExecOrchestrator(config)
        mode = "exec"
    else:
        raise TypeError("Got invalid configuration type.")

    logger.debug(
        f"Successfully instantiated program orchestrator. Mode: {mode}"
    )

    return orchestrator


if __name__ == "__main__":
    main()
