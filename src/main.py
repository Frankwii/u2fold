import logging
from argparse import Namespace
from pathlib import Path

from u2fold import build_parser
from u2fold.config_parsing.config_dataclasses import ExecConfig, TrainConfig, U2FoldConfig
from u2fold.data.uieb_handling import get_dataloaders
from u2fold.orchestrate.orchestrator import (
    ExecOrchestrator,
    Orchestrator,
    TrainOrchestrator,
)

import matplotlib.pyplot as plt


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    path = Path("/home/frank/TFM/code/u2fold/uieb") # TODO: refactor to properly handle "processed".
    train, valid, test = get_dataloaders(path, 16, "cpu").to_tuple()

    fig, ax = plt.subplots(2,2)
    for input, ground_truth in train:
        first_input_image = input[0].permute(2, 1, 0)
        first_ground_truth_image = ground_truth[0].permute(2, 1, 0)
        second_input_image = input[1].permute(2, 1, 0)
        second_ground_truth_image = ground_truth[1].permute(2, 1, 0)

        ax[0,0].imshow(first_input_image)
        ax[0,0].set_title("Input 1")
        ax[0,1].imshow(first_ground_truth_image)
        ax[0,1].set_title("Ground Truth 1")
        ax[1,0].imshow(second_input_image)
        ax[1,0].set_title("Input 2")
        ax[1,1].imshow(second_ground_truth_image)
        ax[1,1].set_title("Ground Truth 2")

        plt.show()


    # print(args)

    # logger = bootstrap_logger(args)
    # program_config = build_config(args, logger)
    #
    # orchestrator = instantiate_orchestrator(program_config, logger)
    #
    # orchestrator.run()


def bootstrap_logger(args: Namespace) -> logging.Logger:
    fmt = "{asctime} | [{levelname:<8}]@{name}(line {lineno:0>3}): {message}"
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        filename=args.log_dir / "exec.log",
        style="{",
        format=fmt,
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    return logging.getLogger(__name__)


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
