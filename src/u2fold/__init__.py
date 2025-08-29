from argparse import ArgumentParser, Namespace
from pathlib import Path
from enum import Enum
import shlex
import shutil
import subprocess
from scripts.generate_spec_schema import show_docs
from scripts.metric_calibration import calibrate_metrics
from scripts.hyperparameter_experimentation import search_best_combination
from u2fold.model import U2FoldSpec
from u2fold.orchestrate import get_orchestrator
import orjson

from u2fold.orchestrate.train import TrainOrchestrator

class Mode(Enum):
    run = "run"
    docs = "docs"
    calibrate = "calibrate"
    search = "search-hyperparameters"

def build_parser() -> ArgumentParser:
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(
        title="Commands",
        dest="command",
        required=True
    )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--spec", type=Path, default="spec.json")
    subparsers.add_parser("docs")

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("path", type=Path)

    subparsers.add_parser("search-hyperparameters")

    return parser

def run(args: Namespace) -> None:
    with open(args.spec, 'r') as f:
        spec_json = orjson.loads(f.read())

    spec = U2FoldSpec.model_validate(spec_json)
    orchestrator = get_orchestrator(spec)

    should_train = isinstance(orchestrator, TrainOrchestrator)
    print("Running!")
    if should_train:
        if orchestrator._tensorboard_log_dir.exists():
            orchestrator._logger.warning(
                "Found an existing tensorboard log directory. Emptying it."
            )
            shutil.rmtree(orchestrator._tensorboard_log_dir)
            orchestrator._tensorboard_log_dir.mkdir()

        orchestrator._logger.debug("Starting tensorboard process")
        tensorboard_process = subprocess.Popen(
            shlex.split(
                f"tensorboard --port 6066 --logdir {orchestrator._tensorboard_log_dir}"
            )
        )
    orchestrator.run()
    if should_train:
        msg = "Traning has finished, but the tensorboard process will be kept running."
        orchestrator._logger.warning(msg)
        print(msg + " Please kill the process to stop it.")
        tensorboard_process.wait()  # pyright: ignore[reportPossiblyUnboundVariable]

def main() -> None:
    parser = build_parser()

    args = parser.parse_args()

    match Mode(args.command):
        case Mode.run: run(args)
        case Mode.docs: show_docs()
        case Mode.calibrate: calibrate_metrics(args.path)
        case Mode.search: search_best_combination()
