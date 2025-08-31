from argparse import ArgumentParser, Namespace
import json
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
from u2fold.model.common_namespaces import EpochMetricData
from u2fold.utils.results_db import get_best_result, get_results_from_spec, save_training_result, spec_is_in_db, table_columns

class Mode(Enum):
    run = "run"
    docs = "docs"
    calibrate = "calibrate"
    search = "search-hyperparameters"
    query = "query-results"

def build_parser() -> ArgumentParser:
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(
        title="Commands",
        dest="command",
        required=True
    )

    run_parser = subparsers.add_parser("run", help="Run with given spec.")
    run_parser.add_argument("--spec", type=Path, default="spec.json", help="Path to spec.")
    subparsers.add_parser("docs", help="Generate documentation for JSON spec.")

    calibrate_parser = subparsers.add_parser("calibrate", help="Calibrate all metrics and loss terms.")
    calibrate_parser.add_argument("path", type=Path)

    subparsers.add_parser("search-hyperparameters", help="'Grid search' a large set of hyperparameter combinations.")
    subparsers.add_parser("query-results", help="Query training results obtained so far.")

    return parser

def run(args: Namespace) -> None:
    with open(args.spec, 'r') as f:
        spec_json = orjson.loads(f.read())

    spec = U2FoldSpec.model_validate(spec_json)
    orchestrator = get_orchestrator(spec)

    should_train = isinstance(orchestrator, TrainOrchestrator)
    print("Running!")
    if should_train:
        if spec_is_in_db(spec):
            print("This exact spec was already trained. Results:\n\n")
            print(json.dumps(get_results_from_spec(spec), indent=2))

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
    res = orchestrator.run()
    if should_train:
        assert isinstance(res, EpochMetricData)
        save_training_result(spec, res)
        msg = "Traning has finished, but the tensorboard process will be kept running."
        orchestrator._logger.warning(msg)
        print(msg + " Please kill the process to stop it.")
        tensorboard_process.wait()  # pyright: ignore[reportPossiblyUnboundVariable]

def get_best_spec_in_db():
    valid_metrics = set(table_columns.keys()) - {"spec"}

    msg = f"Choose a metric or press Ctrl+C to exit. Valid options:\n\t{"\n\t".join(valid_metrics)}\nChoice: "
    while (metric := input(msg)):
        if metric not in valid_metrics:
            print("Unknown metric.")
            continue

        print(json.dumps(get_best_result(metric), indent=2))

def main() -> None:
    parser = build_parser()

    args = parser.parse_args()

    match Mode(args.command):
        case Mode.run: run(args)
        case Mode.docs: show_docs()
        case Mode.calibrate: calibrate_metrics(args.path)
        case Mode.search: search_best_combination()
        case Mode.query: get_best_spec_in_db()
