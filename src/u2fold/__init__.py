from argparse import ArgumentParser, Namespace
from pathlib import Path
from enum import Enum
from scripts.generate_spec_schema import show_docs
from u2fold.model import U2FoldSpec
from u2fold.orchestrate import get_orchestrator
import orjson

class Mode(Enum):
    run = "run"
    docs = "docs"

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

    return parser

def run(args: Namespace) -> None:
    with open(args.spec, 'r') as f:
        spec_json = orjson.loads(f.read())

    spec = U2FoldSpec.model_validate(spec_json)

    get_orchestrator(spec).run()

def main() -> None:
    parser = build_parser()

    args = parser.parse_args()

    match Mode(args.command):
        case Mode.run: run(args)
        case Mode.docs: show_docs()
        case _: print("none!"); exit(-1)
