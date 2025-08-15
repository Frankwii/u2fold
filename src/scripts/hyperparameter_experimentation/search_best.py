import shutil
from pathlib import Path

import orjson
from tqdm import tqdm

from u2fold.model.spec import U2FoldSpec
from u2fold.utils.get_directories import get_weight_directory

from .evaluate_model import measure_spec
from .generate_specs import (
    base_training_spec,
    generate_train_specs,
    generate_unet_specs,
)


def format_spec_training_related_name(spec) -> str:
    optimizer_spec = spec["mode_spec"]["optimizer_spec"]
    scheduler_spec = spec["mode_spec"]["learning_rate_scheduler_spec"]

    optimizer_values = (v for k, v in optimizer_spec.items() if k != "optimizer")
    scheduler_values = (v for k, v in optimizer_spec.items() if k != "scheduler")

    return f"{optimizer_spec['optimizer']}_{'_'.join(map(str, optimizer_values))}__{scheduler_spec['scheduler']}_{'_'.join(map(str, scheduler_values))}"


def move_directory(current_path: Path, new_path: Path):
    new_path.parent.mkdir(parents=True, exist_ok=True)

    if new_path.exists():
        shutil.rmtree(new_path)

    _ = shutil.move(current_path, new_path)


def process_spec(spec) -> dict:  # pyright: ignore[reportUnknownParameterType, reportMissingTypeArgument, reportMissingParameterType]
    with open("debug_spec", "w") as f:
        import json
        json.dump(spec, f)
    spec_model = U2FoldSpec.model_validate(spec)

    model_weight_directory = get_weight_directory(spec_model.neural_network_spec)

    new_weight_directory = (
        model_weight_directory.parent
        / "hyperparameter_search"
        / format_spec_training_related_name(spec)  # pyright: ignore[reportUnknownArgumentType]
        / model_weight_directory.name
    )

    result = {"spec": spec, "score": measure_spec(spec_model)}  # pyright: ignore[reportUnknownVariableType]

    move_directory(model_weight_directory, new_weight_directory)

    return result  # pyright: ignore[reportUnknownVariableType]


def search_best_combination() -> None:
    base_spec = base_training_spec()

    all_model_combinations = generate_unet_specs()

    architectural_results = [  # pyright: ignore[reportUnknownVariableType]
        process_spec(base_spec | {"neural_network_spec": model_spec})  # pyright: ignore[reportOperatorIssue, reportUnknownArgumentType]
        for model_spec in tqdm(
            all_model_combinations,
            desc="Architectural combinations tried",
            total=len(all_model_combinations),
        )
    ]

    architectural_results = sorted(  # pyright: ignore[reportUnknownVariableType]
        architectural_results, key=lambda d: d["score"]  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    )

    best_neural_network_spec = architectural_results[0]["spec"]["neural_network_spec"]  # pyright: ignore[reportUnknownVariableType]

    all_training_related_combinations = generate_train_specs()

    training_related_results = [  # pyright: ignore[reportUnknownVariableType]
        process_spec(
            training_related_spec | {"neural_network_spec": best_neural_network_spec}  # pyright: ignore[reportOperatorIssue, reportUnknownArgumentType]
        )
        for training_related_spec in tqdm(
            all_training_related_combinations,
            desc="Training-related combinatinos tried",
            total=len(all_training_related_combinations),
        )
    ]

    all_results = sorted(  # pyright: ignore[reportUnknownVariableType]
        architectural_results + training_related_results,  # pyright: ignore[reportUnknownArgumentType]
        key=lambda d: d["score"],  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    )

    with open("aunet_results.json", "wb") as f:
        _ = f.write(orjson.dumps(all_results, option=orjson.OPT_INDENT_2))
