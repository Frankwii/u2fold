import orjson
from tqdm import tqdm
from .evaluate_model import measure_spec;
from .generate_specs import base_training_spec, generate_train_specs, generate_unet_specs;
from u2fold.model.spec import U2FoldSpec


def search_best_combination() -> None:
    base_spec = base_training_spec()

    all_model_combinations = generate_unet_specs()

    architectural_results = [  # pyright: ignore[reportUnknownVariableType]
        {
            "spec": (
                (spec := (base_spec | {"neural_network_spec": model_spec}))  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
            ),
            "score": measure_spec(U2FoldSpec.model_validate(spec)),
        }
        for model_spec in tqdm(all_model_combinations, desc="Architectural combinations tried", total = len(all_model_combinations))
    ]

    architectural_results = sorted(architectural_results, key=lambda d: d["score"], reverse=True)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType, reportUnknownLambdaType]

    best_neural_network_spec = architectural_results[0]["spec"]["neural_network_spec"]  # pyright: ignore[reportIndexIssue, reportUnknownVariableType]

    all_training_related_combinations = generate_train_specs()

    training_related_results = [  # pyright: ignore[reportUnknownVariableType]
        {
            "spec": (spec := (training_related_spec | {"neural_network_spec": best_neural_network_spec})),  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
            "score": measure_spec(U2FoldSpec.model_validate(spec))
        }

        for training_related_spec in tqdm(
            all_training_related_combinations, desc="Training-related combinatinos tried", total=len(all_training_related_combinations)
        )
    ]

    all_results = sorted(architectural_results + training_related_results, key=lambda d: d["score"], reverse=True)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportUnknownLambdaType]


    with open("aunet_results.json", "wb") as f:
        _ = f.write(orjson.dumps(all_results, option=orjson.OPT_INDENT_2))
