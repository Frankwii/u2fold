import orjson
from tqdm import tqdm
from .evaluate_model import measure_spec;
from .generate_specs import base_training_spec, generate_unet_specs;
from u2fold.model.spec import U2FoldSpec


def search_best_combination() -> None:
    base_spec = base_training_spec()

    all_model_combinations = generate_unet_specs()

    results = [
        {
            "spec": (
                spec := U2FoldSpec.model_validate(base_spec | {"neural_network_spec": model_spec})  # pyright: ignore[reportOperatorIssue]
            ).model_dump_json(),
            "score": measure_spec(spec),
        }
        for model_spec in tqdm(all_model_combinations, desc="Architectural combinations tried", total = len(all_model_combinations))
    ]

    results = sorted(results, key=lambda d: d["score"])

    print(f"Best combination was:\n\n{results[1]}")

    with open("aunet_results.json", "wb") as f:
        _ = f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))
