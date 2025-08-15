"""Try different hyperparameter combinations in a systematic manner"""

import itertools
from collections.abc import Iterable, Mapping, Sequence

type JSON = Mapping[str, "JSON"] | Sequence["JSON"] | str | int | float | bool | None
type SequenceJSON = Mapping[str, "SequenceJSON" | Iterable[JSON]]


def unroll_all_dict_combinations(d: SequenceJSON) -> Iterable[JSON]:
    keys = list(d.keys())

    return map(lambda t: dict(zip(keys, t)), itertools.product(*d.values()))


def generate_unet_specs() -> list[JSON]:
    channels_per_layer_combinations = [[4, 8, 16], [8, 16, 32], [32, 48]]

    pooling_methods = ["max", "avg", "l2"]

    pooling_method_combinations = [
        {
            "method": method,
            "kernel_size": 2,
            "stride": 2,
        }
        for method in pooling_methods
    ]

    activation_function_combinations = [{"name": n} for n in ("gelu", "relu")]

    combinations = {
        "name": ["aunet"],
        "channels_per_layer": channels_per_layer_combinations,
        "sublayers_per_step": [2, 3],
        "unfolded_step_size": [0.01, 0.001],
        "pooling": pooling_method_combinations,
        "activation": activation_function_combinations,
    }

    return list(unroll_all_dict_combinations(combinations))


def generate_dataset_specs() -> Iterable[JSON]:
    yield {"name": "uieb", "path": "uieb/processed", "n_epocs": 100, "batch_size": 8}


def generate_losses_specs() -> Iterable[JSON]:
    yield [
        {"loss": "ground_truth", "weight": 1.0},
        {"loss": "fidelity", "weight": 0.2},
        {"loss": "consistency", "weight": 0.01},
        {"loss": "color_cosine_similarity", "weight": 0.01},
    ]


def generate_learning_rate_scheduler_specs() -> Iterable[JSON]:
    step_spec = unroll_all_dict_combinations(
        {
            "scheduler": ["step_lr"],
            "step_size": [5, 10, 25],
            "factor": [0.1, 0.2, 0.5],
        }
    )

    cosine_annealing_spec = unroll_all_dict_combinations(
        {
            "scheduler": ["cosine_annealing_lr"],
            "semiperiod": [25, 50],
        }
    )

    return itertools.chain(step_spec, cosine_annealing_spec)


def generate_optimizer_specs() -> Iterable[JSON]:
    adam = unroll_all_dict_combinations(
        {"optimizer": ["adam"], "learning_rate": [0.01, 0.001]}
    )

    sgd = unroll_all_dict_combinations(
        {"optimizer": ["sgd"], "learning_rate": [0.01, 0.001], "momentum": [0, 0.1]}
    )

    return itertools.chain(adam, sgd)


def generate_train_mode_specs() -> Iterable[JSON]:
    combinations = {
        "optimizer_spec": generate_optimizer_specs(),
        "learning_rate_scheduler_spec": generate_learning_rate_scheduler_specs(),
    }

    fixed_mode_spec = {
        "mode": "train",
        "losses": [
            {"loss": "ground_truth", "weight": 0.5},
            {"loss": "fidelity", "weight": 0.3},
            {"loss": "consistency", "weight": 0.1},
            {"loss": "color_cosine_similarity", "weight": 0.1},
        ],
        "dataset_spec": {
            "name": "uieb",
            "path": "uieb/processed",
            "n_epochs": 50,
            "batch_size": 4,
        },
    }

    return (  # pyright: ignore[reportUnknownVariableType]
        {"mode_spec": fixed_mode_spec | d}  # pyright: ignore[reportOperatorIssue]
        for d in unroll_all_dict_combinations(combinations)
    )


def generate_train_specs() -> list[JSON]:
    fixed_top_level_spec = {
        "log_level": "warning",
        "algorithmic_spec": {
            "transmission_map_patch_radius": 8,
            "guided_filter_patch_radius": 15,
            "transmission_map_saturation_coefficient": 0.5,
            "guided_filter_regularization_coefficient": 0.01,
            "step_size": 0.01,
        },
    }

    return [  # pyright: ignore[reportUnknownVariableType]
        fixed_top_level_spec | train_mode_spec  # pyright: ignore[reportOperatorIssue]
        for train_mode_spec in generate_train_mode_specs()
    ]


def base_training_spec() -> JSON:
    return {
        "log_level": "warning",
        "mode_spec": {
            "mode": "train",
            "losses": [
                {"loss": "ground_truth", "weight": 0.5},
                {"loss": "fidelity", "weight": 0.3},
                {"loss": "consistency", "weight": 0.1},
                {"loss": "color_cosine_similarity", "weight": 0.1},
            ],
            "dataset_spec": {
                "name": "uieb",
                "path": "uieb/processed",
                "n_epochs": 50,
                "batch_size": 4,
            },
            "optimizer_spec": {
                "optimizer": "adam",
                "learning_rate": 0.01,
            },
            "learning_rate_scheduler_spec": {
                "scheduler": "step_lr",
                "step_size": 10,
                "factor": 0.5,
            },
        },
        "algorithmic_spec": {
            "transmission_map_patch_radius": 8,
            "guided_filter_patch_radius": 15,
            "transmission_map_saturation_coefficient": 0.5,
            "guided_filter_regularization_coefficient": 0.01,
            "step_size": 0.01,
        },
    }
