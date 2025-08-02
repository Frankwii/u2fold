#!/usr/bin/env python3

"""Try different hyperparameter combinations in a systematic manner"""

from collections.abc import Iterable, Mapping, Sequence
from u2fold.model import U2FoldSpec
import itertools

type JSON = Mapping[str, "JSON"] | Sequence["JSON"] | str | int | float | bool | None
type SequenceJSON = Mapping[str, "SequenceJSON" | Sequence[JSON]]

def unroll_all_dict_combinations(d: SequenceJSON) -> Iterable[JSON]:
    keys = list(d.keys())

    return map(
        lambda t: dict(zip(keys, t)), itertools.product(*d.values())
    )

def generate_unet_specs() -> Iterable[JSON]:  
    channels_per_layer_combinations = [
        [4, 8, 16],
        [8, 16, 32],
        [32, 64]
    ]

    pooling_methods = [
        "max",
        "avg",
        "l2"
    ]

    pooling_method_combinations = [{
        "method": method,
        "kernel_size": 2,
        "stride": 2,
    } for method in pooling_methods]
    
    activation_function_combinations = {"name": n for n in [
        "gelu",
        "relu"
    ]}

    combinations = {
        "name": ["unet"],
        "channels_per_layer": channels_per_layer_combinations,
        "sublayers_per_step": [2, 3, 4],
        "unfolded_step_size": [0.01, 0.001],
        "pooling": pooling_method_combinations,
        "activation": activation_function_combinations
    }

    return unroll_all_dict_combinations(combinations)


def generate_dataset_specs() -> Iterable[JSON]:
    yield {
        "name": "uieb",
        "path": "uieb/processed",
        "n_epocs": 100,
        "batch_size": 8
    }

def generate_losses_specs() -> Iterable[JSON]:
    yield [
      {"loss":"ground_truth", "weight": 1.0},
      {"loss":"fidelity", "weight": 0.2},
      {"loss":"consistency", "weight": 0.01},
      {"loss":"color_cosine_similarity", "weight": 0.01}
    ]

def generate_algorithmic_specs() -> Iterable[JSON]:
    combinations = {
        "transmission_map_patch_radius": [8, 13],
        "guided_filter_patch_radius":[8, 13],
        "transmission_map_saturation_coefficient": [0.5, 0.75],
        "guided_filter_regularization_coefficient": [0.01, 0.001],
        "step_size": [0.01, 0.001]
    }

    return unroll_all_dict_combinations(combinations)

def generate_learning_rate_scheduler_specs() -> Iterable[JSON]:
    step_spec = unroll_all_dict_combinations({
        "scheduler": ["adam_lr"],
        "step_size": [20, 50],
        "factor": [0.1, 0.2, 0.5]
    })

    cosine_annealing_spec = unroll_all_dict_combinations({
        "scheduler": ["cosine_annealing_lr"],
        "semiperiod": [25, 50],
    })

    return itertools.chain(step_spec, cosine_annealing_spec)

def generate_optimizer_specs() -> Iterable[JSON]:
    adam = unroll_all_dict_combinations({
        "optimizer": ["adam"],
        "learning_rate": [0.01, 0.001]
    })

    sgd = unroll_all_dict_combinations({
        "optimizer": ["sgd"],
        "learning_rate": [0.01, 0.001],
        "momentum": [0, 0.1]
    })

    return itertools.chain(adam, sgd)

def evaluate_spec(spec: U2FoldSpec) -> float:
    return 0.0

def search_best_combination():
    default_spec = {
        "algorithmic_spec":{
            "transmission_map_patch_radius": 8,
            "guided_filter_patch_radius": 8,
            "transmission_map_saturation_coefficient": 0.5,
            "guided_filter_regularization_coefficient": 0.01,
            "step_size": 0.01
        }
    }
    all_model_combinations = generate_unet_specs()

    # TODO
