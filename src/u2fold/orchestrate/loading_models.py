"""
Utilities for loading model weights.
"""
from pathlib import Path
from typing import cast

import torch

from u2fold.orchestrate.greedy_iteration_models import GreedyIterationModels
from u2fold.utils.track import get_from_tag


def __from_weight_file(
    model_class: type[torch.nn.Module],
    weight_file: Path
) -> torch.nn.Module:
    model = torch.nn.utils.skip_init(model_class)
    model.load_state_dict(torch.load(weight_file))

    return model

def __from_weight_dir(
    model_class: type[torch.nn.Module],
    weight_dir: Path
) -> list[torch.nn.Module]:
    models = [
        __from_weight_file(model_class, weight_file)
        for weight_file, _, _ in weight_dir.walk()
    ]

    return models

def load_models(
    model_name: str,
    weight_files_dir: Path
) -> list[GreedyIterationModels]:

    model_tag = f"model/{model_name}"
    model_class = cast(type[torch.nn.Module], get_from_tag(model_tag))

    greedy_iter_dirs = [d for d, _, _ in weight_files_dir.walk()]
    models = [([], []) for _ in range(len(greedy_iter_dirs))]

    for greedy_iter, greedy_iter_dir in enumerate(greedy_iter_dirs):
        image_dir = greedy_iter_dir.joinpath("image")

        image_models = __from_weight_dir(model_class, image_dir)

        kernel_dir = greedy_iter_dir.joinpath("kernel")

        kernel_models = __from_weight_dir(model_class, kernel_dir)

        models[greedy_iter] = GreedyIterationModels(
            image=image_models,
            kernel=kernel_models
        )

    return cast(list[GreedyIterationModels], models)
