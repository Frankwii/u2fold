#!/usr/bin/python
"""
Compute transmission maps (both coarse and fine) for each of the images passed as arguments.

As defined in https://doi.org/10.1109/TCSVT.2021.3115791.
"""

# Disable compilation of pytorch functions because images are usually
# heterogeneous in shape. Run this with TORCH_COMPILE_DISABLE=0 if you
# are certain that your images have similar dimensions and want the
# performance gains.
import os

if os.getenv("TORCH_COMPILE_DISABLE") is None:
    os.environ["TORCH_COMPILE_DISABLE"] = "1"

import itertools
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from typing import Iterable, Iterator

import PIL.Image
import torch
from torch import Tensor, multiprocessing
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from u2fold.math.background_light_estimation import estimate_background_light
from u2fold.math.guided_filter import guided_filter
from u2fold.math.transmission_map_estimation import (
    compute_saturation_map,
    estimate_coarse_red_transmission_map,
)


def compute_transmission_maps(
    image: Tensor,
    patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
    use_red_map_only: bool
) -> tuple[Tensor, Tensor, Tensor]:
    # (C, H, W) -> ((1, H, W), (1, H, W))
    H, W = image.shape[-2:]

    batched = image.reshape(1, 3, H, W)

    saturation_map = compute_saturation_map(batched).reshape(1, H, W)

    background_light = estimate_background_light(batched)
    coarse_transmission_map = estimate_coarse_red_transmission_map(
        batched, background_light, patch_radius, saturation_coef
    ).reshape(1, H, W)

    # guide = batched[:, 0, ...].unsqueeze(1) if use_red_map_only else batched
    guide = batched
    # guide = batched[:, 0, ...].unsqueeze(1)
    fine_transmission_map = guided_filter(
        guide=guide,
        input=coarse_transmission_map.reshape(1, 1, H, W),
        patch_radius=int(patch_radius * 1.5),
        regularization_coefficient=regularization_coef,
    ).reshape(1, H, W)

    return (saturation_map, coarse_transmission_map, fine_transmission_map)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "-r",
        "--radius",
        type=int,
        nargs="+",
    )

    parser.add_argument(
        "-l",
        "--lambda",
        "--saturation-coefficient",
        type=float,
        dest="saturation_coef",
        nargs="+",
    )

    parser.add_argument(
        "-e",
        "--epsilon",
        "--regularization-coefficient",
        type=float,
        dest="regularization_coef",
        nargs="+",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory path",
        type=Path,
        dest="output_dir",
        default=Path("transmission_maps"),
    )

    parser.add_argument(
        "-i", help="Input files", type=Path, dest="input_files", nargs="+"
    )

    return parser


def validate_directory(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Invalid image directory path: {path}")


def tensor_to_image(img: Tensor) -> PIL.Image.Image:
    # (C, H, W) -> (H, W, C)
    assert img.ndim == 3, f"{img.shape=}"

    numpy_img = (
        (img * 255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    )

    return PIL.Image.fromarray(numpy_img.squeeze())


def save_transmission_maps(
    dir_path: Path,
    img_name: str,
    saturation: Tensor,
    coarse_tm: Tensor,
    fine_tm: Tensor,
) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "saturation").mkdir(exist_ok=True)
    (dir_path / "coarse").mkdir(exist_ok=True)
    (dir_path / "fine").mkdir(exist_ok=True)

    saturation_path = dir_path / "saturation" / img_name
    coarse_path = dir_path / "coarse" / img_name
    fine_path = dir_path / "fine" / img_name

    tensor_to_image(coarse_tm).save(coarse_path)
    tensor_to_image(fine_tm).save(fine_path)
    tensor_to_image(saturation).save(saturation_path)


def format_param_comb_name(
    patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
) -> str:
    return f"patchRadius_{patch_radius}__saturation_{saturation_coef}__regularization{regularization_coef}"


def format_parameters(
    root_dir: Path,
    image_paths: Iterable[Path],
    hyperparameter_combinations: Iterable[tuple[int, float, float, bool]],
) -> Iterator[tuple[Path, Path, int, float, float, bool]]:
    for image_path, params in product(image_paths, hyperparameter_combinations):
        yield (root_dir, image_path, *params)


def single_image_job(
    output_dir: Path,
    image_path: Path,
    patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
    use_red_map_only: bool,
) -> None:
    image_name = image_path.name
    try:
        pil_image = PIL.Image.open(image_path).convert("RGB")
        image = to_tensor(pil_image)
    except Exception as e:
        print("=======ERROR=======", flush=True)
        print(f"Error parsing {image_name} in {image_path}.", flush=True)
        print("==TRACE==", flush=True)
        print(e, flush=True)
        print("===================", flush=True)
        return

    try:
        saturation, coarse_tm, fine_tm = compute_transmission_maps(
            image, patch_radius, saturation_coef, regularization_coef, use_red_map_only
        )
    except Exception as e:
        print("=======ERROR=======", flush=True)
        print(
            f"Error processing {image_name} with params (patch_radius={patch_radius}, saturation_coef={saturation_coef}, regularization_coef={regularization_coef})",
            flush=True,
        )
        print("==TRACE==", flush=True)
        print(e, flush=True)
        print("===================", flush=True)
        return

    subdir_name = format_param_comb_name(
        patch_radius, saturation_coef, regularization_coef
    )

    try:
        save_transmission_maps(
            output_dir / subdir_name, image_name, saturation, coarse_tm, fine_tm
        )
    except Exception as e:
        print("=======ERROR=======", flush=True)
        print(
            f"Error saving processed {image_name} into subdir {subdir_name}",
            flush=True,
        )
        print("==TRACE==", flush=True)
        print(e, flush=True)
        print("===================", flush=True)


def wrapped_single_image_job(t: tuple[Path, Path, int, float, float, bool]) -> None:
    return single_image_job(*t)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "input_files"):
        raise AttributeError("Missing input files!")

    patch_radii = args.radius
    saturation_coefs = args.saturation_coef
    regularization_coefs = args.regularization_coef
    use_red_map_only = getattr(args, "use_red_map_only", False)

    hyperparameter_combinations = list(itertools.product(
        patch_radii, saturation_coefs, regularization_coefs, [use_red_map_only]
    ))

    all_parameter_combinations = list(
        format_parameters(
            args.output_dir, args.input_files, hyperparameter_combinations
        )
    )

    n_jobs = len(all_parameter_combinations)

    with multiprocessing.Pool() as pool:
        for _ in tqdm(
            pool.imap(wrapped_single_image_job, all_parameter_combinations),
            total=n_jobs,
        ):
            pass


if __name__ == "__main__":
    main()
