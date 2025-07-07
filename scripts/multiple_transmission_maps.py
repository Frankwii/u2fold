#!/usr/bin/python
"""
Compute transmission maps (coarse and multiple refinements) for each of the images passed as arguments, and visualize them.
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
from typing import Iterable, Iterator, List

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from matplotlib.figure import Figure
from numpy import ndarray
from torch import Tensor, multiprocessing
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from u2fold.math.background_light_estimation import estimate_background_light
from u2fold.math.guided_filter import guided_filter
from u2fold.math.transmission_map_estimation import (
    estimate_coarse_transmission_map,
)

TRANSMISSION_MAPS_ROOT_DIR = Path("/tmp/u2fold/transmission_maps_multiple")


def compute_transmission_maps(
    image: Tensor,
    rcp_patch_radius: int,
    guided_filter_patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
    number_of_extra_refinements: int,
) -> tuple[Tensor, list[Tensor]]:
    # (C, H, W) -> ((1, H, W), list[(1, H, W)])
    H, W = image.shape[-2:]

    batched = image.reshape(1, 3, H, W)

    background_light = estimate_background_light(batched)
    coarse_transmission_map = estimate_coarse_transmission_map(
        batched, background_light, rcp_patch_radius, saturation_coef
    ).reshape(1, H, W)

    refined_maps = []
    current_map = coarse_transmission_map.reshape(1, 1, H, W)

    for _ in range(1 + number_of_extra_refinements):
        refined_map = guided_filter(
            guide=batched,
            input=current_map,
            patch_radius=guided_filter_patch_radius,
            regularization_coefficient=regularization_coef,
        )  # output is (1, 1, H, W)
        refined_maps.append(refined_map.reshape(1, H, W))
        current_map = refined_map

    return coarse_transmission_map, refined_maps


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
    coarse_tm: Tensor,
    refined_maps: list[Tensor],
) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "coarse").mkdir(exist_ok=True)

    coarse_path = dir_path / "coarse" / img_name
    tensor_to_image(coarse_tm).save(coarse_path)

    for i, refined_map in enumerate(refined_maps):
        refined_dir = dir_path / f"refined_{i}"
        refined_dir.mkdir(exist_ok=True)
        refined_path = refined_dir / img_name
        tensor_to_image(refined_map).save(refined_path)


def format_param_comb_name(
    rcp_patch_radius: int,
    guided_filter_patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
    number_of_extra_refinements: int,
) -> str:
    return (
        f"rcpPatchRadius_{rcp_patch_radius}__"
        f"guidedFilterPatchRadius_{guided_filter_patch_radius}__"
        f"saturation_{saturation_coef}__"
        f"regularization_{regularization_coef}__"
        f"extraRefinements_{number_of_extra_refinements}"
    )


def format_parameters(
    root_dir: Path,
    image_paths: Iterable[Path],
    hyperparameter_combinations: Iterable[tuple[int, int, float, float, int]],
) -> Iterator[tuple[Path, Path, int, int, float, float, int]]:
    for image_path, params in product(image_paths, hyperparameter_combinations):
        yield (root_dir, image_path, *params)


def single_image_job(
    output_dir: Path,
    image_path: Path,
    rcp_patch_radius: int,
    guided_filter_patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
    number_of_extra_refinements: int,
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
        coarse_tm, refined_maps = compute_transmission_maps(
            image,
            rcp_patch_radius,
            guided_filter_patch_radius,
            saturation_coef,
            regularization_coef,
            number_of_extra_refinements,
        )
    except Exception as e:
        print("=======ERROR=======", flush=True)
        print(
            f"Error processing {image_name} with params ("
            f"rcp_patch_radius={rcp_patch_radius}, "
            f"guided_filter_patch_radius={guided_filter_patch_radius}, "
            f"saturation_coef={saturation_coef}, "
            f"regularization_coef={regularization_coef}, "
            f"number_of_extra_refinements={number_of_extra_refinements})",
            flush=True,
        )
        print("==TRACE==", flush=True)
        print(e, flush=True)
        print("===================", flush=True)
        return

    subdir_name = format_param_comb_name(
        rcp_patch_radius,
        guided_filter_patch_radius,
        saturation_coef,
        regularization_coef,
        number_of_extra_refinements,
    )

    try:
        save_transmission_maps(
            output_dir / subdir_name,
            image_name,
            coarse_tm,
            refined_maps,
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


def wrapped_single_image_job(
    t: tuple[Path, Path, int, int, float, float, int],
) -> None:
    return single_image_job(*t)


def create_transmission_map_axes(
    image_names: list[str],
    rcp_patch_radius: int,
    guided_filter_patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
    number_of_extra_refinements: int,
) -> tuple[Figure, ndarray]:
    column_titles = [
        "Original input image",
        "Coarse TM",
        "Fine TM",
    ] + [
        f"Extra Refinement {i + 1}" for i in range(number_of_extra_refinements)
    ]

    num_images = len(image_names)
    num_columns = len(column_titles)

    fig, axes = plt.subplots(
        num_images, num_columns, figsize=(18, num_images * 4)
    )

    if num_images == 1:
        axes = np.atleast_2d(axes)

    overall_title = (
        "Original images and transmission maps with multiple refinements."
        " Parameters:\n"
        f"RCP Patch Radius = {rcp_patch_radius}, "
        f"Guided Filter Patch Radius = {guided_filter_patch_radius}, "
        f"Saturation Coefficient = {saturation_coef}, "
        f"Regularization Coefficient = {regularization_coef}, "
        f"Num Extra Refinements = {number_of_extra_refinements}"
    )
    fig.suptitle(overall_title, fontsize=16)

    for col_idx, title_text in enumerate(column_titles):
        axes[0, col_idx].set_title(title_text, fontsize=12)

    for row_idx, image_name in enumerate(image_names):
        axes[row_idx, 0].set_ylabel(image_name, rotation=90, fontsize=12)

        for col_idx in range(num_columns):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, axes


def load_transmission_maps(
    image: Path,
    rcp_patch_radius: int,
    guided_filter_patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
    number_of_extra_refinements: int,
    output_dir: Path,
) -> list[ndarray]:
    original = plt.imread(image)

    subdir_name = format_param_comb_name(
        rcp_patch_radius,
        guided_filter_patch_radius,
        saturation_coef,
        regularization_coef,
        number_of_extra_refinements,
    )

    subdir = output_dir / subdir_name
    image_name = image.name

    coarse = plt.imread(subdir / "coarse" / image_name)

    refined_maps = []
    for i in range(1 + number_of_extra_refinements):
        refined_map = plt.imread(subdir / f"refined_{i}" / image_name)
        refined_maps.append(refined_map)

    return [original, coarse] + refined_maps


def visualize_transmission_maps(
    images: list[Path],
    rcp_patch_radius: int,
    guided_filter_patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
    number_of_extra_refinements: int,
    output_dir: Path,
) -> None:
    images_to_display = [
        load_transmission_maps(
            image,
            rcp_patch_radius,
            guided_filter_patch_radius,
            saturation_coef,
            regularization_coef,
            number_of_extra_refinements,
            output_dir,
        )
        for image in images
    ]

    image_names = [image.name for image in images]
    _, axes = create_transmission_map_axes(
        image_names,
        rcp_patch_radius,
        guided_filter_patch_radius,
        saturation_coef,
        regularization_coef,
        number_of_extra_refinements,
    )

    for row_idx, img_results in enumerate(images_to_display):
        for col_idx, img_result in enumerate(img_results):
            axes[row_idx, col_idx].imshow(
                img_result, cmap="jet", vmin=0, vmax=1
            )

    plt.show()


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "--rcp-patch-radius",
        type=int,
        nargs="+",
        required=True,
        dest="rcp_patch_radii",
    )

    parser.add_argument(
        "--guided-filter-patch-radius",
        type=int,
        nargs="+",
        required=True,
        dest="guided_filter_patch_radii",
    )

    parser.add_argument(
        "-l",
        "--lambda",
        "--saturation-coefficient",
        type=float,
        dest="saturation_coefs",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "-e",
        "--epsilon",
        "--regularization-coefficient",
        type=float,
        dest="regularization_coefs",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "-n",
        "--number-of-extra-refinements",
        type=int,
        nargs="+",
        dest="numbers_of_extra_refinements",
        required=True,
        help="Number of extra guided filter refinements to apply.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory path",
        type=Path,
        dest="output_dir",
        default=TRANSMISSION_MAPS_ROOT_DIR,
    )

    parser.add_argument(
        "-i",
        help="Input files",
        type=Path,
        dest="input_files",
        nargs="+",
        required=True,
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    rcp_patch_radii = args.rcp_patch_radii
    guided_filter_patch_radii = args.guided_filter_patch_radii
    saturation_coefs = args.saturation_coefs
    regularization_coefs = args.regularization_coefs
    numbers_of_extra_refinements = args.numbers_of_extra_refinements

    hyperparameter_combinations_for_compute = list(
        itertools.product(
            rcp_patch_radii,
            guided_filter_patch_radii,
            saturation_coefs,
            regularization_coefs,
            numbers_of_extra_refinements,
        )
    )

    all_parameter_combinations = list(
        format_parameters(
            args.output_dir,
            args.input_files,
            hyperparameter_combinations_for_compute,
        )
    )

    n_jobs = len(all_parameter_combinations)
    print(f"Computing {n_jobs} transmission map(s) combinations...")

    with multiprocessing.Pool() as pool:
        for _ in tqdm(
            pool.imap(wrapped_single_image_job, all_parameter_combinations),
            total=n_jobs,
        ):
            pass

    print("Finished computing transmission maps.")

    print("Visualizing results...")
    hyperparameter_combinations_for_viz = list(
        itertools.product(
            rcp_patch_radii,
            guided_filter_patch_radii,
            saturation_coefs,
            regularization_coefs,
            numbers_of_extra_refinements,
        )
    )

    for t in hyperparameter_combinations_for_viz:
        visualize_transmission_maps(args.input_files, *t, args.output_dir)


if __name__ == "__main__":
    main()
