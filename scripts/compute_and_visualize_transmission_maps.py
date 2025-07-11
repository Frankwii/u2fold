#!/usr/bin/python
"""
Compute transmission maps (both coarse and fine) for each of the images passed as arguments,
and visualize them.

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
    compute_saturation_map,
    estimate_coarse_red_transmission_map,
)

TRANSMISSION_MAPS_ROOT_DIR = Path("/tmp/u2fold/transmission_maps")


def compute_transmission_maps(
    image: Tensor,
    rcp_patch_radius: int,
    guided_filter_patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    H, W = image.shape[-2:]

    batched = image.reshape(1, 3, H, W)

    saturation_map = compute_saturation_map(batched).reshape(1, H, W)

    background_light = estimate_background_light(batched)
    coarse_transmission_map = estimate_coarse_red_transmission_map(
        batched, background_light, rcp_patch_radius, saturation_coef
    ).reshape(1, H, W)

    fine_tm_color_guide = guided_filter(
        guide=batched,
        input=coarse_transmission_map.reshape(1, 1, H, W),
        patch_radius=guided_filter_patch_radius,
        regularization_coefficient=regularization_coef,
    ).reshape(1, H, W)

    red_channels = batched[:, 0, ...].unsqueeze(1)
    fine_tm_red_guide = guided_filter(
        guide=red_channels,
        input=coarse_transmission_map.reshape(1, 1, H, W),
        patch_radius=guided_filter_patch_radius,
        regularization_coefficient=regularization_coef,
    ).reshape(1, H, W)

    return (
        saturation_map,
        coarse_transmission_map,
        fine_tm_color_guide,
        fine_tm_red_guide,
    )


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
    fine_tm_color_guide: Tensor,
    fine_tm_red_guide: Tensor,
) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "saturation").mkdir(exist_ok=True)
    (dir_path / "coarse").mkdir(exist_ok=True)
    (dir_path / "fine_color_guide").mkdir(exist_ok=True)
    (dir_path / "fine_red_guide").mkdir(exist_ok=True)

    saturation_path = dir_path / "saturation" / img_name
    coarse_path = dir_path / "coarse" / img_name
    fine_color_guide_path = dir_path / "fine_color_guide" / img_name
    fine_red_guide_path = dir_path / "fine_red_guide" / img_name

    tensor_to_image(coarse_tm).save(coarse_path)
    tensor_to_image(fine_tm_color_guide).save(fine_color_guide_path)
    tensor_to_image(fine_tm_red_guide).save(fine_red_guide_path)
    tensor_to_image(saturation).save(saturation_path)


def format_param_comb_name(
    rcp_patch_radius: int,
    guided_filter_patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
) -> str:
    return (
        f"rcpPatchRadius_{rcp_patch_radius}__"
        f"guidedFilterPatchRadius_{guided_filter_patch_radius}__"
        f"saturation_{saturation_coef}__"
        f"regularization_{regularization_coef}"
    )


def format_parameters(
    root_dir: Path,
    image_paths: Iterable[Path],
    hyperparameter_combinations: Iterable[tuple[int, int, float, float]],
) -> Iterator[tuple[Path, Path, int, int, float, float]]:
    for image_path, params in product(image_paths, hyperparameter_combinations):
        yield (root_dir, image_path, *params)


def single_image_job(
    output_dir: Path,
    image_path: Path,
    rcp_patch_radius: int,
    guided_filter_patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
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
        (
            saturation,
            coarse_tm,
            fine_tm_color_guide,
            fine_tm_red_guide,
        ) = compute_transmission_maps(
            image,
            rcp_patch_radius,
            guided_filter_patch_radius,
            saturation_coef,
            regularization_coef,
        )
    except Exception as e:
        print("=======ERROR=======", flush=True)
        print(
            f"Error processing {image_name} with params ("
            f"rcp_patch_radius={rcp_patch_radius}, "
            f"guided_filter_patch_radius={guided_filter_patch_radius}, "
            f"saturation_coef={saturation_coef}, "
            f"regularization_coef={regularization_coef})",
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
    )

    try:
        save_transmission_maps(
            output_dir / subdir_name,
            image_name,
            saturation,
            coarse_tm,
            fine_tm_color_guide,
            fine_tm_red_guide,
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
    t: tuple[Path, Path, int, int, float, float],
) -> None:
    return single_image_job(*t)


def create_transmission_map_axes(
    image_names: list[str],
    rcp_patch_radius: int,
    guided_filter_patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
) -> tuple[Figure, ndarray]:
    column_titles = [
        "Original input image",
        "Red channel",
        "Saturation map",
        "Coarse TM",
        "Fine TM, color guide",
        "Fine TM, red guide",
    ]

    num_images = len(image_names)
    num_columns = len(column_titles)

    fig, axes = plt.subplots(
        num_images, num_columns, figsize=(18, num_images * 4)
    )

    if num_images == 1:
        axes = np.atleast_2d(axes)

    overall_title = (
        "Original images, red channels, saturation map of transmission maps."
        " Parameters:\n"
        f"RCP Patch Radius = {rcp_patch_radius}, "
        f"Guided Filter Patch Radius = {guided_filter_patch_radius}, "
        f"Saturation Coefficient = {saturation_coef}, "
        f"Regularization Coefficient = {regularization_coef}"
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
    output_dir: Path,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    original = plt.imread(image)

    subdir_name = format_param_comb_name(
        rcp_patch_radius,
        guided_filter_patch_radius,
        saturation_coef,
        regularization_coef,
    )

    subdir = output_dir / subdir_name

    image_name = image.name
    saturation = plt.imread(subdir / "saturation" / image_name)
    coarse = plt.imread(subdir / "coarse" / image_name)
    fine_color_guide = plt.imread(subdir / "fine_color_guide" / image_name)
    fine_red_guide = plt.imread(subdir / "fine_red_guide" / image_name)

    return (
        original,
        original[..., 0],
        saturation,
        coarse,
        fine_color_guide,
        fine_red_guide,
    )


def visualize_transmission_maps(
    images: list[Path],
    rcp_patch_radius: int,
    guided_filter_patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
    output_dir: Path,
) -> None:
    images_to_display = [
        load_transmission_maps(
            image,
            rcp_patch_radius,
            guided_filter_patch_radius,
            saturation_coef,
            regularization_coef,
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

    hyperparameter_combinations_for_compute = list(
        itertools.product(
            rcp_patch_radii,
            guided_filter_patch_radii,
            saturation_coefs,
            regularization_coefs,
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

    # --- Visualization ---
    print("Visualizing results...")
    hyperparameter_combinations_for_viz = list(
        itertools.product(
            rcp_patch_radii,
            guided_filter_patch_radii,
            saturation_coefs,
            regularization_coefs,
        )
    )

    for t in hyperparameter_combinations_for_viz:
        visualize_transmission_maps(args.input_files, *t, args.output_dir)


if __name__ == "__main__":
    main()
