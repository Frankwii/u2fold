#!/usr/bin/python

import itertools
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
from compute_transmission_maps import format_param_comb_name
from matplotlib.figure import Figure
from numpy import ndarray

TRANSMISSION_MAPS_ROOT_DIR = Path("/tmp/u2fold/transmission_maps")
COMPUTE_TRANSMISSION_MAPS_SCRIPT = "scripts/compute_transmission_maps.py"


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
        "-i", help="Input files", type=Path, dest="input_files", nargs="+"
    )

    return parser


def compute_transmission_maps(args: Namespace):
    params = (
        sys.executable,
        COMPUTE_TRANSMISSION_MAPS_SCRIPT,
        "--radius",
        *map(str, args.radius),
        "--lambda",
        *map(str, args.saturation_coef),
        "--epsilon",
        *map(str, args.regularization_coef),
        "--output-dir",
        TRANSMISSION_MAPS_ROOT_DIR,
        "-i",
        *args.input_files,
    )

    subprocess.run(params)


def create_transmission_map_axes(
    image_names: list[str],
    regularization_coef: float,
    patch_radius: int,
    saturation_coef: float,
) -> tuple[Figure, ndarray]:
    column_titles = [
        "Original input image",
        "Red channel",
        "Saturation map",
        "Coarse transmission map",
        "Fine transmission map",
    ]

    num_images = len(image_names)
    num_columns = len(column_titles)

    fig, axes = plt.subplots(num_images, num_columns)

    overall_title = (
        f"Original images, red channels, saturation map of transmission maps."
        " Parameters:\n"
        f"Patch Radius = {patch_radius}, "
        f"Saturation Coefficient = {saturation_coef}, "
        f"Regularization Coefficient = {regularization_coef}"
    )
    fig.suptitle(overall_title, fontsize=16)

    for col_idx, title_text in enumerate(column_titles):
        axes[0, col_idx].set_title(title_text, fontsize=12)

    for row_idx, image_name in enumerate(image_names):
        axes[row_idx, 0].set_ylabel(
            image_name, rotation=90, fontsize=12
        )

        for col_idx in range(num_columns):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    fig.tight_layout()

    return fig, axes


def load_transmission_maps(
    image: Path,
    regularization_coef: float,
    patch_radius: int,
    saturation_coef: float,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    original = plt.imread(image)

    subdir_name = format_param_comb_name(
        patch_radius, saturation_coef, regularization_coef
    )

    subdir = TRANSMISSION_MAPS_ROOT_DIR / subdir_name

    image_name = image.name
    saturation = plt.imread(subdir / "saturation" / image_name)
    coarse = plt.imread(subdir / "coarse" / image_name)
    fine = plt.imread(subdir / "fine" / image_name)

    return original, original[..., 0], saturation, coarse, fine


def process_images(
    images: list[Path],
    regularization_coef: float,
    patch_radius: int,
    saturation_coef: float,
) -> None:
    images_to_display = [
        load_transmission_maps(
            image, regularization_coef, patch_radius, saturation_coef 
        )
        for image in images
    ]

    image_names = [image.name for image in images]
    _, axes = create_transmission_map_axes(
        image_names, regularization_coef, patch_radius, saturation_coef
    )

    for row_idx, img_results in enumerate(images_to_display):
        for col_idx, img_result in enumerate(img_results):
            axes[row_idx, col_idx].imshow(
                img_result, cmap="jet", vmin=0, vmax=1
            )

    plt.show()


def main():
    parser = build_parser()
    args = parser.parse_args()

    compute_transmission_maps(args)

    hyperparameter_combinations = list(
        itertools.product(
            args.regularization_coef, args.radius, args.saturation_coef,
        )
    )

    for t in hyperparameter_combinations:
        process_images(args.input_files, *t)


if __name__ == "__main__":
    main()
