#!/usr/bin/python

import itertools
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
from compute_transmission_maps import format_param_comb_name
from matplotlib.axes import Axes
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
    image_name: str,
    patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
) -> tuple[Axes, Axes, Axes, Axes, Axes]:
    title = f"Transmission maps for {image_name}. Parameters:\n Patch radius={patch_radius}, Saturation coefficient={saturation_coef}, Regularization coefficient={regularization_coef}"

    fig, (original, red, saturation, coarse, fine) = plt.subplots(1, 5)
    fig.suptitle(title)

    original.set_title("Original input image.")
    red.set_title("Red channel.")
    saturation.set_title("Saturation map.")
    coarse.set_title("Coarse transmission map.")
    fine.set_title("Fine transmission map.")

    return original, red, saturation, coarse, fine


def load_transmission_maps(
    image: Path,
    patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
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


def process_image(
    image: Path,
    patch_radius: int,
    saturation_coef: float,
    regularization_coef: float,
) -> None:
    images = load_transmission_maps(
        image, patch_radius, saturation_coef, regularization_coef
    )

    image_name = image.name
    axes = create_transmission_map_axes(
        image_name, patch_radius, saturation_coef, regularization_coef
    )

    for img, ax in zip(images, axes):
        ax.imshow(img, cmap="jet", vmin=0, vmax=1)

    plt.show()


def main():
    parser = build_parser()
    args = parser.parse_args()

    compute_transmission_maps(args)

    hyperparameter_combinations = list(
        itertools.product(
            args.radius, args.saturation_coef, args.regularization_coef
        )
    )

    for image_path in args.input_files:
        for t in hyperparameter_combinations:
            process_image(image_path, *t)


if __name__ == "__main__":
    main()
