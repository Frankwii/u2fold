#!/usr/bin/python
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(type=Path, dest="img_paths", nargs="+")

    return parser


def plot_grayscale_image(*paths: Path):
    imgs = [plt.imread(p) for p in paths]

    for img in imgs:
        plt.imshow(img, cmap="jet")
        plt.show()


if __name__ == "__main__":
    input = (
        build_parser().parse_args().img_paths
    )

    plot_grayscale_image(*input)
