import argparse
import shutil
from pathlib import Path
from typing import cast

import PIL.Image


def __assert_directory_exists(p: Path) -> None:
    if not p.exists():
        raise IOError(f"Provided path {p} does not exist.")

    if not p.is_dir():
        raise NotADirectoryError(f"Provided path {p} is not a directory.")


def get_input_arguments() -> tuple[Path, tuple[int, int]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "uieb_path", type=Path, help="Path to the UEIB dataset directory."
    )
    parser.add_argument(
        "output_resolution",
        type=int,
        help= ("Output resolution for the images"
               " (width and height will be the same)."),
    )

    args = parser.parse_args()

    path = cast(Path, args.uieb_path)
    output_resolution_side = cast(int, args.output_resolution)
    if output_resolution_side <= 0:
        raise ValueError("Output resolution must be a positive integer.")

    __assert_directory_exists(path)

    return path, (output_resolution_side, output_resolution_side)


def validate_raw_path(uieb_path: Path) -> None:
    raw_path = uieb_path / "raw"
    __assert_directory_exists(raw_path)

    input_path = raw_path / "input"
    ground_truth_path = raw_path / "ground_truth"

    __assert_directory_exists(input_path)
    __assert_directory_exists(ground_truth_path)


def prepare_processed_directory(uieb_path: Path) -> None:
    """Prepare the processed directory for the UEIB dataset.
    Returns True if the execution of this script should continue.
    """
    processed_path = uieb_path / "processed"

    if processed_path.exists():
        user_input = input(
            f"Directory for processed UEIB dataset already exists:"
            f" {processed_path}.\nWould you like to overwrite it? [y/N] "
        )

        if user_input.lower() != "y":
            print("Exiting...")
            exit(0)

    processed_path.mkdir(exist_ok=True)

    for contents in processed_path.iterdir():
        if contents.is_dir():
            shutil.rmtree(contents)
        else:
            contents.unlink()

    for subdir in ["input", "ground_truth"]:
        subdir_path = processed_path / subdir
        subdir_path.mkdir()


def crop_upper_left_square(image: PIL.Image.Image) -> PIL.Image.Image:
    """Crop the upper left square from the image."""
    width, height = image.size
    min_side = min(width, height)
    return image.crop((0, 0, min_side, min_side))


def resize_square(
    image: PIL.Image.Image, output_resolution: tuple[int, int]
) -> PIL.Image.Image:
    """Resize the square image to the specified resolution."""
    return image.resize(output_resolution, PIL.Image.Resampling.BICUBIC)


def process_image(
    input_path: Path, output_path: Path, output_resolution: tuple[int, int]
) -> None:
    """Process a single image by cropping it to the upper left square and
    resizing it to the specified resolution."""
    image = PIL.Image.open(input_path)
    cropped_image = crop_upper_left_square(image)
    resized_image = resize_square(cropped_image, output_resolution)
    resized_image.save(output_path)


if __name__ == "__main__":
    uieb_path, output_resolution = get_input_arguments()

    validate_raw_path(uieb_path)
    prepare_processed_directory(uieb_path)

    raw_path = uieb_path / "raw"
    processed_path = uieb_path / "processed"

    for subdir in ["input", "ground_truth"]:
        input_subdir = raw_path / subdir
        output_subdir = processed_path / subdir

        for image_file in input_subdir.iterdir():
            output_file = output_subdir / image_file.name
            process_image(image_file, output_file, output_resolution)
