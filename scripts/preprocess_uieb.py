#!/usr/bin/python
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

import PIL.Image


def __assert_directory_exists(p: Path) -> None:
    if not p.exists():
        raise IOError(f"Provided path {p} does not exist.")

    if not p.is_dir():
        raise NotADirectoryError(f"Provided path {p} is not a directory.")


def get_input_arguments() -> tuple[Path, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "uieb_path", type=Path, help="Path to the UEIB dataset directory."
    )
    parser.add_argument(
        "output_resolution",
        type=int,
        help=(
            "Output resolution for the images"
            " (width and height will be the same)."
        ),
    )

    args = parser.parse_args()

    path = cast(Path, args.uieb_path)
    output_resolution_side = cast(int, args.output_resolution)
    if output_resolution_side <= 0:
        raise ValueError("Output resolution must be a positive integer.")

    __assert_directory_exists(path)

    return path, output_resolution_side


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


def crop_upper_left(
    image: PIL.Image.Image, width: int, height: int
) -> PIL.Image.Image:
    return image.crop((0, 0, width, height))


def crop_largest_band(
    image: PIL.Image.Image, multiplier: float = 0.8
) -> PIL.Image.Image:
    """Crop the upper left square from the image."""
    width, height = image.size

    if width < height:
        width = int(width * multiplier)
    else:
        height = int(height * multiplier)

    return crop_upper_left(image, width, height)


def resize_square(
    image: PIL.Image.Image, output_resolution: int
) -> PIL.Image.Image:
    """Resize the square image to the specified resolution."""
    width, height = image.size
    aspect_ratio = height / width

    if width < height:
        resolution = (output_resolution, int(output_resolution * aspect_ratio))
    else:
        resolution = (int(output_resolution / aspect_ratio), output_resolution)

    return image.resize(resolution, PIL.Image.Resampling.BICUBIC)


def process_image(
    input_path: Path, output_path: Path, output_resolution: int
) -> None:
    """Process a single image by cropping it to the upper left square and
    resizing it to the specified resolution."""
    image = PIL.Image.open(input_path)
    cropped_image = crop_largest_band(image, 0.8)
    resized_image = resize_square(cropped_image, output_resolution)
    resized_image.save(output_path)


def get_watermarked_images() -> set[str]:
    return {
        "326_img_.png",
        "189_img_.png",
        "259_img_.png",
        "336_img_.png",
        "348_img_.png",
        "834_img_.png",
        "871_img_.png",
        "35.png",
        "7027.png",
        "7643.png",
        "7654.png",
        "8046.png",
        "8262.png",
    }


def get_low_quality_images() -> set[str]:
    return {
        "18_img_.png",
        "710_img_.png",
        "294_img_.png",
        "295_img_.png",
        "299_img_.png",
        "313_img_.png",
        "324_img_.png",
        "330_img_.png",
        "337_img_.png",
        "290_img_.png",
        "347_img_.png",
        "300_img_.png",
        "301_img_.png",
        "355_img_.png",
        "302_img_.png",
        "361_img_.png",
        "306_img_.png",
        "332_img_.png",
        "362_img_.png",
        "307_img_.png",
        "364_img_.png",
        "366_img_.png",
        "404_img_.png",
        "421_img_.png",
        "445_img_.png",
        "454_img_.png",
        "455_img_.png",
        "457_img_.png",
        "460_img_.png",
        "471_img_.png",
        "472_img_.png",
        "476_img_.png",
        "481_img_.png",
        "487_img_.png",
        "516_img_.png",
        "555_img_.png",
        "567_img_.png",
        "568_img_.png",
        "580_img_.png",
        "608_img_.png",
        "679_img_.png",
        "77_img_.png",
        "793_img_.png",
        "849_img_.png",
        "850_img_.png",
        "860_img_.png",
        "904_img_.png",
        "905_img_.png",
        "534.png",
        "557.png",
        "551.png",
        "567.png",
        "756.png",
        "837.png",
        "841.png",
        "810.png",
        "2774.png",
        "6062.png",
        "9900.png",
        "12290.png",
        "12348.png",
        "15426.png",
    }

def prepare_images(uieb_path: Path, output_resolution: int):
    validate_raw_path(uieb_path)
    prepare_processed_directory(uieb_path)

    raw_path = uieb_path / "raw"
    processed_path = uieb_path / "processed"

    should_be_excluded = get_low_quality_images() | get_watermarked_images()

    for subdir in ["input", "ground_truth"]:
        input_subdir = raw_path / subdir
        output_subdir = processed_path / subdir

        for image_file in input_subdir.iterdir():
            if image_file.name not in should_be_excluded:
                output_file = output_subdir / image_file.name
                yield (image_file, output_file, output_resolution)


if __name__ == "__main__":
    uieb_path, output_resolution = get_input_arguments()

    image_args = list(prepare_images(uieb_path, output_resolution))

    image_files, output_files, output_resolutions = zip(*image_args)


    with ProcessPoolExecutor() as executor:
        list(executor.map(process_image, image_files, output_files, output_resolutions))
