from typing import Iterable

from numpy import random
from torch import Tensor
from torchvision.transforms.functional import crop, hflip, rotate

from u2fold.utils.probability import Distribution, Probability


def horizontal_flip(
    p: Probability, images: Iterable[Tensor]
) -> Iterable[Tensor]:
    """Flips images with probability p.

    Either flips all of them or flips none of them."""
    if p.happens():
        return (hflip(image) for image in images)
    return images


def rotate_right_angle(
    dist: Distribution, images: Iterable[Tensor]
) -> Iterable[Tensor]:
    """Rotates an angle positively in accordance to the specified distribution.

    - Images are all rotated in the same way (just one "coin toss").
    - The number of right angle rotations is the index sampled from `dist`.
    - Rotations are mathematically positive, i.e., counter-clockwise.
    """

    number_of_rotations = dist.sample()

    return (rotate(image, 90 * number_of_rotations) for image in images)

def crop_random_maximal_square(
    width: int, height: int, images: Iterable[Tensor]
) -> Iterable[Tensor]:
    """Crops a random, maximally sized square from the given images.

    All given images should have the same size (width, height).
    """
    if width < height:
        square_side = width
        offset = random.randint(height - square_side)

        return (
            crop(image, offset, 0, square_side, width)
            for image in images
        )
    else:
        square_side = height
        offset = random.randint(width - square_side)

        return (
            crop(image, 0, offset, height, square_side)
            for image in images
        )

def crop_top_left_maximal_square(
    width: int, height: int, images: Iterable[Tensor]
) -> Iterable[Tensor]:
    """Crops a maximally sized square from the top left corner of the given
    images.

    All given images should have the same size (width, height).
    """
    square_side = min(width, height)

    return (
        crop(image, 0, 0, square_side, square_side)
        for image in images
    )
