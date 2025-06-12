from typing import Iterable

from torch import Tensor
from torchvision.transforms.functional import hflip, rotate

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
