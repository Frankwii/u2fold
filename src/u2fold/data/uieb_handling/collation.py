from torch import Tensor
from torch.utils.data import default_collate

from u2fold.data.image_transformations import (
    horizontal_flip,
    rotate_right_angle,
)
from u2fold.utils.probability import Distribution, Probability
from u2fold.utils.singleton_metaclasses import Singleton


class UIEBCollateAndTransform(metaclass=Singleton):
    def __init__(self) -> None:
        self.__flip_probability = Probability(0.5)
        self.__rotation_probabilities = Distribution(0.25, 0.25, 0.25, 0.25)

    def __call__(
        self,
        batch_pairs: list[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        input_batch, ground_truth_batch = default_collate(batch_pairs)

        possibly_rotated_batches = horizontal_flip(
            self.__flip_probability, (input_batch, ground_truth_batch)
        )

        input_batch, ground_truth_batch = rotate_right_angle(
            self.__rotation_probabilities, possibly_rotated_batches
        )

        return input_batch, ground_truth_batch
