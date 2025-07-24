import bisect
import random

import numpy as np


class Probability:
    def __init__(self, val: float) -> None:
        if not 0 <= val <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        self.__val = val

    def __repr__(self) -> str:
        return f"Probability({self.__val})"

    def happens(self) -> bool:
        return random.random() <= self.__val


class Distribution:
    """A mathematical probability distribution to be sampled from.

    The class should be instantiated with an iterable of nonnegative
    numbers summing to one.

    The instantiation is O(n) compute and memory-wise, where n is the number
    of values used to instantiate it.
    """

    def __init__(self, *values: float) -> None:
        if not abs(sum(values) - 1) < 1e-10:
            raise ValueError("Probability distribution must sum to 1.")

        for value in values:
            if value < 0:
                raise ValueError("Probability must be nonnegative.")

        self.__separators = np.cumsum(values)
        self.__separators[-1] = 1  # To account for rounding errors

    def __repr__(self) -> str:
        original_values = np.diff(self.__separators, prepend=0)

        return f"Distribution({','.join(original_values)})"

    def sample(self) -> int:
        """Samples an index from this distribution.

        Sampling is done according to the probabilities used to instantiate it.
        This implementation is O(log(n)), where n is the number of values used
        to instantiate the distribution.

        Examples:
            >>> dist = Distribution(0.5, 0.5)
            >>> dist.sample()

            The above yields 0 half the times and 1 the other half.

            >>> dist = Distribution(0.5, 0.1, 0.4)
            >>> dist.sample()

            The above yields 0 with a 50% probability, 1 with a 10% probability
            and 2 with a 40% probability.
        """
        return bisect.bisect_left(self.__separators, random.random())
