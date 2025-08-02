"""Create splits randomly (but reproducibly) for a given Dataset."""

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar, cast

from torch import Generator, Tensor
from torch.utils.data import Dataset, random_split

RANDOM_SEED = 42


T = TypeVar("T", covariant=True)
@dataclass
class SplitData(Generic[T]):
    training: T
    validation: T
    test: T

    def to_tuple(self) -> tuple[T, T, T]:
        return (
            self.training,
            self.validation,
            self.test,
        )

    def map[A, U](
        self,
        f: Callable[[T, A], U] | Callable[[T], U],
        params: "SplitData[A] | None" = None,
    ) -> "SplitData[U]":
        if params is not None:
            f = cast(Callable[[T, A], U], f)
            return SplitData(
                training=f(self.training, params.training),
                validation=f(self.validation, params.validation),
                test=f(self.test, params.test),
            )

        f = cast(Callable[[T], U], f)
        return SplitData(
            training=f(self.training),
            validation=f(self.validation),
            test=f(self.test),
        )

@dataclass
class DatasetSplits(SplitData[float]):
    def __post_init__(self) -> None:
        splits = self.to_tuple()
        if not abs(s:=sum(splits) - 1) < 1e-8:
            raise ValueError(f"Splits must sum to 1! Value: {s}.")
        for val in splits:
            if not val > 0:
                raise ValueError("All split fractions must be positive.")


def split_dataset[T: Dataset[Tensor]](
    dataset: T,
    splits: DatasetSplits,
) -> SplitData[T]:
    train, valid, test = random_split(
        dataset,
        splits.to_tuple(),
        generator=Generator().manual_seed(RANDOM_SEED),
    )

    return cast(SplitData[T], SplitData(train, valid, test))
