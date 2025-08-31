from typing import Any, NamedTuple

from torch import Tensor

from u2fold.model.common_namespaces import ForwardPassResult
from u2fold.utils.dict_utils import merge_sum, shallow_dict_map


def no_op(*_: Any, **__: Any): ... # pyright: ignore[reportAny, reportUnusedParameter]
"""Do nothing."""


class MetricRegister(NamedTuple):
    overall_loss: Tensor
    granular_loss: dict[str, float]
    metrics: dict[str, float]

class EpochMetricData(NamedTuple):
    overall_loss: float
    granular_loss: dict[str, float]
    metrics: dict[str, float]


class MetricAccumulator:
    def __init__(self):
        self.__cumulative_overall_loss: float = 0.0
        self.__cumulative_granular_loss: dict[str, float] = {}
        self.__cumulative_metrics: dict[str, float] = {}
        self.__counter = 0

    def add_register(self, register: MetricRegister) -> None:
        self.__counter += 1
        self.__cumulative_overall_loss += register.overall_loss.detach().mean().item()
        self.__cumulative_granular_loss = merge_sum(
            self.__cumulative_granular_loss, register.granular_loss
        )
        self.__cumulative_metrics = merge_sum(
            self.__cumulative_metrics, register.metrics
        )

    def average(self) -> EpochMetricData:
        if self.__counter == 0:
            raise ValueError("Add at least one register first!")
        return EpochMetricData(
            self.__cumulative_overall_loss / self.__counter,
            shallow_dict_map(
                lambda n: n / self.__counter, self.__cumulative_granular_loss
            ),
            shallow_dict_map(
                lambda n: n / self.__counter, self.__cumulative_metrics
            )
        )

class LossComputingData(NamedTuple):
    input: Tensor
    output: ForwardPassResult
    ground_truth: Tensor
