"""
"Calibrate" metrics for the uieb dataset so that they all have an average value of 1 for the UIEB dataset.

More concretely, for each metric, the average value for UIEB images is computed. When computing the metrics later on, the obtained values should be divided by said average, so that they all have an average of 1 and thus it makes more sense to combine their values with arithmetic operations.

In the case of unsupervised metrics, the computation is done for the ground truth. For supervised metrics, it is done for the (input, gruond truth) pair.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, NamedTuple

import torch
from torch import Tensor

from u2fold.data.uieb_handling.dataset import UIEBDataset
from u2fold.neural_networks.metrics_and_losses import (
    color_minimizable,
    dssim,
    mse,
    psnr_minimizable,
    total_variation,
    uciqe_minimizable,
)
from u2fold.orchestrate.functional.computation import compute_deterministic_components


class Stats(NamedTuple):
    std: Tensor
    avg: Tensor


type UnsupervisedMetric = Callable[[Tensor], Tensor]

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_unsupervised_metric_stats(
    metric: UnsupervisedMetric,
    dataset: UIEBDataset,
) -> Stats:
    metric_values = torch.tensor(
        [
            metric(ground_truth.unsqueeze(0).to(_DEVICE))
            for (_, ground_truth) in iter(dataset)
        ]
    )

    return Stats(*torch.std_mean(metric_values.abs()))


type SupervisedMetric = Callable[[Tensor, Tensor], Tensor]


def compute_supervised_metric_stats(
    metric: SupervisedMetric,
    dataset: UIEBDataset,
) -> Stats:
    metric_values = torch.tensor(
        [
            metric(
                input.unsqueeze(0).to(_DEVICE), ground_truth.unsqueeze(0).to(_DEVICE)
            )
            for (input, ground_truth) in iter(dataset)
        ]
    )

    return Stats(*torch.std_mean(metric_values.abs()))


def fidelity(input: Tensor, ground_truth: Tensor) -> Tensor:
    """
    The fidelity loss is hard to calibrate since one would need to actually run the algorithm
    to estimate the gaussian kernel, and that is intertwined with the model execution, which
    in turn needs prior training and therefore having already calibrated the fidelity term.

    Therefore, and as a gross simplification, a Dirac delta is assumed for the kernel and the
    fidelity is minimized. The "fidelity term" is then computed for the input image (as it
    would be regularly) with sensible hyperparameters and then compared against the ground truth
    convolved with this Dirac Delta kernel.

    This turns out to be equivalent to computing the fidelity term and simply comparing it against the ground truth.
    """
    fidelity_term = compute_deterministic_components(input, 15, 8, 0.5, 0.001).fidelity

    return mse(fidelity_term, ground_truth)


unsupervised_metrics_and_losses = {
    "uciqe_minimizable": uciqe_minimizable,
    "total_variation": total_variation,
}
supervised_metrics_and_losses = {
    "psnr_minimizable": psnr_minimizable,
    "dssim": dssim,
    "mse": mse,
    "color_minimizable": color_minimizable,
    "fidelity": fidelity,
}


def _calibrate(
    dataset: UIEBDataset,
    unsupervised_metrics: Mapping[
        str, UnsupervisedMetric
    ] = unsupervised_metrics_and_losses,
    supervised_metrics: Mapping[str, SupervisedMetric] = supervised_metrics_and_losses,
) -> dict[str, Stats]:
    unsupervised_results = {
        metric_name: compute_unsupervised_metric_stats(metric, dataset)
        for metric_name, metric in unsupervised_metrics.items()
    }
    supervised_results = {
        metric_name: compute_supervised_metric_stats(metric, dataset)
        for metric_name, metric in supervised_metrics.items()
    }
    return unsupervised_results | supervised_results


def calibrate_metrics(
    uieb_path: Path,
):
    dataset = UIEBDataset(uieb_path)
    results = _calibrate(dataset)
    deserializable_results = {
        name: {"average": res.avg.item(), "standard_deviation": res.std.item()}
        for name, res in results.items()
    }

    with open("calibration_results.json", "w") as f:
        json.dump(deserializable_results, f)
