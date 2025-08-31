from collections.abc import Sequence
from typing import Any, Callable, NamedTuple, cast

from torch import Tensor
from tqdm import tqdm
from u2fold.model.spec import U2FoldSpec
from u2fold.neural_networks.metrics_and_losses.psnr import psnr_minimazible_calibrated
from u2fold.neural_networks.metrics_and_losses.ssim import dssim_calibrated
from u2fold.neural_networks.metrics_and_losses.uciqe import uciqe_minimizable_calibrated
from u2fold.orchestrate import get_orchestrator
from u2fold.orchestrate.train import TrainOrchestrator

class MetricResults(NamedTuple):
    supervised: list[float]
    unsupervised: list[float]

def compute_metrics_for_orchestrator_model(
    orchestrator: TrainOrchestrator,
    supervised_metrics: Sequence[Callable[[Tensor, Tensor], Tensor]],
    unsupervised_metrics: Sequence[Callable[[Tensor], Tensor]],
) -> MetricResults:
    supervised_metric_values = [0.0] * len(supervised_metrics)
    unsupervised_metric_values = [0.0] * len(unsupervised_metrics)
    test_dataloader = orchestrator._dataloaders.test
    for input, ground_truth in tqdm(
        test_dataloader,
        desc="Training",
        total=len(test_dataloader)
    ):
        output = orchestrator.forward_pass(input).primal_variable_history[-1]
        for idx, metric in enumerate(supervised_metrics):
            supervised_metric_values[idx] += metric(output, ground_truth).item()

        for idx, metric in enumerate(unsupervised_metrics):
            unsupervised_metric_values[idx] += metric(output).item()

    return MetricResults(
        supervised = [x/len(test_dataloader) for x in supervised_metric_values],
        unsupervised = [x/len(test_dataloader) for x in unsupervised_metric_values]
    )

def measure_spec(spec: U2FoldSpec[Any]) -> float:  
    orchestrator = cast(TrainOrchestrator, get_orchestrator(spec))

    last_epoch_loss = cast(float, orchestrator.run().overall_loss)

    supervised_metrics = [dssim_calibrated, psnr_minimazible_calibrated]
    unsupervised_metrics = [uciqe_minimizable_calibrated]

    results = compute_metrics_for_orchestrator_model(orchestrator, supervised_metrics, unsupervised_metrics)

    supervised_score = sum(results.supervised) / len(results.supervised)
    unsupervised_score = sum(results.unsupervised) / len(results.unsupervised)

    final_score = 0.5 * last_epoch_loss + 0.25 * supervised_score * 0.25 * unsupervised_score

    return final_score
