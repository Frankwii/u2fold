from functools import wraps
from typing import Callable

from torch import Tensor

from u2fold.neural_networks.metrics_and_losses.psnr import psnr_minimazible_calibrated
from u2fold.neural_networks.metrics_and_losses.ssim import dssim_calibrated
from u2fold.neural_networks.metrics_and_losses.uciqe import uciqe_minimizable_calibrated
from u2fold.orchestrate.train.auxiliary_methods_and_classes import LossComputingData
from u2fold.utils.dict_utils import shallow_dict_map

def wrap_unsupervised_metric(metric: Callable[[Tensor], Tensor]) -> Callable[[LossComputingData], float]:
    @wraps(metric)
    def wrapper(data: LossComputingData) -> float:
        return metric(data.output.radiance).mean().item()

    return wrapper

def wrap_supervised_metric(metric: Callable[[Tensor, Tensor], Tensor]) -> Callable[[LossComputingData], float]:
    @wraps(metric)
    def wrapper(data: LossComputingData) -> float:
        return metric(data.output.radiance, data.ground_truth).mean().item()

    return wrapper

unsupervised_minimizable_calibrated = {
    "uciqe_minimizable": uciqe_minimizable_calibrated,
}

supervised_minimizable_calibrated = {
    "dssim": dssim_calibrated,
    "psnr_minimizable": psnr_minimazible_calibrated
}

metrics_to_log = (
    shallow_dict_map(wrap_supervised_metric,  supervised_minimizable_calibrated) | 
    shallow_dict_map(wrap_unsupervised_metric, unsupervised_minimizable_calibrated)
)

def compute_metrics_to_log(data: LossComputingData) -> dict[str, float]:
    return shallow_dict_map(lambda f: f(data), metrics_to_log) 
