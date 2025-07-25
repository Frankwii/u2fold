import torch
from torch import Tensor

def rescale_color(image: Tensor) -> Tensor:
    B, C = image.shape[:2]
    
    min = image.reshape(B, C, -1).min(dim=-1, keepdim=True).values.unsqueeze(-1)
    max = image.reshape(B, C, -1).max(dim=-1, keepdim=True).values.unsqueeze(-1)

    return (image - min)/(max - min)


def histogram_color_rescaling(image: Tensor, threshold: float) -> Tensor:
    B, C = image.shape[:2]

    quantiles = torch.quantile(
        input = image.reshape(B, C, -1),
        q = torch.Tensor([threshold, 1-threshold]).to(image.device),
        dim=-1
    ).view(2, B, C, 1, 1)

    return rescale_color(image.clamp(min=quantiles[0], max=quantiles[1]))
