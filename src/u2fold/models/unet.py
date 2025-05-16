from dataclasses import dataclass
from typing import Optional

import torch

from u2fold.utils.track import tag

from .generic import Model, ModelConfig


@dataclass
class ConfigUNet(ModelConfig):
    ...

@tag("model/unet-like")
class UNet(Model[ConfigUNet]):
    def __init__(
        self, config: ConfigUNet, device: Optional[str] = None
    ) -> None:
        torch.nn.Module.__init__(self)
        pass
