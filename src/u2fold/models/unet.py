import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import torch

from u2fold.utils.track import tag

from .generic import Model, ModelConfig


@tag("config/model/unet")
@dataclass
class ConfigUNet(ModelConfig):
    sizes: list[int] = field(
        metadata={"desc": "List of hidden-state sizes of the UNet."}
    )

    def __post_init__(self):
        super().__post_init__()

        if len(self.sizes) <= 2:
            raise ValueError("Insufficient sizes for UNet")


@tag("model/unet")
class UNet(Model[ConfigUNet]):
    """Mimick the UNet architecture."""

    def __init__(
        self, config: ConfigUNet, device: Optional[str] = None
    ) -> None:
        torch.nn.Module.__init__(self)
