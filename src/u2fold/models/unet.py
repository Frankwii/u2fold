from dataclasses import dataclass

import torch

from .generic import Model, ModelConfig
from u2fold.utils.track import tag


@dataclass
class ConfigUNet(ModelConfig):
    ...

@tag("model/unet-like")
class UNet(Model):
    """
    Note to self: Read this link before implementing the module:
    https://docs.pytorch.org/tutorials/prototype/skip_param_init.html.

    The constructor should take a `device` kwarg and pass it to some
    of its attributes.

    This is important so that it is possible to skip initialization.
    """
    def __init__(self, conf: ConfigUNet) -> None:
        ...
