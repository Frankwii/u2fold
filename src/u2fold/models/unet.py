from torch import nn

from u2fold.utils.track import tag


@tag("model/unet-like")
class UNet(nn.Module):
    """
    Note to self: Read this link before implementing the module:
    https://docs.pytorch.org/tutorials/prototype/skip_param_init.html.

    The constructor should take a `device` kwarg and pass it to some
    of its attributes.

    This is important so that it is possible to skip initialization.
    """
    ...
