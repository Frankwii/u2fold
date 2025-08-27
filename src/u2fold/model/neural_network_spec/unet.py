from typing import Literal, final
from .generic import UNetLikeSpec

@final
class UNetSpec(UNetLikeSpec):  # pyright: ignore[reportUninitializedInstanceVariable]
    """Config for a UNet architecture"""
    name: Literal["unet"]
